"""
Feature engineering for the complexity/routing model.

Builds features from:
- Structured claim fields (available at claim creation time)
- Policy / vehicle / policyholder context
- Document metadata (what's present vs missing)
- Optional: LLM extraction output (if available)

Key principle: TIME CORRECTNESS
Every feature must be computable at the moment the claim arrives.
We never use future information (settlement amount, handling days, etc).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_tables(
    claims_path: str,
    policies_path: str,
    vehicles_path: str,
    policyholders_path: str,
    documents_path: str,
    labels_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_csv(claims_path)
    policies = pd.read_csv(policies_path)
    vehicles = pd.read_csv(vehicles_path)
    policyholders = pd.read_csv(policyholders_path)
    documents = pd.read_csv(documents_path)
    labels = pd.read_csv(labels_path)
    return claims, policies, vehicles, policyholders, documents, labels


def _parse_dates(claims: pd.DataFrame, policies: pd.DataFrame, policyholders: pd.DataFrame) -> None:
    claims["incident_date"] = pd.to_datetime(claims["incident_date"])
    claims["created_at"] = pd.to_datetime(claims["created_at"])
    policies["inception_date"] = pd.to_datetime(policies["inception_date"])
    policies["expiry_date"] = pd.to_datetime(policies["expiry_date"])
    policyholders["date_of_birth"] = pd.to_datetime(policyholders["date_of_birth"])


def _build_claim_context(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    vehicles: pd.DataFrame,
    policyholders: pd.DataFrame,
) -> pd.DataFrame:
    return (
        claims
        .merge(policies, on="policy_id", suffixes=("", "_pol"))
        .merge(vehicles, on="vehicle_id", suffixes=("", "_veh"))
        .merge(policyholders, on="policyholder_id", suffixes=("", "_ph"))
    )


def _join_dummies(
    features: pd.DataFrame,
    series: pd.Series,
    index: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    dummies = pd.get_dummies(series, prefix=prefix)
    dummies.index = index
    return features.join(dummies)


def _to_top_n_with_other(series: pd.Series, n: int = 10, other_label: str = "other") -> pd.Series:
    top_values = series.value_counts().head(n).index.tolist()
    return series.where(series.isin(top_values), other_label)


def _add_claim_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features = _join_dummies(features, df["incident_type"], df["claim_id"], "incident")

    features["num_parties"] = df["num_parties"]
    features["injuries"] = df["injuries"].astype(int)
    features["police_report"] = df["police_report"].astype(int)
    features["damage_estimate"] = df["damage_estimate"]

    severity_map = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    features["injury_severity_ord"] = (
        df["injury_severity"].map(severity_map).fillna(0).astype(int)
    )

    features["incident_hour"] = pd.to_datetime(
        df["incident_time"].astype(str), format="%H:%M:%S", errors="coerce"
    ).dt.hour.fillna(12).astype(int)
    features["incident_dow"] = df["incident_date"].dt.dayofweek
    features["incident_month"] = df["incident_date"].dt.month
    features["is_night_incident"] = (
        (features["incident_hour"] >= 22) | (features["incident_hour"] <= 5)
    ).astype(int)
    features["is_weekend_incident"] = (features["incident_dow"] >= 5).astype(int)

    return features


def _add_policy_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features = _join_dummies(features, df["policy_type"], df["claim_id"], "policy")

    features["annual_premium"] = df["annual_premium"]
    features["deductible"] = df["deductible"]
    features["coverage_limit"] = df["coverage_limit"]
    features["days_since_inception"] = (
        (df["incident_date"] - df["inception_date"]).dt.days
    )
    features["is_early_claim"] = (features["days_since_inception"] <= 60).astype(int)

    return features


def _add_vehicle_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features["vehicle_value"] = df["estimated_value"]
    features["vehicle_age"] = (2024 - df["year"])
    features["damage_ratio"] = np.where(
        df["estimated_value"] > 0,
        df["damage_estimate"] / df["estimated_value"],
        0,
    )

    p75 = df["estimated_value"].quantile(0.75)
    features["is_high_value_vehicle"] = (df["estimated_value"] > p75).astype(int)

    features = _join_dummies(features, df["fuel_type"], df["claim_id"], "fuel")

    make_col = _to_top_n_with_other(df["make"], n=10, other_label="other")
    features = _join_dummies(features, make_col, df["claim_id"], "make")

    return features


def _add_policyholder_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features["policyholder_age"] = (
        (df["incident_date"] - df["date_of_birth"]).dt.days / 365.25
    ).astype(int)
    features["driving_experience"] = (
        df["incident_date"].dt.year - df["driving_license_year"]
    )
    features["gender_M"] = (df["gender"] == "M").astype(int)

    prov_col = _to_top_n_with_other(df["province"], n=10, other_label="other")
    features = _join_dummies(features, prov_col, df["claim_id"], "prov")

    features["location_mismatch"] = (
        df["incident_province"] != df["province"]
    ).astype(int)

    return features


def _add_document_features(features: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    doc_agg = documents.groupby("claim_id").agg(
        n_docs_total=("document_id", "count"),
        n_docs_present=("present", "sum"),
        n_docs_missing=("present", lambda x: (~x).sum()),
    )

    features = features.join(doc_agg, how="left")
    count_cols = ["n_docs_total", "n_docs_present", "n_docs_missing"]
    features[count_cols] = features[count_cols].fillna(0).astype(int)
    features["doc_completeness"] = np.where(
        features["n_docs_total"] > 0,
        features["n_docs_present"] / features["n_docs_total"],
        0,
    )

    doc_pivot = documents.pivot_table(
        index="claim_id",
        columns="doc_type",
        values="present",
        aggfunc="max",
        fill_value=False,
    ).astype(int)
    doc_pivot.columns = [f"has_doc_{c}" for c in doc_pivot.columns]
    features = features.join(doc_pivot, how="left")

    doc_cols = [c for c in features.columns if c.startswith("has_doc_")]
    features[doc_cols] = features[doc_cols].fillna(0).astype(int)

    return features


def _add_text_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    features["description_length"] = df["description"].str.len()
    features["description_word_count"] = df["description"].str.split().str.len()

    italian_markers = ["incidente", "stavo", "percorrendo", "danni", "veicolo", "circa"]
    features["is_italian_text"] = (
        df["description"]
        .str.lower()
        .apply(lambda x: int(any(marker in str(x) for marker in italian_markers)))
    )
    return features


def build_features(
    claims_path: str = "data/claims.csv",
    policies_path: str = "data/policies.csv",
    vehicles_path: str = "data/vehicles.csv",
    policyholders_path: str = "data/policyholders.csv",
    documents_path: str = "data/claim_documents.csv",
    labels_path: str = "data/claim_labels.csv",
    extractions_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build feature matrix + targets.

    Returns:
        X: feature DataFrame (one row per claim)
        y_complexity: target labels (simple/medium/complex)
        y_handling_days: regression target
        y_fraud: binary fraud target
    """

    claims, policies, vehicles, policyholders, documents, labels = _load_tables(
        claims_path,
        policies_path,
        vehicles_path,
        policyholders_path,
        documents_path,
        labels_path,
    )
    _parse_dates(claims, policies, policyholders)
    df = _build_claim_context(claims, policies, vehicles, policyholders)

    features = pd.DataFrame(index=df["claim_id"])
    features = _add_claim_features(features, df)
    features = _add_policy_features(features, df)
    features = _add_vehicle_features(features, df)
    features = _add_policyholder_features(features, df)
    features = _add_document_features(features, documents)
    features = _add_text_features(features, df)

    if extractions_path and Path(extractions_path).exists():
        features = _add_extraction_features(features, extractions_path)

    labels_indexed = labels.set_index("claim_id")
    y_complexity = labels_indexed.loc[features.index, "complexity"]
    y_handling_days = labels_indexed.loc[features.index, "handling_days"]
    y_fraud = labels_indexed.loc[features.index, "is_fraud"].astype(int)

    # Clean up any remaining NaN
    features = features.fillna(0)

    return features, y_complexity, y_handling_days, y_fraud


def _add_extraction_features(features: pd.DataFrame, extractions_path: str) -> pd.DataFrame:
    """Add features derived from LLM extraction output."""
    with open(extractions_path) as f:
        extractions = json.load(f)

    extraction_map = {e["claim_id"]: e for e in extractions}

    # Number of missing info items flagged by LLM
    features["llm_n_missing"] = features.index.map(
        lambda cid: len(extraction_map.get(cid, {}).get("missing_info", []))
    ).fillna(0).astype(int)

    # Number of high-importance missing items
    features["llm_n_missing_high"] = features.index.map(
        lambda cid: sum(
            1 for m in extraction_map.get(cid, {}).get("missing_info", [])
            if m.get("importance") == "high"
        )
    ).fillna(0).astype(int)

    # Has extraction notes (inconsistencies flagged)
    features["llm_has_notes"] = features.index.map(
        lambda cid: int(bool(extraction_map.get(cid, {}).get("extraction_notes")))
    ).fillna(0).astype(int)

    # Average confidence of extracted fields
    def avg_confidence(cid):
        ext = extraction_map.get(cid, {})
        facts = ext.get("facts", {})
        conf_map = {"high": 1.0, "medium": 0.66, "low": 0.33, "unknown": 0.0}
        confs = []
        for k, v in facts.items():
            if isinstance(v, dict) and "confidence" in v:
                confs.append(conf_map.get(v["confidence"], 0))
        return np.mean(confs) if confs else 0

    features["llm_avg_confidence"] = features.index.map(avg_confidence)

    return features


def get_feature_names() -> dict[str, list[str]]:
    """Return feature groups for documentation / SHAP analysis."""
    return {
        "claim": [
            "num_parties", "injuries", "police_report", "damage_estimate",
            "injury_severity_ord", "incident_hour", "incident_dow", "incident_month",
            "is_night_incident", "is_weekend_incident", "description_length",
            "description_word_count", "is_italian_text",
        ],
        "incident_type": [f"incident_{t}" for t in [
            "collision", "hit_and_run", "parking", "single_vehicle",
            "theft", "vandalism", "weather",
        ]],
        "policy": [
            "annual_premium", "deductible", "coverage_limit",
            "days_since_inception", "is_early_claim",
        ],
        "vehicle": [
            "vehicle_value", "vehicle_age", "damage_ratio", "is_high_value_vehicle",
        ],
        "policyholder": [
            "policyholder_age", "driving_experience", "gender_M", "location_mismatch",
        ],
        "documents": [
            "n_docs_total", "n_docs_present", "n_docs_missing", "doc_completeness",
        ],
        "llm": [
            "llm_n_missing", "llm_n_missing_high", "llm_has_notes", "llm_avg_confidence",
        ],
    }
