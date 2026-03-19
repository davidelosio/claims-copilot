from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from src.models.features import build_features
from src.models.training import ComplexityPredictor
from src.ui.constants import DATA_DIR, EXTRACTIONS_PATH, MODEL_DIR


@st.cache_data
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and join all claims data."""
    claims = pd.read_csv(DATA_DIR / "claims.csv")
    policies = pd.read_csv(DATA_DIR / "policies.csv")
    vehicles = pd.read_csv(DATA_DIR / "vehicles.csv")
    policyholders = pd.read_csv(DATA_DIR / "policyholders.csv")
    labels = pd.read_csv(DATA_DIR / "claim_labels.csv")
    documents = pd.read_csv(DATA_DIR / "claim_documents.csv")

    merged = (
        claims
        .merge(policies, on="policy_id", suffixes=("", "_pol"))
        .merge(vehicles, on="vehicle_id", suffixes=("", "_veh"))
        .merge(policyholders, on="policyholder_id", suffixes=("", "_ph"))
        .merge(labels, on="claim_id", suffixes=("", "_lbl"))
    )
    return merged, documents


@st.cache_data
def load_extractions() -> dict[str, dict]:
    """Load LLM extraction results if available."""
    if not EXTRACTIONS_PATH.exists():
        return {}

    with open(EXTRACTIONS_PATH) as f:
        data = json.load(f)
    return {entry["claim_id"]: entry for entry in data}


@st.cache_data
def get_missing_model_artifacts() -> tuple[str, ...]:
    """Return model artifact filenames that are still missing."""
    return tuple(ComplexityPredictor.missing_artifacts(MODEL_DIR))


@st.cache_resource
def load_predictor() -> Optional[ComplexityPredictor]:
    """Load trained models when all artifacts are present."""
    if get_missing_model_artifacts():
        return None
    return ComplexityPredictor(model_dir=str(MODEL_DIR))


@st.cache_data
def load_features():
    """Load feature matrix."""
    extractions = str(EXTRACTIONS_PATH) if EXTRACTIONS_PATH.exists() else None
    return build_features(
        claims_path=str(DATA_DIR / "claims.csv"),
        policies_path=str(DATA_DIR / "policies.csv"),
        vehicles_path=str(DATA_DIR / "vehicles.csv"),
        policyholders_path=str(DATA_DIR / "policyholders.csv"),
        documents_path=str(DATA_DIR / "claim_documents.csv"),
        labels_path=str(DATA_DIR / "claim_labels.csv"),
        extractions_path=extractions,
    )


@st.cache_data
def load_metrics() -> dict:
    """Load model metrics."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def get_claim_prediction(
    claim_id: str,
    predictor: Optional[ComplexityPredictor],
) -> Optional[dict]:
    """Predict for a selected claim if features and models are available."""
    if predictor is None:
        return None

    features, _, _, _ = load_features()
    if claim_id not in features.index:
        return None

    return predictor.predict(features.loc[[claim_id]])[0]
