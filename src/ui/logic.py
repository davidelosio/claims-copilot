from __future__ import annotations

import pandas as pd


def filter_claims(
    df: pd.DataFrame,
    incident_filter: list[str],
    complexity_filter: list[str],
    fraud_only: bool,
) -> pd.DataFrame:
    """Apply sidebar filters to the claims dataframe."""
    filtered = df
    if incident_filter:
        filtered = filtered[filtered["incident_type"].isin(incident_filter)]
    if complexity_filter:
        filtered = filtered[filtered["complexity"].isin(complexity_filter)]
    if fraud_only:
        filtered = filtered[filtered["is_fraud"]]
    return filtered


def build_nba_claim(row: pd.Series) -> dict[str, object]:
    """Build the claim payload consumed by the next-best-action engine."""
    return {
        "incident_type": row["incident_type"],
        "injuries": bool(row.get("injuries")),
        "injury_severity": row.get("injury_severity", "none"),
        "num_parties": int(row.get("num_parties", 1)),
        "damage_estimate": float(row.get("damage_estimate", 0)),
        "vehicle_value": float(row.get("estimated_value", 1)),
        "status": row.get("status", "new"),
        "policy_type": row.get("policy_type", ""),
        "is_early_claim": _is_early_claim(row),
        "is_night_incident": is_night(row.get("incident_time")),
    }


def collect_fraud_signals(row: pd.Series) -> list[str]:
    """Return simple human-readable fraud signals for the detail panel."""
    signals: list[str] = []

    damage_ratio = row["damage_estimate"] / max(row.get("estimated_value", 1), 1)
    if damage_ratio > 0.5:
        signals.append(f"High damage ratio ({damage_ratio:.0%} of vehicle value)")

    if row.get("is_fraud"):
        signals.append("⚡ Ground truth: this IS a fraud case")

    if _is_early_claim(row):
        days = _days_since_inception(row)
        if days is not None:
            signals.append(f"Early claim ({days} days after inception)")

    incident_time = row.get("incident_time")
    if is_night(incident_time):
        signals.append(f"Night incident ({incident_time})")

    return signals


def is_night(incident_time) -> bool:
    """Check if incident time is between 22:00 and 06:00."""
    try:
        hour = int(str(incident_time).split(":")[0])
        return hour >= 22 or hour <= 5
    except (ValueError, IndexError, TypeError):
        return False


def _is_early_claim(row: pd.Series) -> bool:
    days = _days_since_inception(row)
    return days is not None and days <= 60


def _days_since_inception(row: pd.Series) -> int | None:
    try:
        return (pd.to_datetime(row["incident_date"]) - pd.to_datetime(row["inception_date"])).days
    except (KeyError, ValueError, TypeError):
        return None
