from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_claims(csv_dir: str, n_sample: int | None = None, seed: int = 42) -> list[dict]:
    """Load claims and minimal policy context for extraction."""
    claims = pd.read_csv(Path(csv_dir) / "claims.csv")
    policies = pd.read_csv(Path(csv_dir) / "policies.csv")
    vehicles = pd.read_csv(Path(csv_dir) / "vehicles.csv")
    policyholders = pd.read_csv(Path(csv_dir) / "policyholders.csv")

    merged = (
        claims
        .merge(policies[["policy_id", "policyholder_id", "vehicle_id", "policy_type"]], on="policy_id")
        .merge(vehicles[["vehicle_id", "make", "model", "year"]], on="vehicle_id")
        .merge(policyholders[["policyholder_id", "city"]], on="policyholder_id")
    )

    if n_sample is not None and n_sample < len(merged):
        merged = merged.sample(n=n_sample, random_state=seed)

    claims_with_context: list[dict] = []
    for row in merged.to_dict("records"):
        claims_with_context.append({
            "claim_id": row["claim_id"],
            "description": row["description"],
            "policy_context": {
                "policy_type": row["policy_type"],
                "vehicle": {
                    "make": row["make"],
                    "model": row["model"],
                    "year": int(row["year"]),
                },
                "policyholder_city": row["city"],
            },
        })

    return claims_with_context
