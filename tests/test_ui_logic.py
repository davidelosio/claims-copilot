from __future__ import annotations

import pandas as pd

from src.ui.logic import build_nba_claim, collect_fraud_signals, filter_claims, is_night


def test_build_nba_claim_derives_timing_flags():
    row = pd.Series(
        {
            "incident_type": "collision",
            "injuries": True,
            "injury_severity": "minor",
            "num_parties": 2,
            "damage_estimate": 3500,
            "estimated_value": 10000,
            "status": "new",
            "policy_type": "comprehensive",
            "incident_date": "2024-02-20",
            "inception_date": "2024-01-10",
            "incident_time": "23:15:00",
        }
    )

    claim = build_nba_claim(row)

    assert claim["is_early_claim"] is True
    assert claim["is_night_incident"] is True
    assert claim["num_parties"] == 2


def test_collect_fraud_signals_and_filter_claims_work_on_plain_dataframes():
    row = pd.Series(
        {
            "damage_estimate": 8000,
            "estimated_value": 10000,
            "is_fraud": True,
            "incident_date": "2024-02-20",
            "inception_date": "2024-01-10",
            "incident_time": "01:30:00",
        }
    )
    signals = collect_fraud_signals(row)

    assert any("High damage ratio" in signal for signal in signals)
    assert any("Ground truth" in signal for signal in signals)
    assert any("Early claim" in signal for signal in signals)
    assert any("Night incident" in signal for signal in signals)
    assert is_night("01:30:00") is True

    df = pd.DataFrame(
        [
            {"claim_id": "1", "incident_type": "collision", "complexity": "simple", "is_fraud": False},
            {"claim_id": "2", "incident_type": "theft", "complexity": "complex", "is_fraud": True},
        ]
    )
    filtered = filter_claims(df, ["theft"], ["complex"], fraud_only=True)

    assert filtered["claim_id"].tolist() == ["2"]
