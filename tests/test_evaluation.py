from __future__ import annotations

import pandas as pd

from src.extraction.evaluation import evaluate


def test_evaluate_computes_expected_metrics(tmp_path):
    claims_path = tmp_path / "claims.csv"
    claims = pd.DataFrame([
        {
            "claim_id": "CLM-1",
            "incident_type": "collision",
            "injuries": True,
            "police_report": True,
            "incident_city": "Milano",
        },
        {
            "claim_id": "CLM-2",
            "incident_type": "theft",
            "injuries": False,
            "police_report": False,
            "incident_city": "Roma",
        },
    ])
    claims.to_csv(claims_path, index=False)

    extractions = [
        {
            "claim_id": "CLM-1",
            "facts": {
                "incident_type": "collision",
                "injuries_reported": True,
                "police_report_mentioned": {"value": "yes"},
                "incident_city": {"value": "Milano"},
            },
        },
        {
            "claim_id": "CLM-2",
            "facts": {
                "incident_type": "collision",
                "injuries_reported": True,
                "police_report_mentioned": {"value": "true"},
                "incident_city": {"value": "Roma centro"},
            },
        },
    ]

    result = evaluate(extractions, claims_path=str(claims_path))

    assert result["total"] == 2
    assert result["incident_type_accuracy"] == 1
    assert result["injury_detection_accuracy"] == 1
    assert result["police_report_accuracy"] == 1
    assert result["city_accuracy"] == 2
    assert result["incident_type_accuracy_pct"] == 50.0
    assert result["injury_detection_accuracy_pct"] == 50.0
    assert result["police_report_accuracy_pct"] == 50.0
    assert result["city_accuracy_pct"] == 100.0
    assert len(result["details"]) == 2


def test_evaluate_empty_input_returns_zeroed_metrics(tmp_path):
    claims_path = tmp_path / "claims.csv"
    pd.DataFrame(
        [{"claim_id": "CLM-1", "incident_type": "collision", "injuries": True, "police_report": True, "incident_city": "Milano"}]
    ).to_csv(claims_path, index=False)

    result = evaluate([], claims_path=str(claims_path))

    assert result["total"] == 0
    assert result["incident_type_accuracy"] == 0
    assert result["injury_detection_accuracy"] == 0
    assert result["police_report_accuracy"] == 0
    assert result["city_accuracy"] == 0
    assert result["details"] == []
