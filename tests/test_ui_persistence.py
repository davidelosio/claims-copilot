from __future__ import annotations

import json

from src.serving.next_best_action import NextBestActionEngine
from src.ui.persistence import serialize_nba_result


def _build_nba_result():
    engine = NextBestActionEngine()
    return engine.generate(
        claim_id="CLM-1",
        claim={
            "incident_type": "collision",
            "injuries": True,
            "injury_severity": "minor",
            "num_parties": 2,
            "status": "new",
            "policy_type": "comprehensive",
            "damage_estimate": 2000,
            "vehicle_value": 12000,
        },
        model_output={
            "fraud_score": 0.1,
            "fraud_label": "low",
            "complexity_label": "medium",
            "complexity_confidence": 0.82,
            "recommended_queue": "standard",
            "expected_handling_days": 7,
        },
        documents=[],
        extraction={
            "missing_info": [{"field": "plate number", "importance": "medium", "reason": "Needed"}],
        },
    )


def test_serialize_nba_result_returns_json_safe_content():
    result = _build_nba_result()

    serialized = serialize_nba_result(result)

    assert serialized["claim_id"] == "CLM-1"
    assert serialized["routing"]["queue"] == "standard"
    assert all(isinstance(action["priority"], str) for action in serialized["actions"])
    assert all(isinstance(action["category"], str) for action in serialized["actions"])
    json.dumps(serialized)
