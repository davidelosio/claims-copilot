from __future__ import annotations

from src.serving.next_best_action import ActionPriority, NextBestActionEngine


def test_next_best_action_generates_document_contact_and_medical_actions():
    engine = NextBestActionEngine()

    result = engine.generate(
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
            "complexity_label": "medium",
            "complexity_confidence": 0.8,
            "recommended_queue": "standard",
            "expected_handling_days": 7,
        },
        documents=[],
    )

    actions = {action.action: action for action in result.actions}

    assert actions["Request missing Damage photos"].priority == ActionPriority.URGENT
    assert actions["Request missing CAI form (joint statement)"].priority == ActionPriority.URGENT
    assert actions["Request medical report"].priority == ActionPriority.HIGH
    assert "Send acknowledgment to policyholder" in actions
    assert "Contact other party/parties for their version" in actions


def test_next_best_action_routes_high_fraud_claim_to_fraud_review():
    engine = NextBestActionEngine()

    result = engine.generate(
        claim_id="CLM-2",
        claim={
            "incident_type": "theft",
            "injuries": False,
            "num_parties": 1,
            "status": "in_progress",
            "policy_type": "third_party",
            "damage_estimate": 9000,
            "vehicle_value": 10000,
            "is_early_claim": True,
            "is_night_incident": True,
        },
        model_output={
            "fraud_score": 0.8,
            "complexity_label": "complex",
            "complexity_confidence": 0.55,
            "recommended_queue": "specialist",
            "expected_handling_days": 18,
        },
        documents=[],
        extraction={
            "missing_info": [{"field": "police report number", "importance": "high", "reason": "Needed"}],
            "extraction_notes": "Description conflicts with policy details",
        },
    )

    action_names = {action.action for action in result.actions}

    assert result.routing.queue == "fraud_review"
    assert result.routing.override_warning is not None
    assert "Escalate to fraud review team" in action_names
    assert "Verify coverage — policy may not cover this incident type" in action_names
    assert "Review flagged inconsistencies in claim description" in action_names
    assert "⚠️ Fraud risk 80%" in result.summary_note
