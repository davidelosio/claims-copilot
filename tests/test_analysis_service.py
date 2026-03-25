from __future__ import annotations

from datetime import datetime, timezone

from src.api.repository import CopilotRepository
from src.api.schemas import (
    ClaimAnalysisRequest,
    CopilotFeedback,
    CopilotFeedbackCreate,
    CopilotOutput,
    CopilotOutputCreate,
)
from src.api.services import AnalysisService


class FakeCopilotRepository(CopilotRepository):
    def __init__(self):
        self._outputs: list[CopilotOutput] = []

    def create_output(self, payload: CopilotOutputCreate) -> CopilotOutput:
        output = CopilotOutput(
            output_id=len(self._outputs) + 1,
            created_at=datetime(2026, 3, 25, 9, 0, tzinfo=timezone.utc),
            **payload.model_dump(),
        )
        self._outputs.append(output)
        return output

    def get_latest_output(self, claim_id: str) -> CopilotOutput | None:
        matches = [output for output in self._outputs if output.claim_id == claim_id]
        if not matches:
            return None
        return matches[-1]

    def create_feedback(self, payload: CopilotFeedbackCreate) -> CopilotFeedback:
        raise NotImplementedError("Feedback persistence is not needed in this test")


def test_analysis_service_generates_next_actions_and_persists_snapshot():
    repository = FakeCopilotRepository()
    service = AnalysisService(repository)
    payload = ClaimAnalysisRequest.model_validate(
        {
            "claim": {
                "incident_type": "hit_and_run",
                "injuries": True,
                "injury_severity": "minor",
                "num_parties": 1,
                "damage_estimate": 5000,
                "vehicle_value": 12000,
                "status": "new",
                "policy_type": "comprehensive",
                "is_early_claim": False,
                "is_night_incident": True,
            },
            "documents": [{"doc_type": "photo_damage", "present": False}],
            "model_output": {
                "complexity_label": "complex",
                "complexity_confidence": 0.88,
                "recommended_queue": "specialist",
                "expected_handling_days": 14,
                "fraud_score": 0.67,
            },
            "extraction": {
                "summary": "Driver reports a night-time hit and run.",
                "facts": {"incident_type": "hit_and_run"},
                "missing_info": [
                    {
                        "field": "other vehicle plate",
                        "importance": "high",
                        "reason": "Needed for recovery attempts",
                    }
                ],
                "extraction_notes": "Police report number missing.",
            },
            "model_versions": {"predictor": "complexity_predictor_v1"},
        }
    )

    output = service.analyze_and_persist("CLM-900", payload)

    assert output.claim_id == "CLM-900"
    assert output.summary == "Driver reports a night-time hit and run."
    assert output.extracted_facts == {"incident_type": "hit_and_run"}
    assert output.fraud_label == "high"
    assert output.model_versions == {
        "predictor": "complexity_predictor_v1",
        "workflow_engine": "next_best_action_v1",
    }

    actions = {action["action"] for action in output.next_actions["actions"]}
    assert "Escalate to fraud review team" in actions
    assert "Request missing Police report" in actions
    assert "Request medical report" in actions
    assert output.next_actions["routing"]["queue"] == "fraud_review"
