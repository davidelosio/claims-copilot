from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import app, get_repository
from src.api.repository import CopilotRepository
from src.api.schemas import (
    CopilotFeedback,
    CopilotFeedbackCreate,
    CopilotOutput,
    CopilotOutputCreate,
)


class FakeCopilotRepository(CopilotRepository):
    def __init__(self):
        self._outputs: list[CopilotOutput] = []
        self._feedback: list[CopilotFeedback] = []

    def create_output(self, payload: CopilotOutputCreate) -> CopilotOutput:
        output = CopilotOutput(
            output_id=len(self._outputs) + 1,
            created_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
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
        feedback = CopilotFeedback(
            feedback_id=len(self._feedback) + 1,
            created_at=datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
            **payload.model_dump(),
        )
        self._feedback.append(feedback)
        return feedback


def _build_client() -> tuple[TestClient, FakeCopilotRepository]:
    repo = FakeCopilotRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app), repo


def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_output_and_fetch_latest_output():
    client, _repo = _build_client()

    create_response = client.post(
        "/copilot/outputs",
        json={
            "claim_id": "CLM-100",
            "summary": "Collision reported with rear bumper damage.",
            "extracted_facts": {"incident_type": "collision"},
            "complexity_score": 0.62,
            "complexity_label": "medium",
            "fraud_score": 0.11,
            "fraud_label": "low",
            "next_actions": [{"action": "Request CAI form"}],
            "model_versions": {"extractor": "mistral:7b"},
        },
    )

    assert create_response.status_code == 201
    assert create_response.json()["claim_id"] == "CLM-100"

    latest_response = client.get("/claims/CLM-100/copilot/latest")

    assert latest_response.status_code == 200
    body = latest_response.json()
    assert body["output_id"] == 1
    assert body["summary"] == "Collision reported with rear bumper damage."


def test_analyze_claim_generates_and_persists_backend_owned_output():
    client, _repo = _build_client()

    response = client.post(
        "/claims/CLM-300/analysis",
        json={
            "claim": {
                "incident_type": "collision",
                "injuries": True,
                "injury_severity": "minor",
                "num_parties": 2,
                "damage_estimate": 2200,
                "vehicle_value": 12000,
                "status": "new",
                "policy_type": "comprehensive",
                "is_early_claim": False,
                "is_night_incident": False,
            },
            "documents": [],
            "model_output": {
                "complexity_label": "medium",
                "complexity_confidence": 0.74,
                "recommended_queue": "standard",
                "expected_handling_days": 8,
                "fraud_score": 0.18,
                "fraud_label": "low",
            },
            "extraction": {
                "summary": "Two-party collision with injuries reported.",
                "facts": {"incident_type": "collision"},
                "missing_info": [
                    {
                        "field": "other driver details",
                        "importance": "medium",
                        "reason": "Needed for liability review",
                    }
                ],
            },
            "model_versions": {"predictor": "complexity_predictor_v1"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["claim_id"] == "CLM-300"
    assert body["summary"] == "Two-party collision with injuries reported."
    assert body["model_versions"] == {
        "predictor": "complexity_predictor_v1",
        "workflow_engine": "next_best_action_v1",
    }
    action_names = {action["action"] for action in body["next_actions"]["actions"]}
    assert "Request missing Damage photos" in action_names
    assert "Request medical report" in action_names
    assert "Contact other party/parties for their version" in action_names

    latest_response = client.get("/claims/CLM-300/copilot/latest")

    assert latest_response.status_code == 200
    assert latest_response.json()["output_id"] == body["output_id"]


def test_get_latest_output_returns_404_when_missing():
    client, _repo = _build_client()

    response = client.get("/claims/CLM-404/copilot/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No copilot output found for claim CLM-404"


def test_create_feedback_persists_handler_signal():
    client, repo = _build_client()
    created_output = repo.create_output(
        CopilotOutputCreate(
            claim_id="CLM-200",
            summary="Theft claim with police report pending.",
        )
    )

    response = client.post(
        "/copilot/feedback",
        json={
            "output_id": created_output.output_id,
            "claim_id": "CLM-200",
            "handler_id": "handler-7",
            "feedback_type": "edited",
            "feedback_detail": {"reason": "Added a missing document request"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["feedback_id"] == 1
    assert body["feedback_type"] == "edited"
    assert body["feedback_detail"] == {"reason": "Added a missing document request"}
