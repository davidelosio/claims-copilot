from __future__ import annotations

import json
import os
from typing import Optional
from urllib import error, parse, request

from src.serving.next_best_action import NextBestActionResult

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class ClaimsCopilotApiClient:
    """Very small HTTP client for the local FastAPI persistence service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0):
        self.base_url = (base_url or os.getenv("CLAIMS_COPILOT_API_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.timeout = timeout

    def get_latest_output(self, claim_id: str) -> Optional[dict]:
        try:
            return self._request(
                "GET",
                f"/claims/{parse.quote(claim_id, safe='')}/copilot/latest",
            )
        except error.HTTPError as exc:
            if exc.code != 404:
                raise
            return None

    def create_output(self, payload: dict) -> dict:
        return self._request("POST", "/copilot/outputs", payload=payload)

    def create_feedback(self, payload: dict) -> dict:
        return self._request("POST", "/copilot/feedback", payload=payload)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        req = request.Request(url=url, data=body, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except (error.HTTPError, error.URLError):
            raise


def build_copilot_output_payload(
    claim_id: str,
    extraction: Optional[dict],
    model_output: dict,
    nba_result: NextBestActionResult,
) -> dict:
    """Build a JSON-safe payload for saving the currently displayed copilot output."""
    return {
        "claim_id": claim_id,
        "summary": extraction.get("summary") if extraction else None,
        "extracted_facts": extraction.get("facts") if extraction else None,
        "complexity_score": model_output.get("complexity_confidence"),
        "complexity_label": model_output.get("complexity_label"),
        "fraud_score": model_output.get("fraud_score"),
        "fraud_label": model_output.get("fraud_label"),
        "next_actions": serialize_nba_result(nba_result),
        "model_versions": {
            "predictor": "complexity_predictor",
            "workflow_engine": "next_best_action_v1",
        },
    }


def build_feedback_payload(
    claim_id: str,
    output_id: int,
    feedback_type: str,
    detail_text: str,
) -> dict:
    """Build feedback payload tied to a saved copilot output."""
    detail_text = detail_text.strip()
    feedback_detail = {"note": detail_text} if detail_text else None
    return {
        "output_id": output_id,
        "claim_id": claim_id,
        "feedback_type": feedback_type,
        "feedback_detail": feedback_detail,
    }


def serialize_nba_result(nba_result: NextBestActionResult) -> dict:
    """Convert NextBestActionResult to plain JSON-safe data."""
    routing = None
    if nba_result.routing:
        routing = {
            "queue": nba_result.routing.queue,
            "confidence": nba_result.routing.confidence,
            "reasons": nba_result.routing.reasons,
            "override_warning": nba_result.routing.override_warning,
        }

    return {
        "claim_id": nba_result.claim_id,
        "summary_note": nba_result.summary_note,
        "routing": routing,
        "actions": [
            {
                "action": action.action,
                "priority": action.priority.value,
                "category": action.category.value,
                "reason": action.reason,
                "template_key": action.template_key,
                "auto_completable": action.auto_completable,
            }
            for action in nba_result.actions
        ],
    }
