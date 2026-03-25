from __future__ import annotations

from src.api.repository import CopilotRepository
from src.api.schemas import ClaimAnalysisRequest, CopilotOutput, CopilotOutputCreate
from src.serving.next_best_action import NextBestActionEngine, NextBestActionResult


class AnalysisService:
    """Application service for generating and persisting copilot analyses."""

    def __init__(
        self,
        repository: CopilotRepository,
        nba_engine: NextBestActionEngine | None = None,
    ):
        self.repository = repository
        self.nba_engine = nba_engine or NextBestActionEngine()

    def analyze_and_persist(
        self,
        claim_id: str,
        payload: ClaimAnalysisRequest,
    ) -> CopilotOutput:
        extraction = payload.extraction.model_dump(exclude_none=True) if payload.extraction else None
        nba_result = self.nba_engine.generate(
            claim_id=claim_id,
            claim=payload.claim.model_dump(),
            model_output=payload.model_output.model_dump(exclude_none=True),
            documents=[document.model_dump() for document in payload.documents],
            extraction=extraction,
        )

        output = CopilotOutputCreate(
            claim_id=claim_id,
            summary=payload.extraction.summary if payload.extraction else None,
            extracted_facts=payload.extraction.facts if payload.extraction else None,
            complexity_score=payload.model_output.complexity_confidence,
            complexity_label=payload.model_output.complexity_label,
            fraud_score=payload.model_output.fraud_score,
            fraud_label=payload.model_output.fraud_label or self._fraud_label(payload.model_output.fraud_score),
            next_actions=self._serialize_nba_result(nba_result),
            model_versions=self._build_model_versions(payload),
        )
        return self.repository.create_output(output)

    @staticmethod
    def _build_model_versions(payload: ClaimAnalysisRequest) -> dict:
        versions = dict(payload.model_versions or {})
        versions.setdefault("workflow_engine", "next_best_action_v1")
        return versions

    @staticmethod
    def _fraud_label(score: float) -> str:
        if score >= 0.5:
            return "high"
        if score >= 0.2:
            return "medium"
        return "low"

    @staticmethod
    def _serialize_nba_result(nba_result: NextBestActionResult) -> dict:
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
