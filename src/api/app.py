from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException

from src.api.repository import CopilotRepository, PostgresCopilotRepository
from src.api.services import AnalysisService
from src.api.schemas import (
    ClaimAnalysisRequest,
    CopilotFeedback,
    CopilotFeedbackCreate,
    CopilotOutput,
    CopilotOutputCreate,
)

app = FastAPI(
    title="Claims Copilot API",
    version="0.1.0",
    summary="Persistence and retrieval endpoints for copilot outputs and feedback.",
)


def get_repository() -> CopilotRepository:
    dsn = os.getenv("CLAIMS_COPILOT_DSN")
    if not dsn:
        raise HTTPException(
            status_code=500,
            detail="CLAIMS_COPILOT_DSN is not set",
        )
    return PostgresCopilotRepository(dsn)


def get_analysis_service(
    repository: CopilotRepository = Depends(get_repository),
) -> AnalysisService:
    return AnalysisService(repository)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/copilot/outputs", response_model=CopilotOutput, status_code=201)
def create_output(
    payload: CopilotOutputCreate,
    repository: CopilotRepository = Depends(get_repository),
) -> CopilotOutput:
    return repository.create_output(payload)


@app.post("/claims/{claim_id}/analysis", response_model=CopilotOutput, status_code=201)
def analyze_claim(
    claim_id: str,
    payload: ClaimAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> CopilotOutput:
    return service.analyze_and_persist(claim_id, payload)


@app.get("/claims/{claim_id}/copilot/latest", response_model=CopilotOutput)
def get_latest_output(
    claim_id: str,
    repository: CopilotRepository = Depends(get_repository),
) -> CopilotOutput:
    output = repository.get_latest_output(claim_id)
    if output is None:
        raise HTTPException(
            status_code=404,
            detail=f"No copilot output found for claim {claim_id}",
        )
    return output


@app.post("/copilot/feedback", response_model=CopilotFeedback, status_code=201)
def create_feedback(
    payload: CopilotFeedbackCreate,
    repository: CopilotRepository = Depends(get_repository),
) -> CopilotFeedback:
    return repository.create_feedback(payload)
