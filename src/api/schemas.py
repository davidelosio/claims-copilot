from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JsonValue = dict[str, Any] | list[Any]


class CopilotOutputCreate(BaseModel):
    claim_id: str
    summary: str | None = None
    extracted_facts: JsonValue | None = None
    complexity_score: float | None = Field(default=None, ge=0, le=1)
    complexity_label: str | None = None
    fraud_score: float | None = Field(default=None, ge=0, le=1)
    fraud_label: str | None = None
    next_actions: JsonValue | None = None
    model_versions: JsonValue | None = None


class CopilotOutput(CopilotOutputCreate):
    output_id: int
    created_at: datetime


class CopilotFeedbackCreate(BaseModel):
    output_id: int
    claim_id: str
    handler_id: str | None = None
    feedback_type: Literal["accepted", "edited", "rejected"]
    feedback_detail: JsonValue | None = None


class CopilotFeedback(CopilotFeedbackCreate):
    feedback_id: int
    created_at: datetime


class ClaimDocumentInput(BaseModel):
    doc_type: str
    present: bool = True


class ClaimAnalysisInput(BaseModel):
    incident_type: str
    injuries: bool = False
    injury_severity: str = "none"
    num_parties: int = Field(default=1, ge=1)
    damage_estimate: float = Field(default=0, ge=0)
    vehicle_value: float = Field(default=1, gt=0)
    status: str = "new"
    policy_type: str = ""
    is_early_claim: bool = False
    is_night_incident: bool = False


class MissingInfoInput(BaseModel):
    field: str
    importance: Literal["high", "medium", "low"] = "medium"
    reason: str


class ExtractionInput(BaseModel):
    summary: str | None = None
    facts: JsonValue | None = None
    missing_info: list[MissingInfoInput] = Field(default_factory=list)
    extraction_notes: str | None = None


class ModelOutputInput(BaseModel):
    complexity_label: str
    complexity_confidence: float = Field(ge=0, le=1)
    recommended_queue: str
    expected_handling_days: float = Field(ge=0)
    fraud_score: float = Field(ge=0, le=1)
    fraud_label: str | None = None


class ClaimAnalysisRequest(BaseModel):
    claim: ClaimAnalysisInput
    model_output: ModelOutputInput
    documents: list[ClaimDocumentInput] = Field(default_factory=list)
    extraction: ExtractionInput | None = None
    model_versions: JsonValue | None = None
