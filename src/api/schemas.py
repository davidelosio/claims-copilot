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
