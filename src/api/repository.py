from __future__ import annotations

from abc import ABC, abstractmethod

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from src.api.schemas import (
    CopilotFeedback,
    CopilotFeedbackCreate,
    CopilotOutput,
    CopilotOutputCreate,
)


class CopilotRepository(ABC):
    """Persistence contract for copilot outputs and feedback."""

    @abstractmethod
    def create_output(self, payload: CopilotOutputCreate) -> CopilotOutput:
        """Persist a copilot output."""

    @abstractmethod
    def get_latest_output(self, claim_id: str) -> CopilotOutput | None:
        """Return the latest output for a claim, if any."""

    @abstractmethod
    def create_feedback(self, payload: CopilotFeedbackCreate) -> CopilotFeedback:
        """Persist handler feedback for a generated output."""


class PostgresCopilotRepository(CopilotRepository):
    """Thin psycopg repository.

    This keeps the first API slice simple:
    one class, plain SQL, no ORM, no session machinery.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def create_output(self, payload: CopilotOutputCreate) -> CopilotOutput:
        data = payload.model_dump()
        row = self._fetchone(
            """
            INSERT INTO copilot_outputs (
                claim_id,
                summary,
                extracted_facts,
                complexity_score,
                complexity_label,
                fraud_score,
                fraud_label,
                next_actions,
                model_versions
            )
            VALUES (
                %(claim_id)s,
                %(summary)s,
                %(extracted_facts)s,
                %(complexity_score)s,
                %(complexity_label)s,
                %(fraud_score)s,
                %(fraud_label)s,
                %(next_actions)s,
                %(model_versions)s
            )
            RETURNING
                output_id,
                claim_id,
                created_at,
                summary,
                extracted_facts,
                complexity_score,
                complexity_label,
                fraud_score,
                fraud_label,
                next_actions,
                model_versions
            """,
            self._adapt_json_fields(
                data,
                "extracted_facts",
                "next_actions",
                "model_versions",
            ),
        )
        return CopilotOutput.model_validate(row)

    def get_latest_output(self, claim_id: str) -> CopilotOutput | None:
        row = self._fetchone(
            """
            SELECT
                output_id,
                claim_id,
                created_at,
                summary,
                extracted_facts,
                complexity_score,
                complexity_label,
                fraud_score,
                fraud_label,
                next_actions,
                model_versions
            FROM copilot_outputs
            WHERE claim_id = %(claim_id)s
            ORDER BY created_at DESC, output_id DESC
            LIMIT 1
            """,
            {"claim_id": claim_id},
            commit=False,
        )
        if row is None:
            return None
        return CopilotOutput.model_validate(row)

    def create_feedback(self, payload: CopilotFeedbackCreate) -> CopilotFeedback:
        data = payload.model_dump()
        row = self._fetchone(
            """
            INSERT INTO copilot_feedback (
                output_id,
                claim_id,
                handler_id,
                feedback_type,
                feedback_detail
            )
            VALUES (
                %(output_id)s,
                %(claim_id)s,
                %(handler_id)s,
                %(feedback_type)s,
                %(feedback_detail)s
            )
            RETURNING
                feedback_id,
                output_id,
                claim_id,
                handler_id,
                feedback_type,
                feedback_detail,
                created_at
            """,
            self._adapt_json_fields(data, "feedback_detail"),
        )
        return CopilotFeedback.model_validate(row)

    def _fetchone(
        self,
        sql: str,
        params: dict,
        *,
        commit: bool = True,
    ) -> dict | None:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            if commit:
                conn.commit()
        return row

    @staticmethod
    def _adapt_json_fields(data: dict, *json_fields: str) -> dict:
        adapted = dict(data)
        for field in json_fields:
            value = adapted.get(field)
            adapted[field] = Json(value) if value is not None else None
        return adapted
