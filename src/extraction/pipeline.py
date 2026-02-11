"""
LLM-based extraction pipeline for claim descriptions.

Takes raw free-text descriptions and produces:
1. Structured fact card (with confidence + source snippets)
2. Concise handler summary
3. Missing information checklist

Uses Claude API with tool_use for reliable structured output.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import anthropic

from src.extraction.schemas import (
    ConfidenceLevel,
    ExtractionResult,
    ExtractedFacts,
    ExtractedField,
    IncidentType,
    InjurySeverity,
    MissingInfo,
)

logger = logging.getLogger(__name__)

# ============================================================
# EXTRACTION TOOL SCHEMA (for Claude tool_use)
# ============================================================

EXTRACTION_TOOL = {
    "name": "submit_extraction",
    "description": (
        "Submit the structured extraction results for a motor insurance claim. "
        "Extract all available facts from the claim description with confidence levels. "
        "For each field, include the exact source snippet from the text that supports it. "
        "If information is not present, set confidence to 'unknown' and value to null."
    ),
    "input_schema": {
        "type": "object",
        "required": ["summary", "facts", "missing_info", "language"],
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A concise 2-4 sentence summary of the claim for a claims handler. "
                    "Focus on: what happened, who was involved, what damage occurred, "
                    "and what immediate actions are needed."
                ),
            },
            "facts": {
                "type": "object",
                "required": [
                    "incident_date", "incident_time", "incident_location",
                    "incident_city", "incident_type", "num_parties",
                    "other_vehicle", "damage_description", "damage_areas",
                    "injuries_reported", "injury_severity", "injury_details",
                    "police_report_mentioned", "police_report_number",
                    "witnesses_mentioned",
                ],
                "properties": {
                    "incident_date": {"$ref": "#/$defs/extracted_field"},
                    "incident_time": {"$ref": "#/$defs/extracted_field"},
                    "incident_location": {"$ref": "#/$defs/extracted_field"},
                    "incident_city": {"$ref": "#/$defs/extracted_field"},
                    "incident_type": {
                        "type": "string",
                        "enum": [
                            "collision", "single_vehicle", "hit_and_run",
                            "parking", "theft", "vandalism", "weather", "unknown",
                        ],
                    },
                    "num_parties": {"$ref": "#/$defs/extracted_field"},
                    "other_vehicle": {"$ref": "#/$defs/extracted_field"},
                    "damage_description": {"$ref": "#/$defs/extracted_field"},
                    "damage_areas": {"$ref": "#/$defs/extracted_field"},
                    "injuries_reported": {"type": "boolean"},
                    "injury_severity": {
                        "type": "string",
                        "enum": ["none", "minor", "moderate", "severe", "unknown"],
                    },
                    "injury_details": {"$ref": "#/$defs/extracted_field"},
                    "police_report_mentioned": {"$ref": "#/$defs/extracted_field"},
                    "police_report_number": {"$ref": "#/$defs/extracted_field"},
                    "witnesses_mentioned": {"$ref": "#/$defs/extracted_field"},
                },
            },
            "missing_info": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["field", "importance", "reason"],
                    "properties": {
                        "field": {"type": "string"},
                        "importance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {"type": "string"},
                    },
                },
            },
            "language": {
                "type": "string",
                "enum": ["en", "it", "mixed"],
            },
            "extraction_notes": {
                "type": ["string", "null"],
                "description": (
                    "Flag any inconsistencies, vagueness, or suspicious patterns "
                    "in the description. null if nothing notable."
                ),
            },
        },
        "$defs": {
            "extracted_field": {
                "type": "object",
                "required": ["value", "confidence"],
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "description": "Extracted value, or null if not found",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "unknown"],
                    },
                    "source_snippet": {
                        "type": ["string", "null"],
                        "description": "Exact text span from the description supporting this extraction",
                    },
                },
            },
        },
    },
}

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """\
You are a claims extraction assistant for an Italian motor insurance company.

Your job is to read a claim description (which may be in English or Italian) and extract structured facts for the claims handler.

Rules:
1. Extract ONLY what is explicitly stated in the text. Never infer or assume.
2. For each field, provide the exact source snippet from the text.
3. If information is not present, set value to null and confidence to "unknown".
4. If information is ambiguous, set confidence to "low" and note it.
5. The summary should be written in English regardless of the description language.
6. Flag any inconsistencies (e.g., date/time conflicts, vague details with precise amounts).
7. For missing_info, focus on what a handler would need to process this claim efficiently.

Incident type classification:
- collision: two or more vehicles involved in an accident
- single_vehicle: vehicle hit obstacle/lost control, no other vehicle
- hit_and_run: other party fled the scene
- parking: damage found while vehicle was parked, no other party identified
- theft: vehicle stolen
- vandalism: intentional damage by unknown person
- weather: damage from hail, flooding, storms, etc.
- unknown: cannot determine from description

Be precise and concise. The handler will review your extraction."""


# ============================================================
# EXTRACTION PIPELINE
# ============================================================

class ClaimExtractor:
    """Extract structured facts from claim descriptions using Claude."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 2,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def extract(
        self,
        claim_id: str,
        description: str,
        policy_context: Optional[dict] = None,
    ) -> ExtractionResult:
        """Extract facts from a single claim description.

        Args:
            claim_id: Claim identifier
            description: Free-text claim description
            policy_context: Optional dict with policy/vehicle info to enrich extraction

        Returns:
            ExtractionResult with structured facts, summary, and missing info
        """
        user_message = self._build_user_message(description, policy_context)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=[EXTRACTION_TOOL],
                    tool_choice={"type": "tool", "name": "submit_extraction"},
                    messages=[{"role": "user", "content": user_message}],
                )

                # Parse tool use response
                tool_block = next(
                    (b for b in response.content if b.type == "tool_use"),
                    None,
                )
                if tool_block is None:
                    raise ValueError("No tool_use block in response")

                return self._parse_result(claim_id, tool_block.input)

            except anthropic.RateLimitError:
                if attempt < self.max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Extraction failed (attempt {attempt+1}): {e}")
                else:
                    logger.error(f"Extraction failed after {self.max_retries+1} attempts: {e}")
                    return self._fallback_result(claim_id, description)

        return self._fallback_result(claim_id, description)

    def extract_batch(
        self,
        claims: list[dict],
        delay: float = 0.5,
    ) -> list[ExtractionResult]:
        """Extract facts from multiple claims.

        Args:
            claims: List of dicts with 'claim_id', 'description', and optional 'policy_context'
            delay: Seconds between API calls (rate limiting)

        Returns:
            List of ExtractionResult
        """
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Extracting {i+1}/{len(claims)}: {claim['claim_id']}")
            result = self.extract(
                claim_id=claim["claim_id"],
                description=claim["description"],
                policy_context=claim.get("policy_context"),
            )
            results.append(result)
            if i < len(claims) - 1:
                time.sleep(delay)
        return results

    # ================================================================
    # INTERNALS
    # ================================================================

    def _build_user_message(
        self, description: str, policy_context: Optional[dict],
    ) -> str:
        parts = [f"<claim_description>\n{description}\n</claim_description>"]

        if policy_context:
            ctx_lines = []
            if "policy_type" in policy_context:
                ctx_lines.append(f"Policy type: {policy_context['policy_type']}")
            if "vehicle" in policy_context:
                v = policy_context["vehicle"]
                ctx_lines.append(f"Insured vehicle: {v.get('make', '')} {v.get('model', '')} ({v.get('year', '')})")
            if "policyholder_city" in policy_context:
                ctx_lines.append(f"Policyholder city: {policy_context['policyholder_city']}")
            if ctx_lines:
                ctx_text = "\n".join(ctx_lines)
                parts.append(f"\n<policy_context>\n{ctx_text}\n</policy_context>")

        parts.append(
            "\nExtract all facts from this claim description. "
            "Use the submit_extraction tool to provide structured results."
        )
        return "\n".join(parts)

    def _parse_result(self, claim_id: str, raw: dict[str, Any]) -> ExtractionResult:
        """Parse the raw tool_use output into an ExtractionResult."""
        facts_raw = raw.get("facts", {})

        def parse_field(data: Any) -> ExtractedField:
            if isinstance(data, dict):
                return ExtractedField(
                    value=data.get("value"),
                    confidence=ConfidenceLevel(data.get("confidence", "unknown")),
                    source_snippet=data.get("source_snippet"),
                )
            return ExtractedField()

        facts = ExtractedFacts(
            incident_date=parse_field(facts_raw.get("incident_date")),
            incident_time=parse_field(facts_raw.get("incident_time")),
            incident_location=parse_field(facts_raw.get("incident_location")),
            incident_city=parse_field(facts_raw.get("incident_city")),
            incident_type=IncidentType(facts_raw.get("incident_type", "unknown")),
            num_parties=parse_field(facts_raw.get("num_parties")),
            other_vehicle=parse_field(facts_raw.get("other_vehicle")),
            damage_description=parse_field(facts_raw.get("damage_description")),
            damage_areas=parse_field(facts_raw.get("damage_areas")),
            injuries_reported=facts_raw.get("injuries_reported", False),
            injury_severity=InjurySeverity(facts_raw.get("injury_severity", "unknown")),
            injury_details=parse_field(facts_raw.get("injury_details")),
            police_report_mentioned=parse_field(facts_raw.get("police_report_mentioned")),
            police_report_number=parse_field(facts_raw.get("police_report_number")),
            witnesses_mentioned=parse_field(facts_raw.get("witnesses_mentioned")),
        )

        missing = [
            MissingInfo(
                field=m.get("field", ""),
                importance=m.get("importance", "medium"),
                reason=m.get("reason", ""),
            )
            for m in raw.get("missing_info", [])
        ]

        return ExtractionResult(
            claim_id=claim_id,
            summary=raw.get("summary", ""),
            facts=facts,
            missing_info=missing,
            language=raw.get("language", "en"),
            extraction_notes=raw.get("extraction_notes"),
        )

    def _fallback_result(self, claim_id: str, description: str) -> ExtractionResult:
        """Return a minimal result when extraction fails."""
        return ExtractionResult(
            claim_id=claim_id,
            summary=f"[Extraction failed] Raw description: {description[:200]}...",
            facts=ExtractedFacts(
                incident_date=ExtractedField(),
                incident_time=ExtractedField(),
                incident_location=ExtractedField(),
                incident_city=ExtractedField(),
                num_parties=ExtractedField(),
                other_vehicle=ExtractedField(),
                damage_description=ExtractedField(value=description[:200], confidence=ConfidenceLevel.LOW),
                damage_areas=ExtractedField(),
                police_report_mentioned=ExtractedField(),
            ),
            missing_info=[
                MissingInfo(
                    field="all",
                    importance="high",
                    reason="Automated extraction failed — manual review required",
                ),
            ],
            extraction_notes="Extraction pipeline failed. Manual review needed.",
        )
