"""
Local LLM extraction pipeline using Ollama.

Uses JSON-mode prompting since local models don't support tool_use.

Recommended models:
- mistral-nemo:12b-instruct-2407-q4_K_M (24GB+ Apple Silicon)
- mistral:7b-instruct-q4_K_M (safer for 16GB)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import httpx

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

OUTPUT_SCHEMA_TEXT = """\
{
  "summary": "2-4 sentence summary for the claims handler",
  "facts": {
    "incident_date": {"value": "DD/MM/YYYY or null", "confidence": "high|medium|low|unknown", "source_snippet": "exact text or null"},
    "incident_time": {"value": "HH:MM or null", "confidence": "...", "source_snippet": "..."},
    "incident_location": {"value": "street/road or null", "confidence": "...", "source_snippet": "..."},
    "incident_city": {"value": "city name or null", "confidence": "...", "source_snippet": "..."},
    "incident_type": "collision|single_vehicle|hit_and_run|parking|theft|vandalism|weather|unknown",
    "num_parties": {"value": "number as string or null", "confidence": "...", "source_snippet": "..."},
    "other_vehicle": {"value": "make/model or null", "confidence": "...", "source_snippet": "..."},
    "damage_description": {"value": "description or null", "confidence": "...", "source_snippet": "..."},
    "damage_areas": {"value": "parts damaged or null", "confidence": "...", "source_snippet": "..."},
    "injuries_reported": true/false,
    "injury_severity": "none|minor|moderate|severe|unknown",
    "injury_details": {"value": "details or null", "confidence": "...", "source_snippet": "..."},
    "police_report_mentioned": {"value": "yes|no|null", "confidence": "...", "source_snippet": "..."},
    "police_report_number": {"value": "number or null", "confidence": "...", "source_snippet": "..."},
    "witnesses_mentioned": {"value": "yes|no|null", "confidence": "...", "source_snippet": "..."}
  },
  "missing_info": [
    {"field": "what's missing", "importance": "high|medium|low", "reason": "why it matters"}
  ],
  "language": "en|it|mixed",
  "extraction_notes": "any inconsistencies or concerns, or null"
}"""

SYSTEM_PROMPT = f"""\
You are a claims extraction assistant for an Italian motor insurance company.
You read claim descriptions (English or Italian) and extract structured facts.

Rules:
1. Extract ONLY what is explicitly stated. Never infer or assume.
2. For each field, include the exact source_snippet from the text.
3. If not present, set value to null and confidence to "unknown".
4. The summary must be in English regardless of description language.
5. Flag inconsistencies in extraction_notes.

You MUST respond with ONLY valid JSON matching this exact schema, nothing else:

{OUTPUT_SCHEMA_TEXT}

Respond with ONLY the JSON object. No markdown, no explanation, no backticks."""


class ClaimExtractor:
    """Extract structured facts from claim descriptions using a local Ollama model."""

    def __init__(
        self,
        model: str = "mistral-nemo:12b-instruct-2407-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: float = 240.0,
        max_retries: int = 2,
        temperature: float = 0.1,
        num_batch: int | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.num_batch = num_batch
        self._client = httpx.Client(timeout=timeout)

    def extract(
        self,
        claim_id: str,
        description: str,
        policy_context: Optional[dict] = None,
    ) -> ExtractionResult:
        """Extract facts from a single claim description."""
        user_message = self._build_user_message(description, policy_context)

        for attempt in range(self.max_retries + 1):
            try:
                raw_json = self._call_ollama(user_message)
                parsed = self._parse_json(raw_json)
                return self._to_result(claim_id, parsed)
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Extraction failed for {claim_id} (attempt {attempt+1}): {e}"
                    )
                    time.sleep(1)
                else:
                    logger.error(f"Extraction failed for {claim_id} after retries: {e}")
                    return self._fallback_result(claim_id, description)

        return self._fallback_result(claim_id, description)

    def extract_batch(
        self,
        claims: list[dict],
        delay: float = 0.0,
    ) -> list[ExtractionResult]:
        """Extract facts from multiple claims."""
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Extracting {i+1}/{len(claims)}: {claim['claim_id']}")
            result = self.extract(
                claim_id=claim["claim_id"],
                description=claim["description"],
                policy_context=claim.get("policy_context"),
            )
            results.append(result)
            if delay > 0 and i < len(claims) - 1:
                time.sleep(delay)
        return results

    def check_health(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (with or without tag) is available
            available = any(
                self.model in m or m.startswith(self.model)
                for m in models
            )
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {models}\n"
                    f"Run: ollama pull {self.model}"
                )
            return available
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama. Is it running?\n"
                "Start with: ollama serve"
            )
            return False


    def _call_ollama(self, user_message: str) -> str:
        """Call Ollama API and return raw text response."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": 1024,
        }
        if self.num_batch is not None:
            options["num_batch"] = self.num_batch

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",
            "options": options,
        }

        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _build_user_message(
        self, description: str, policy_context: Optional[dict],
    ) -> str:
        parts = [f"Claim description:\n\n{description}"]

        if policy_context:
            ctx_lines = []
            if "policy_type" in policy_context:
                ctx_lines.append(f"Policy type: {policy_context['policy_type']}")
            if "vehicle" in policy_context:
                v = policy_context["vehicle"]
                ctx_lines.append(
                    f"Insured vehicle: {v.get('make', '')} {v.get('model', '')} ({v.get('year', '')})"
                )
            if "policyholder_city" in policy_context:
                ctx_lines.append(f"Policyholder city: {policy_context['policyholder_city']}")
            if ctx_lines:
                parts.append("\nPolicy context:\n" + "\n".join(ctx_lines))

        parts.append("\nExtract all facts and respond with ONLY valid JSON.")
        return "\n".join(parts)

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from model output, handling common issues."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON from model: %s\nRaw model output:\n%s",
                e,
                text,
            )
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from model output: {text[:200]}...")

    def _to_result(self, claim_id: str, parsed: dict) -> ExtractionResult:
        """Convert parsed JSON dict to ExtractionResult."""
        facts_raw = parsed.get("facts", {})

        def parse_field(data: Any) -> ExtractedField:
            if isinstance(data, dict):
                return ExtractedField(
                    value=data.get("value"),
                    confidence=_safe_confidence(data.get("confidence", "unknown")),
                    source_snippet=data.get("source_snippet"),
                )
            if isinstance(data, str):
                return ExtractedField(value=data, confidence=ConfidenceLevel.MEDIUM)
            return ExtractedField()

        facts = ExtractedFacts(
            incident_date=parse_field(facts_raw.get("incident_date")),
            incident_time=parse_field(facts_raw.get("incident_time")),
            incident_location=parse_field(facts_raw.get("incident_location")),
            incident_city=parse_field(facts_raw.get("incident_city")),
            incident_type=_safe_incident_type(facts_raw.get("incident_type", "unknown")),
            num_parties=parse_field(facts_raw.get("num_parties")),
            other_vehicle=parse_field(facts_raw.get("other_vehicle")),
            damage_description=parse_field(facts_raw.get("damage_description")),
            damage_areas=parse_field(facts_raw.get("damage_areas")),
            injuries_reported=_safe_bool(facts_raw.get("injuries_reported")),
            injury_severity=_safe_injury_severity(facts_raw.get("injury_severity", "unknown")),
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
            for m in parsed.get("missing_info", [])
            if isinstance(m, dict)
        ]

        return ExtractionResult(
            claim_id=claim_id,
            summary=parsed.get("summary", ""),
            facts=facts,
            missing_info=missing,
            language=parsed.get("language", "en"),
            extraction_notes=parsed.get("extraction_notes"),
        )

    def _fallback_result(self, claim_id: str, description: str) -> ExtractionResult:
        """Return a minimal result when extraction fails."""
        return ExtractionResult(
            claim_id=claim_id,
            summary=f"[Extraction failed] Raw: {description[:200]}...",
            facts=ExtractedFacts(
                incident_date=ExtractedField(),
                incident_time=ExtractedField(),
                incident_location=ExtractedField(),
                incident_city=ExtractedField(),
                num_parties=ExtractedField(),
                other_vehicle=ExtractedField(),
                damage_description=ExtractedField(
                    value=description[:200], confidence=ConfidenceLevel.LOW,
                ),
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
            extraction_notes="Local model extraction failed. Manual review needed.",
        )


def _safe_confidence(val: Any) -> ConfidenceLevel:
    try:
        return ConfidenceLevel(str(val).lower().strip())
    except ValueError:
        return ConfidenceLevel.UNKNOWN


def _safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).lower().strip() in {"true", "1", "yes"}


def _safe_incident_type(val: Any) -> IncidentType:
    try:
        return IncidentType(str(val).lower().strip().replace(" ", "_"))
    except ValueError:
        return IncidentType.UNKNOWN


def _safe_injury_severity(val: Any) -> InjurySeverity:
    try:
        return InjurySeverity(str(val).lower().strip())
    except ValueError:
        return InjurySeverity.UNKNOWN
