from __future__ import annotations

import pytest

from src.extraction.pipeline import ClaimExtractor
from src.extraction.schemas import IncidentType, InjurySeverity


def test_parse_json_accepts_markdown_code_fence():
    extractor = ClaimExtractor(model="mistral:7b-instruct-q4_K_M")
    raw = """```json
{"summary":"ok","facts":{"incident_type":"collision","injuries_reported":true}}
```"""

    parsed = extractor._parse_json(raw)

    assert parsed["summary"] == "ok"
    assert parsed["facts"]["incident_type"] == "collision"


def test_parse_json_raises_on_invalid_payload():
    extractor = ClaimExtractor(model="mistral:7b-instruct-q4_K_M")

    with pytest.raises(ValueError):
        extractor._parse_json("not json")


def test_to_result_handles_bool_and_enum_fallbacks():
    extractor = ClaimExtractor(model="mistral:7b-instruct-q4_K_M")
    parsed = {
        "summary": "test",
        "facts": {
            "incident_type": "not-a-real-type",
            "injuries_reported": "false",
            "injury_severity": "bad-level",
        },
        "missing_info": [],
        "language": "en",
        "extraction_notes": None,
    }

    result = extractor._to_result("CLM-1", parsed)

    assert result.facts.incident_type == IncidentType.UNKNOWN
    assert result.facts.injuries_reported is False
    assert result.facts.injury_severity == InjurySeverity.UNKNOWN
