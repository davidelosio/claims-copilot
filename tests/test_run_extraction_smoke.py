from __future__ import annotations

from click.testing import CliRunner

import scripts.run_extraction as run_extraction
from src.extraction.schemas import (
    ConfidenceLevel,
    ExtractionResult,
    ExtractedFacts,
    ExtractedField,
    IncidentType,
    InjurySeverity,
)


def _field(value: str | None = None) -> ExtractedField:
    return ExtractedField(value=value, confidence=ConfidenceLevel.HIGH, source_snippet=value)


def _fake_result(claim_id: str) -> ExtractionResult:
    return ExtractionResult(
        claim_id=claim_id,
        summary="smoke",
        facts=ExtractedFacts(
            incident_date=_field("01/01/2026"),
            incident_time=_field("09:00"),
            incident_location=_field("Via Roma"),
            incident_city=_field("Milano"),
            incident_type=IncidentType.COLLISION,
            num_parties=_field("2"),
            other_vehicle=_field("Fiat Panda"),
            damage_description=_field("Bumper dent"),
            damage_areas=_field("front bumper"),
            injuries_reported=False,
            injury_severity=InjurySeverity.NONE,
            injury_details=_field(None),
            police_report_mentioned=_field("yes"),
            police_report_number=_field("1234/2026"),
            witnesses_mentioned=_field("no"),
        ),
    )


def test_run_extraction_cli_smoke(monkeypatch, tmp_path):
    def fake_load_claims(csv_dir: str, n_sample: int | None = None, seed: int = 42):
        return [{"claim_id": "CLM-1", "description": "desc", "policy_context": {}}]

    class FakeExtractor:
        def __init__(self, model: str, timeout: float, num_batch: int | None):
            self.model = model
            self.timeout = timeout
            self.num_batch = num_batch

        def extract_batch(self, claims: list[dict], delay: float = 0.0):
            return [_fake_result(claims[0]["claim_id"])]

    def fake_evaluate(extractions: list[dict], claims_path: str):
        return {
            "incident_type_accuracy": 1,
            "injury_detection_accuracy": 1,
            "police_report_accuracy": 1,
            "city_accuracy": 1,
            "total": 1,
            "details": [],
            "incident_type_accuracy_pct": 100.0,
            "injury_detection_accuracy_pct": 100.0,
            "police_report_accuracy_pct": 100.0,
            "city_accuracy_pct": 100.0,
        }

    monkeypatch.setattr(run_extraction, "load_claims", fake_load_claims)
    monkeypatch.setattr(run_extraction, "ClaimExtractor", FakeExtractor)
    monkeypatch.setattr(run_extraction, "evaluate", fake_evaluate)

    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        run_extraction.main,
        ["--n-sample", "1", "--output-dir", str(out_dir), "--eval", "--no-batch"],
    )

    assert result.exit_code == 0
    assert (out_dir / "extractions.json").exists()
    assert (out_dir / "eval_results.json").exists()
