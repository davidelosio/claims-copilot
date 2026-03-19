from __future__ import annotations

import json
import pickle

import pytest

from src.models.training import ComplexityPredictor


def _write_complete_model_dir(tmp_path):
    for filename in [
        "complexity_model.pkl",
        "handling_days_model.pkl",
        "fraud_model.pkl",
    ]:
        with open(tmp_path / filename, "wb") as f:
            pickle.dump({"artifact": filename}, f)

    with open(tmp_path / "feature_names.json", "w") as f:
        json.dump(["feature_a", "feature_b"], f)


def test_missing_artifacts_reports_expected_files(tmp_path):
    missing = ComplexityPredictor.missing_artifacts(tmp_path)

    assert set(missing) == set(ComplexityPredictor.REQUIRED_ARTIFACTS)


def test_predictor_raises_when_artifacts_are_incomplete(tmp_path):
    _write_complete_model_dir(tmp_path)
    (tmp_path / "fraud_model.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="fraud_model.pkl"):
        ComplexityPredictor(model_dir=str(tmp_path))


def test_predictor_loads_when_all_artifacts_exist(tmp_path):
    _write_complete_model_dir(tmp_path)

    predictor = ComplexityPredictor(model_dir=str(tmp_path))

    assert predictor.complexity_model == {"artifact": "complexity_model.pkl"}
    assert predictor.handling_model == {"artifact": "handling_days_model.pkl"}
    assert predictor.fraud_model == {"artifact": "fraud_model.pkl"}
    assert predictor.feature_names == ["feature_a", "feature_b"]
