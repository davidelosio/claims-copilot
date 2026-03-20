from __future__ import annotations

from src.models.training import get_feature_importance


class _FakeBooster:
    def __init__(self, score: dict[str, float]):
        self._score = score

    def get_score(self, importance_type: str = "gain") -> dict[str, float]:
        assert importance_type == "gain"
        return self._score


class _FakeEstimator:
    def __init__(self, score: dict[str, float]):
        self._booster = _FakeBooster(score)

    def get_booster(self) -> _FakeBooster:
        return self._booster


class _FakeCalibratedClassifier:
    def __init__(self, estimator: _FakeEstimator):
        self.estimator = estimator


class _FakeCalibratedModel:
    def __init__(self, score: dict[str, float]):
        self.calibrated_classifiers_ = [_FakeCalibratedClassifier(_FakeEstimator(score))]


def test_get_feature_importance_supports_named_features():
    model = _FakeEstimator({"incident_collision": 3.2, "police_report_yes": 1.4})

    result = get_feature_importance(
        model=model,
        feature_names=["incident_collision", "police_report_yes", "injuries_true"],
        top_n=3,
    )

    assert result == [
        {"feature": "incident_collision", "importance": 3.2},
        {"feature": "police_report_yes", "importance": 1.4},
    ]


def test_get_feature_importance_supports_calibrated_models_with_f_indexes():
    model = _FakeCalibratedModel({"f2": 0.8, "f0": 2.5})

    result = get_feature_importance(
        model=model,
        feature_names=["incident_collision", "police_report_yes", "injuries_true"],
        top_n=3,
    )

    assert result == [
        {"feature": "incident_collision", "importance": 2.5},
        {"feature": "injuries_true", "importance": 0.8},
    ]
