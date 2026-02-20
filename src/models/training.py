"""
Training pipeline for the complexity/routing model.

Key design decisions:
- Time-based train/val/test split (no leakage from future claims)
- XGBoost for classification (simple/medium/complex) + calibration
- Secondary: regression for handling_days, binary for fraud
- SHAP explanations for feature importance + per-prediction drivers
- Saves model artifacts for serving
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor


def time_based_split(
    claims_path: str,
    features: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split claim IDs by incident date (train → val → test).

    This is crucial: we simulate a real deployment where the model
    is trained on past claims and evaluated on future ones.

    """
    claims = pd.read_csv(claims_path)
    claims["incident_date"] = pd.to_datetime(claims["incident_date"])
    claims = claims.set_index("claim_id")

    # Only use claims that are in our feature set
    common_ids = features.index.intersection(claims.index)
    dates = claims.loc[common_ids, "incident_date"].sort_values()

    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_ids = dates.iloc[:train_end].index.tolist()
    val_ids = dates.iloc[train_end:val_end].index.tolist()
    test_ids = dates.iloc[val_end:].index.tolist()

    return train_ids, val_ids, test_ids


def _fit_xgb_with_best_n(
    estimator_cls,
    train_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    probe_n_estimators: int = 500,
    early_stopping_rounds: int = 30,
) -> tuple[object, int]:
    """Tune n_estimators with early stopping, then retrain with fixed best_n."""
    probe_params = {
        **train_params,
        "n_estimators": probe_n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
    }
    probe_model = estimator_cls(**probe_params)
    probe_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_n = probe_model.best_iteration + 1
    final_params = {**train_params, "n_estimators": best_n}
    model = estimator_cls(**final_params)
    model.fit(X_train, y_train, verbose=False)

    return model, best_n


def train_complexity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    calibrate: bool = True,
) -> tuple[object, dict]:
    """Train XGBoost classifier for complexity (simple/medium/complex).

    Returns:
        model: trained (and optionally calibrated) classifier
        metrics: dict of evaluation metrics on validation set
    """
    # Encode labels
    label_map = {"simple": 0, "medium": 1, "complex": 2}
    y_train_enc = y_train.map(label_map)
    y_val_enc = y_val.map(label_map)

    base_params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    base_model, best_n = _fit_xgb_with_best_n(
        estimator_cls=XGBClassifier,
        train_params=base_params,
        X_train=X_train,
        y_train=y_train_enc,
        X_val=X_val,
        y_val=y_val_enc,
    )

    # Calibration (isotonic, cross-validated on training data)
    if calibrate:
        model = CalibratedClassifierCV(
            base_model,
            cv=3,
            method="isotonic",
        )
        model.fit(X_train, y_train_enc)
    else:
        model = base_model

    # Evaluate
    y_pred = model.predict(X_val)
    inv_label_map = {v: k for k, v in label_map.items()}
    y_pred_labels = pd.Series(y_pred).map(inv_label_map)
    y_val_labels = y_val.reset_index(drop=True)

    metrics = {
        "accuracy": round(accuracy_score(y_val_enc, y_pred), 4),
        "f1_macro": round(f1_score(y_val_enc, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_val_enc, y_pred, average="weighted"), 4),
        "classification_report": classification_report(
            y_val_labels, y_pred_labels, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_val_enc, y_pred).tolist(),
        "label_map": label_map,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
        "best_iteration": best_n,
    }

    return model, metrics


def train_handling_days_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[object, dict]:
    """Train regression model for expected handling days."""
    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
    }
    model, _best_n = _fit_xgb_with_best_n(
        estimator_cls=XGBRegressor,
        train_params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    y_pred = model.predict(X_val)

    metrics = {
        "mae": round(mean_absolute_error(y_val, y_pred), 2),
        "median_ae": round(float(np.median(np.abs(y_val - y_pred))), 2),
        "mean_pred": round(float(y_pred.mean()), 2),
        "mean_actual": round(float(y_val.mean()), 2),
    }

    return model, metrics


def train_fraud_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[object, dict]:
    """Train binary classifier for fraud detection.

    Optimized for precision@k since only top-k get investigated.
    """

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "scale_pos_weight": scale_pos,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }
    model, _best_n = _fit_xgb_with_best_n(
        estimator_cls=XGBClassifier,
        train_params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Precision@k (top 5%, 10%, 20%)
    def precision_at_k(y_true, y_scores, k_pct):
        k = max(1, int(len(y_true) * k_pct))
        top_k_idx = np.argsort(y_scores)[-k:]
        return round(float(y_true.iloc[top_k_idx].mean()), 4)

    try:
        auc = round(roc_auc_score(y_val, y_pred_proba), 4)
    except ValueError:
        auc = None

    metrics = {
        "auc_roc": auc,
        "precision_at_5pct": precision_at_k(y_val, y_pred_proba, 0.05),
        "precision_at_10pct": precision_at_k(y_val, y_pred_proba, 0.10),
        "precision_at_20pct": precision_at_k(y_val, y_pred_proba, 0.20),
        "fraud_rate_train": round(float(y_train.mean()), 4),
        "fraud_rate_val": round(float(y_val.mean()), 4),
    }

    return model, metrics


def get_feature_importance(
    model: object,
    feature_names: list[str],
    top_n: int = 20,
) -> list[dict]:
    """Extract top feature importances from the model."""
    # For calibrated models, get the base estimator
    estimator = model
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV — get first calibrated classifier's estimator
        estimator = model.calibrated_classifiers_[0].estimator

    if not hasattr(estimator, "get_booster"):
        return []

    score = estimator.get_booster().get_score(importance_type="gain")
    importances = np.zeros(len(feature_names))
    for fname, imp in score.items():
        # XGBoost uses f0, f1, ... as default names
        idx = int(fname.replace("f", ""))
        if idx < len(importances):
            importances[idx] = imp

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in sorted_idx
        if importances[i] > 0
    ]


class ComplexityPredictor:
    """Load trained models and predict on new claims."""

    LABEL_MAP = {"simple": 0, "medium": 1, "complex": 2}
    INV_LABEL_MAP = {0: "simple", 1: "medium", 2: "complex"}
    QUEUE_MAP = {"simple": "fast_lane", "medium": "standard", "complex": "specialist"}

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.complexity_model = self._load("complexity_model.pkl")
        self.handling_model = self._load("handling_days_model.pkl")
        self.fraud_model = self._load("fraud_model.pkl")
        self.feature_names = self._load_json("feature_names.json")

    def predict(self, features: pd.DataFrame) -> list[dict]:
        """Predict complexity, handling days, and fraud for one or more claims.

        Returns list of dicts with:
            - complexity_label: simple/medium/complex
            - complexity_confidence: probability of predicted class
            - complexity_probs: {simple: p, medium: p, complex: p}
            - recommended_queue: fast_lane/standard/specialist
            - expected_handling_days: float
            - fraud_score: probability of fraud
            - fraud_label: low/medium/high
            - top_drivers: list of feature contributions
        """
        results = []

        # Ensure feature order matches training
        if self.feature_names:
            missing = set(self.feature_names) - set(features.columns)
            for col in missing:
                features[col] = 0
            features = features[self.feature_names]

        # Complexity
        complexity_probs = self.complexity_model.predict_proba(features)
        complexity_preds = self.complexity_model.predict(features)

        # Handling days
        handling_preds = self.handling_model.predict(features)

        # Fraud
        fraud_probs = self.fraud_model.predict_proba(features)[:, 1]

        for i in range(len(features)):
            pred_label = self.INV_LABEL_MAP[complexity_preds[i]]
            probs = {
                self.INV_LABEL_MAP[j]: round(float(complexity_probs[i][j]), 3)
                for j in range(3)
            }
            fraud_score = round(float(fraud_probs[i]), 3)

            results.append({
                "complexity_label": pred_label,
                "complexity_confidence": round(float(max(complexity_probs[i])), 3),
                "complexity_probs": probs,
                "recommended_queue": self.QUEUE_MAP[pred_label],
                "expected_handling_days": round(float(handling_preds[i]), 1),
                "fraud_score": fraud_score,
                "fraud_label": self._fraud_label(fraud_score),
            })

        return results

    @staticmethod
    def _fraud_label(score: float) -> str:
        if score >= 0.5:
            return "high"
        if score >= 0.2:
            return "medium"
        return "low"

    def _load(self, filename: str) -> object:
        path = self.model_dir / filename
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _load_json(self, filename: str) -> Optional[list]:
        path = self.model_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
