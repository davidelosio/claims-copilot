#!/usr/bin/env python3
"""
Train the complexity/routing, handling time, and fraud models.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --with-extractions data/extractions/extractions.json
    uv run python scripts/train_models.py --model-dir models --csv-dir data
"""

import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score

import click
import numpy as np

from src.models.features import build_features
from src.models.training import (
    get_feature_importance,
    time_based_split,
    train_complexity_model,
    train_fraud_model,
    train_handling_days_model,
)


@click.command(help="Train claims models")
@click.option("--csv-dir", default="data", show_default=True, help="Directory with generated CSVs")
@click.option("--model-dir", default="models", show_default=True, help="Where to save model artifacts")
@click.option(
    "--with-extractions",
    default=None,
    help="Path to LLM extractions JSON (optional enrichment)",
)
def main(csv_dir: str, model_dir: str, with_extractions: str | None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # FEATURE ENGINEERING

    print("\nBuilding features...")
    X, y_complexity, y_handling_days, y_fraud = build_features(
        claims_path=f"{csv_dir}/claims.csv",
        policies_path=f"{csv_dir}/policies.csv",
        vehicles_path=f"{csv_dir}/vehicles.csv",
        policyholders_path=f"{csv_dir}/policyholders.csv",
        documents_path=f"{csv_dir}/claim_documents.csv",
        labels_path=f"{csv_dir}/claim_labels.csv",
        extractions_path=with_extractions,
    )
    print(f"   Features: {X.shape[1]} columns, {X.shape[0]} claims")
    print(f"   Complexity distribution: {y_complexity.value_counts().to_dict()}")
    print(f"   Fraud rate: {y_fraud.mean():.1%}")

    # TIME-BASED SPLIT
    print("\nTime-based split...")
    train_ids, val_ids, test_ids = time_based_split(
        claims_path=f"{csv_dir}/claims.csv",
        features=X,
    )
    print(f"   Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    X_train, X_val, X_test = X.loc[train_ids], X.loc[val_ids], X.loc[test_ids]
    y_cx_train = y_complexity.loc[train_ids]
    y_cx_val = y_complexity.loc[val_ids]
    y_cx_test = y_complexity.loc[test_ids]
    y_hd_train = y_handling_days.loc[train_ids]
    y_hd_val = y_handling_days.loc[val_ids]
    y_hd_test = y_handling_days.loc[test_ids]
    y_fr_train = y_fraud.loc[train_ids]
    y_fr_val = y_fraud.loc[val_ids]
    y_fr_test = y_fraud.loc[test_ids]


    # TRAIN COMPLEXITY MODEL

    print("\nTraining complexity model...")
    cx_model, cx_metrics = train_complexity_model(X_train, y_cx_train, X_val, y_cx_val)
    print(f"   Val accuracy: {cx_metrics['accuracy']:.1%}")
    print(f"   Val F1 (macro): {cx_metrics['f1_macro']:.3f}")
    print(f"   Val F1 (weighted): {cx_metrics['f1_weighted']:.3f}")
    print(f"   Confusion matrix:\n{np.array(cx_metrics['confusion_matrix'])}")

    # Test set evaluation
    y_cx_test_enc = y_cx_test.map({"simple": 0, "medium": 1, "complex": 2})
    cx_test_pred = cx_model.predict(X_test)
    cx_test_acc = float(np.mean(cx_test_pred == y_cx_test_enc))
    print(f"   Test accuracy: {cx_test_acc:.1%}")
    cx_metrics["test_accuracy"] = round(cx_test_acc, 4)

    # Feature importance
    importance = get_feature_importance(cx_model, X.columns.tolist())
    print("\n   Top features:")
    for feat in importance[:10]:
        print(f"     {feat['feature']}: {feat['importance']:.4f}")

    # TRAIN HANDLING DAYS MODEL
    print("\nTraining handling days model...")
    hd_model, hd_metrics = train_handling_days_model(X_train, y_hd_train, X_val, y_hd_val)
    print(f"   Val MAE: {hd_metrics['mae']} days")
    print(f"   Val median AE: {hd_metrics['median_ae']} days")

    # Test
    hd_test_pred = hd_model.predict(X_test)
    hd_test_mae = round(float(np.mean(np.abs(y_hd_test - hd_test_pred))), 2)
    print(f"   Test MAE: {hd_test_mae} days")
    hd_metrics["test_mae"] = hd_test_mae

    # TRAIN FRAUD MODEL
    print("\nTraining fraud model...")
    fr_model, fr_metrics = train_fraud_model(X_train, y_fr_train, X_val, y_fr_val)
    print(f"   Val AUC-ROC: {fr_metrics['auc_roc']}")
    print(f"   Precision@5%: {fr_metrics['precision_at_5pct']:.1%}")
    print(f"   Precision@10%: {fr_metrics['precision_at_10pct']:.1%}")
    print(f"   Precision@20%: {fr_metrics['precision_at_20pct']:.1%}")

    # Test
    fr_test_proba = fr_model.predict_proba(X_test)[:, 1]

    fr_test_auc = round(roc_auc_score(y_fr_test, fr_test_proba), 4)
    print(f"   Test AUC-ROC: {fr_test_auc}")
    fr_metrics["test_auc_roc"] = fr_test_auc

    fraud_importance = get_feature_importance(fr_model, X.columns.tolist())
    print("\n   Top fraud features:")
    for feat in fraud_importance[:10]:
        print(f"     {feat['feature']}: {feat['importance']:.4f}")

    # SAVE ARTIFACTS
    print(f"\nSaving models to {model_dir}/...")

    with open(model_dir / "complexity_model.pkl", "wb") as f:
        pickle.dump(cx_model, f)
    with open(model_dir / "handling_days_model.pkl", "wb") as f:
        pickle.dump(hd_model, f)
    with open(model_dir / "fraud_model.pkl", "wb") as f:
        pickle.dump(fr_model, f)

    # Save feature names (for prediction-time alignment)
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(X.columns.tolist(), f)

    # Save all metrics
    all_metrics = {
        "complexity": cx_metrics,
        "handling_days": hd_metrics,
        "fraud": fr_metrics,
        "feature_importance_complexity": importance,
        "feature_importance_fraud": fraud_importance,
        "split_sizes": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print("   ✓ complexity_model.pkl")
    print("   ✓ handling_days_model.pkl")
    print("   ✓ fraud_model.pkl")
    print("   ✓ feature_names.json")
    print("   ✓ metrics.json")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
