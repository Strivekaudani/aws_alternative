"""Training script for flight delay classification.

Two baseline models are trained:
- Logistic Regression (with class balancing and scaling)
- RandomForestClassifier

Outputs
-------
- ``model_log_reg.pkl`` and ``model_random_forest.pkl``
- ``metrics.json`` summarizing evaluation metrics
- ``predictions.csv`` containing validation predictions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from shared_constants import TARGET_COL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y."""

    if TARGET_COL not in df:
        raise ValueError(f"Target column {TARGET_COL} not found in dataframe")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_logistic_regression_pipeline() -> Pipeline:
    """Create a pipeline with scaling and logistic regression."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1),
            ),
        ]
    )


def build_random_forest_model() -> RandomForestClassifier:
    """Instantiate a RandomForest model with sensible defaults."""

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )


def evaluate_model(model, X_val, y_val) -> Dict[str, float]:
    """Compute evaluation metrics for a fitted model."""

    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    metrics: Dict[str, float] = {
        "roc_auc": roc_auc_score(y_val, proba) if proba is not None else float("nan"),
    }
    report = classification_report(y_val, preds, output_dict=True)
    metrics.update({f"{label}_f1": scores["f1-score"] for label, scores in report.items() if "f1-score" in scores})
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    output_dir: Path | str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train two models and persist artifacts and metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = split_features_labels(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Training Logistic Regression model...")
    log_reg_pipeline = build_logistic_regression_pipeline()
    log_reg_pipeline.fit(X_train, y_train)
    log_reg_metrics = evaluate_model(log_reg_pipeline, X_val, y_val)
    joblib.dump(log_reg_pipeline, output_path / "model_log_reg.pkl")

    logger.info("Training Random Forest model...")
    rf_model = build_random_forest_model()
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_val, y_val)
    joblib.dump(rf_model, output_path / "model_random_forest.pkl")

    predictions = pd.DataFrame(
        {
            "true_label": y_val,
            "log_reg_pred": log_reg_pipeline.predict(X_val),
            "log_reg_proba": log_reg_pipeline.predict_proba(X_val)[:, 1],
            "rf_pred": rf_model.predict(X_val),
            "rf_proba": rf_model.predict_proba(X_val)[:, 1],
        }
    )
    predictions.to_csv(output_path / "predictions.csv", index=False)

    metrics = {
        "logistic_regression": log_reg_metrics,
        "random_forest": rf_metrics,
    }
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Metrics saved to %s", output_path / "metrics.json")
    return metrics


__all__ = [
    "train_and_evaluate",
    "split_features_labels",
    "build_logistic_regression_pipeline",
    "build_random_forest_model",
]
