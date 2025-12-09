"""Inference utilities for generating predictions and exporting to AWS S3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import boto3
import joblib
import pandas as pd

from shared_constants import TARGET_COL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(model_path: Path | str):
    """Load a serialized model from disk."""

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def generate_predictions(model, df: pd.DataFrame, output_path: Path | str) -> pd.DataFrame:
    """Generate predictions and persist to CSV and Parquet."""

    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df else df
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    preds = model.predict(X)
    results = df.copy()
    results["prediction"] = preds
    results["prediction_proba"] = proba

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path.with_suffix(".csv"), index=False)
    results.to_parquet(output_path.with_suffix(".parquet"), index=False)
    logger.info("Predictions saved to %s and %s", output_path.with_suffix(".csv"), output_path.with_suffix(".parquet"))
    return results


def upload_to_s3(
    file_path: Path | str,
    bucket: str,
    key_prefix: str,
    aws_region: Optional[str] = None,
) -> str:
    """Upload a local file to S3 using environment credentials."""

    session = boto3.session.Session(region_name=aws_region)
    s3 = session.client("s3")
    file_path = Path(file_path)
    key = f"{key_prefix.rstrip('/')}/{file_path.name}"
    s3.upload_file(str(file_path), bucket, key)
    logger.info("Uploaded %s to s3://%s/%s", file_path, bucket, key)
    return f"s3://{bucket}/{key}"


def export_predictions(
    model_path: Path | str,
    data_path: Path | str,
    output_path: Path | str,
    bucket: Optional[str] = None,
    key_prefix: str = "flight-delays/predictions",
    aws_region: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper to load data, predict, and export locally and to S3."""

    model = load_model(model_path)
    df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)
    predictions = generate_predictions(model, df, output_path)

    if bucket:
        upload_to_s3(Path(output_path).with_suffix(".csv"), bucket, key_prefix, aws_region)
        upload_to_s3(Path(output_path).with_suffix(".parquet"), bucket, key_prefix, aws_region)
    return predictions


__all__ = [
    "export_predictions",
    "generate_predictions",
    "upload_to_s3",
    "load_model",
]
