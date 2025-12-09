"""Visualization helpers for exploratory analysis and model interpretability."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_delay_distribution(df: pd.DataFrame, output_path: Optional[Path | str] = None) -> None:
    """Plot distribution of arrival delays."""

    plt.figure(figsize=(8, 4))
    df["ARR_DELAY"].dropna().hist(bins=50)
    plt.xlabel("Arrival Delay (minutes)")
    plt.ylabel("Count")
    plt.title("Arrival Delay Distribution")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info("Saved delay distribution plot to %s", output_path)
    else:
        plt.show()
    plt.close()


def plot_roc_curves(models, X_val, y_val, output_path: Optional[Path | str] = None) -> None:
    """Plot ROC curves for multiple models."""

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_val, y_val, name=name)
    plt.title("ROC Curves")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info("Saved ROC curves to %s", output_path)
    else:
        plt.show()
    plt.close()


__all__ = ["plot_delay_distribution", "plot_roc_curves"]
