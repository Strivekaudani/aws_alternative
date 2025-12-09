"""Feature engineering utilities for flight delay prediction.

This module adds derived fields such as hour-of-day, day-of-week, and
binned distance, and optionally merges weather data provided by
``weather_api_utils``. Categorical features are encoded using
``OneHotEncoder`` from scikit-learn, returning both the processed
``DataFrame`` and the fitted encoder for reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from shared_constants import TARGET_COL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CATEGORICAL_COLS: List[str] = ["OP_CARRIER", "ORIGIN", "DEST"]
NUMERIC_COLS: List[str] = ["DEP_DELAY", "ARR_DELAY", "DISTANCE", "DEP_HOUR", "DAY_OF_WEEK"]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal columns such as hour-of-day and day-of-week."""

    df = df.copy()
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int)
    df["DAY_OF_WEEK"] = df["FLIGHT_DATE"].dt.dayofweek.astype(int)
    return df


def _bin_distance(df: pd.DataFrame, bin_size: int = 250) -> pd.DataFrame:
    """Create distance category bins to capture non-linear effects."""

    df = df.copy()
    df["DISTANCE_BIN"] = (df["DISTANCE"] // bin_size).astype(int)
    return df


def merge_weather(df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge flight data with weather features if provided."""

    if weather_df is None or weather_df.empty:
        logger.info("No weather data provided; skipping merge.")
        return df

    merge_keys = ["FLIGHT_DATE", "ORIGIN"]
    logger.info("Merging weather data on keys: %s", merge_keys)
    merged = df.merge(weather_df, on=merge_keys, how="left")
    return merged


def encode_features(df: pd.DataFrame, drop_first: bool = False) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """One-hot encode categorical columns and keep numerical columns untouched."""

    df = df.copy()
    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None),
                CATEGORICAL_COLS + ["DISTANCE_BIN"],
            ),
            ("numeric", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
    )
    feature_array = transformer.fit_transform(df)
    feature_names = transformer.get_feature_names_out()
    features = pd.DataFrame(feature_array.toarray() if hasattr(feature_array, "toarray") else feature_array)
    features.columns = feature_names
    features.index = df.index
    if TARGET_COL in df:
        features[TARGET_COL] = df[TARGET_COL].values
    logger.info("Encoded features shape: %s", features.shape)
    return features, transformer


def build_ml_ready_dataset(
    df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    output_path: Path | str | None = None,
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Create ML-ready dataset with engineered and encoded features.

    Parameters
    ----------
    df:
        Curated flight dataframe from ``01_load_flight_data``.
    weather_df:
        Optional weather dataframe aligned on FLIGHT_DATE and ORIGIN.
    output_path:
        Optional path to persist the ML-ready dataset as Parquet.
    drop_first:
        Whether to drop the first category during one-hot encoding to
        reduce multicollinearity.
    """

    df_augmented = _add_time_features(df)
    df_augmented = _bin_distance(df_augmented)
    df_augmented = merge_weather(df_augmented, weather_df)
    encoded_df, encoder = encode_features(df_augmented, drop_first=drop_first)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded_df.to_parquet(path, index=False)
        logger.info("ML-ready dataset saved to %s", path)
    return encoded_df, encoder


__all__ = [
    "build_ml_ready_dataset",
    "encode_features",
    "merge_weather",
    "CATEGORICAL_COLS",
    "NUMERIC_COLS",
    "TARGET_COL",
]
