"""Utilities for loading and validating flight performance data.

This module focuses on reading raw CSV exports from the DOT or Kaggle
on-time performance datasets, performing light validation, and returning
a curated ``pandas.DataFrame`` ready for feature engineering.

Usage
-----
>>> from pathlib import Path
>>> from 01_load_flight_data import load_flight_data
>>> df = load_flight_data(Path("data/raw/flights_sample.csv"))
>>> print(df.head())

The function intentionally avoids API calls or heavy transformations to
keep it student-friendly and reproducible on laptops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RELEVANT_COLUMNS: List[str] = [
    "FLIGHT_DATE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DISTANCE",
]


def _ensure_columns(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    """Validate that all required columns exist.

    Parameters
    ----------
    df:
        Input dataframe.
    required:
        List of column names expected to be present.

    Raises
    ------
    ValueError
        If any required column is missing.
    """

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns to appropriate dtypes and strip whitespace."""

    df = df.copy()
    df["FLIGHT_DATE"] = pd.to_datetime(df["FLIGHT_DATE"], errors="coerce")
    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    integer_cols: Iterable[str] = ["OP_CARRIER_FL_NUM"]
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    numeric_cols: Iterable[str] = [
        "CRS_DEP_TIME",
        "DEP_TIME",
        "DEP_DELAY",
        "ARR_TIME",
        "ARR_DELAY",
        "DISTANCE",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["CANCELLED"] = df["CANCELLED"].fillna(0).astype(int)
    df["DIVERTED"] = df["DIVERTED"].fillna(0).astype(int)
    return df


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or invalid key fields."""

    df = df.dropna(subset=["FLIGHT_DATE", "ORIGIN", "DEST", "CRS_DEP_TIME"])
    df = df[df["CANCELLED"] == 0]
    df = df[df["DIVERTED"] == 0]
    df = df[df["ARR_DELAY"].notna()]
    return df


def load_flight_data(csv_path: Path | str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load and curate raw flight data.

    Parameters
    ----------
    csv_path:
        Path to the raw CSV file downloaded from DOT or Kaggle.
    columns:
        Optional custom subset of columns to retain. Defaults to ``RELEVANT_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        Curated dataframe with validated schema and basic cleaning applied.
    """

    selected_columns = list(columns) if columns else RELEVANT_COLUMNS
    logger.info("Loading flight data from %s", csv_path)
    df = pd.read_csv(csv_path, usecols=selected_columns, low_memory=False)
    _ensure_columns(df, selected_columns)
    df = _coerce_types(df)
    df = _filter_valid_rows(df)
    df["DELAYED"] = (df["ARR_DELAY"] >= 15).astype(int)
    logger.info("Loaded %d validated flight records", len(df))
    return df


def save_curated_data(df: pd.DataFrame, output_path: Path | str) -> None:
    """Persist curated flight data to disk in Parquet format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Curated data saved to %s", path)


__all__ = ["load_flight_data", "save_curated_data", "RELEVANT_COLUMNS"]
