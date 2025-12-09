"""Utility functions for fetching weather data from OpenWeatherMap.

The API key is read from the ``OPENWEATHER_API_KEY`` environment variable.
This module intentionally returns lightweight pandas DataFrames keyed by
``FLIGHT_DATE`` and ``ORIGIN`` to merge with flight data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_weather_for_airport(airport_code: str, date: datetime) -> Optional[Dict]:
    """Fetch weather for a single airport and date."""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set; skipping weather fetch.")
        return None

    params = {
        "q": airport_code,
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return {
        "FLIGHT_DATE": pd.to_datetime(date).normalize(),
        "ORIGIN": airport_code,
        "temp_c": payload.get("main", {}).get("temp"),
        "humidity": payload.get("main", {}).get("humidity"),
        "weather_main": payload.get("weather", [{}])[0].get("main"),
        "wind_speed": payload.get("wind", {}).get("speed"),
    }


def fetch_weather_batch(airports: List[str], date: datetime) -> pd.DataFrame:
    """Fetch weather for multiple airports and compile into a dataframe."""

    records = []
    for airport in airports:
        try:
            record = fetch_weather_for_airport(airport, date)
            if record:
                records.append(record)
        except requests.RequestException as exc:
            logger.error("Weather fetch failed for %s: %s", airport, exc)
    return pd.DataFrame(records)


__all__ = ["fetch_weather_batch", "fetch_weather_for_airport"]
