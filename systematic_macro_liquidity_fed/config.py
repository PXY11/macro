from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class Settings:
    """Central place for knobs used across the Fed buckets framework."""

    start_date: str = "2014-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    fred_api_key: str = os.getenv("FRED_API_KEY", "17e3eee9b99fa64f01bb2e192c655c59")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")

    # Default weekly sampling frequency for all downstream processing.
    weekly_freq: str = "W-WED"

    # Local cache directory (created lazily by the data fetcher).
    data_dir: str = "data"


settings = Settings()
