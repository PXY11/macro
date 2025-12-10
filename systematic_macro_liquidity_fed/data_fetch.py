from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from fredapi import Fred
from pandas_datareader import data as pdr
fred = Fred(api_key="17e3eee9b99fa64f01bb2e192c655c59")  # 你原来已经有这一行就别动
from config import settings

# Fed "three buckets" + benchmark price we need from FRED.
THREE_BUCKETS_SERIES: Dict[str, str] = {
    "TGA": "WTREGEN",  # Treasury General Account
    "RRP": "RRPONTSYD",  # Overnight Reverse Repo
    "RESERVE_BALANCE": "WRESBAL",  # Reserve Balances with Federal Reserve Banks
}
SP500_SERIES = {"SP500": "SP500"}

# Macro-chain inputs that are available on FRED (funding stress, credit/liquidity, deleveraging proxy).
MACRO_CHAIN_SERIES: Dict[str, str] = {
    # Funding stress
    "TEDRATE": "TEDRATE",  # 3M Libor - 3M T-bill
    "CP_30D_AA": "DCPN30",  # 30-day AA nonfinancial CP
    "FED_FUNDS": "DFF",  # Effective Fed Funds
    "SOFR": "SOFR",  # Overnight secured rate
    # Market liquidity / credit risk
    "VIX": "VIXCLS",
    "HY_OAS": "BAMLH0A0HYM2",  # ICE BofA HY OAS
    "IG_OAS": "BAMLC0A0CM",  # ICE BofA IG OAS
    # Deleveraging proxy via CFTC positioning: not universally available on FRED; left empty by default
    # Users can extend here with valid CFTC series IDs (e.g., for CL/GC/ED) if needed.
}


class DataFetcher:
    """Thin wrapper that handles FRED access and local caching."""

    def __init__(self, api_key: str | None = None, data_dir: str | None = None):
        api_key = api_key or settings.fred_api_key
        if not api_key:
            raise ValueError(
                "FRED API key missing. Set FRED_API_KEY in env or config.Settings."
            )
        self.fred = Fred(api_key=api_key)

        data_dir = data_dir or settings.data_dir
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API

    def fetch_three_buckets(
        self,
        start: str | None = None,
        end: str | None = None,
        weekly_freq: str | None = None,
        include_sp500: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Pull the Fed three-bucket time series and (optionally) the SP500 index.
        Returns a weekly DataFrame indexed by date.
        """

        start = start or settings.start_date
        end = end or settings.end_date
        weekly_freq = weekly_freq or settings.weekly_freq

        cache_key = "with_px" if include_sp500 else "liq_only"
        cache_path = self._cache_path("three_buckets", weekly_freq, start, end, cache_key)

        if use_cache and cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        series_map = dict(THREE_BUCKETS_SERIES)
        if include_sp500:
            series_map.update(SP500_SERIES)

        df = self._download(series_map, start, end)
        df_weekly = df.resample(weekly_freq).last().dropna(how="all")

        if df_weekly.empty:
            raise ValueError("Weekly resampled data is empty. Check date range or API key.")

        if use_cache:
            df_weekly.to_csv(cache_path)

        return df_weekly

    def fetch_assets(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        weekly_freq: str | None = None,
        source: str = "stooq",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch asset close prices from pandas-datareader sources (default stooq, free).
        Returns a weekly-resampled price panel.
        """
        start = start or settings.start_date
        end = end or settings.end_date
        weekly_freq = weekly_freq or settings.weekly_freq

        tickers_sorted = sorted(tickers)
        cache_tag = "_".join(tickers_sorted)
        cache_path = self._cache_path(f"assets_{source}", weekly_freq, start, end, cache_tag)

        if use_cache and cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        frames = {}
        for t in tickers_sorted:
            print(f"[{source}] downloading {t}")
            if source == "av":
                df = pdr.DataReader(t, "av-daily", start=start, end=end, api_key=settings.alpha_vantage_key)
            else:
                df = pdr.DataReader(t, source, start=start, end=end)
            if df is None or len(df) == 0:
                raise ValueError(f"{source} returned no data for {t}")
            # Stooq returns reverse chronological; ensure ascending index
            df = df.sort_index()
            price_col = df.columns.intersection(["Close", "close"])
            if len(price_col) == 0:
                raise ValueError(f"No close price found for {t} from {source}")
            series = df[price_col[0]].copy()
            series.index = pd.to_datetime(series.index)
            frames[t] = series

        wide = pd.DataFrame(frames).sort_index()
        weekly = wide.resample(weekly_freq).last().dropna(how="all")

        if weekly.empty:
            raise ValueError("Weekly resampled asset prices are empty. Check tickers or source.")

        if use_cache:
            weekly.to_csv(cache_path)

        return weekly

    def fetch_macro_chain_inputs(
        self,
        start: str | None = None,
        end: str | None = None,
        weekly_freq: str | None = None,
        include_sp500: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Pull the baseline Fed buckets plus the macro-chain inputs that are available on FRED.
        This covers funding stress (TED, CP, Fed funds/SOFR), credit/liquidity (VIX, HY/IG OAS),
        and a deleveraging proxy (CFTC ES long/short).
        """
        start = start or settings.start_date
        end = end or settings.end_date
        weekly_freq = weekly_freq or settings.weekly_freq

        cache_key = "macro_chain_px" if include_sp500 else "macro_chain"
        cache_path = self._cache_path("macro_chain", weekly_freq, start, end, cache_key)

        if use_cache and cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        series_map = dict(THREE_BUCKETS_SERIES)
        series_map.update(MACRO_CHAIN_SERIES)
        if include_sp500:
            series_map.update(SP500_SERIES)

        df = self._download(series_map, start, end)
        df_weekly = df.resample(weekly_freq).last().dropna(how="all")

        if df_weekly.empty:
            raise ValueError("Weekly resampled macro-chain data is empty. Check date range or API key.")

        if use_cache:
            df_weekly.to_csv(cache_path)

        return df_weekly

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _download(self, series_map: Dict[str, str], start: str, end: str) -> pd.DataFrame:
        frames = {}
        for col, fred_id in series_map.items():
            print(f"[FRED] downloading {fred_id} -> {col}")
            s = self.fred.get_series(
                fred_id,
                observation_start=start,
                observation_end=end,
            )
            if s is None or len(s) == 0:
                raise ValueError(f"FRED returned no data for {fred_id}")
            series = pd.Series(s, name=col)
            series.index = pd.to_datetime(series.index)
            frames[col] = series

        return pd.DataFrame(frames).sort_index()

    def _cache_path(
        self,
        prefix: str,
        weekly_freq: str,
        start: str,
        end: str,
        suffix: str,
    ) -> Path:
        fname = f"{prefix}_{weekly_freq.replace('-', '')}_{start}_{end}_{suffix}.csv"
        return self.data_dir / fname
