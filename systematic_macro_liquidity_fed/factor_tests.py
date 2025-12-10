from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_forward_returns(price: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Forward returns for a given horizon (e.g., horizon=1 => next-period return).
    """
    price = price.sort_index()
    return price.pct_change(periods=horizon).shift(-horizon)


def _corr_t_stat(r: float, n: int) -> float:
    """
    T-statistic for a correlation coefficient with n samples.
    """
    if n < 3 or not np.isfinite(r) or abs(r) == 1.0:
        return np.nan
    return r * np.sqrt((n - 2) / (1 - r * r))


@dataclass
class FactorDiagnostics:
    name: str
    n_obs: int
    horizon: int
    signal_lag: int
    ic_pearson: float
    ic_spearman: float
    ic_tstat: float
    mean_ret: float
    vol_ret: float
    sharpe_ann: float
    hit_rate: float
    turnover: float
    decay_lag1: float
    decay_lag4: float
    decay_lag12: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_factor(
    factor: pd.Series,
    price: pd.Series,
    name: str | None = None,
    horizon: int = 1,
    signal_lag: int = 1,
    annualization: float = 52.0,
) -> FactorDiagnostics:
    """
    Time-series factor validation (for a single asset):
    - Align factor_t (lagged) with forward return_{t,t+horizon}
    - Compute IC (Pearson/Spearman) and correlation t-stat
    - Compute simple z-scored exposure PnL stats, turnover, and decay
    """
    name = name or factor.name or "factor"
    fwd_ret = compute_forward_returns(price, horizon=horizon)

    aligned = pd.DataFrame(
        {
            "factor": factor.sort_index(),
            "fwd_ret": fwd_ret.sort_index(),
        }
    )
    aligned["factor"] = aligned["factor"].shift(signal_lag)
    aligned = aligned.dropna()

    if aligned.empty:
        raise ValueError(f"No overlapping samples for {name} after alignment.")

    n_obs = len(aligned)
    fac = aligned["factor"]
    ret = aligned["fwd_ret"]

    ic_pearson = fac.corr(ret, method="pearson")
    ic_spearman = fac.corr(ret, method="spearman")
    ic_t = _corr_t_stat(ic_pearson, n_obs)

    fac_z = (fac - fac.mean()) / fac.std()
    pnl = fac_z * ret

    mean_ret = float(pnl.mean())
    vol_ret = float(pnl.std())
    sharpe = float(mean_ret / vol_ret * np.sqrt(annualization)) if vol_ret not in (0.0, np.nan) else np.nan
    hit_rate = float((pnl > 0).mean())
    turnover = float(fac_z.diff().abs().mean())

    decay_1 = float(fac_z.autocorr(lag=1))
    decay_4 = float(fac_z.autocorr(lag=4))
    decay_12 = float(fac_z.autocorr(lag=12))

    return FactorDiagnostics(
        name=name,
        n_obs=n_obs,
        horizon=horizon,
        signal_lag=signal_lag,
        ic_pearson=float(ic_pearson),
        ic_spearman=float(ic_spearman),
        ic_tstat=float(ic_t),
        mean_ret=mean_ret,
        vol_ret=vol_ret,
        sharpe_ann=sharpe,
        hit_rate=hit_rate,
        turnover=turnover,
        decay_lag1=decay_1,
        decay_lag4=decay_4,
        decay_lag12=decay_12,
    )


def evaluate_factors(
    factors: pd.DataFrame,
    price: pd.Series,
    horizon: int = 1,
    signal_lag: int = 1,
    annualization: float = 52.0,
) -> pd.DataFrame:
    """
    Batch-evaluate multiple factor columns. Returns a DataFrame of diagnostics.
    """
    results: List[Dict[str, float]] = []
    for col in factors.columns:
        diag = evaluate_factor(
            factors[col],
            price=price,
            name=col,
            horizon=horizon,
            signal_lag=signal_lag,
            annualization=annualization,
        )
        results.append(diag.to_dict())
    return pd.DataFrame(results).set_index("name")
