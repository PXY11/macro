from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_position_signal(
    liquidity_signal: pd.Series,
    lag: int = 1,
    clip: float | None = 1.0,
    long_only: bool = True,
) -> pd.Series:
    """
    Turn the liquidity score into a tradable position.
    If long_only is True, negative signals are set to 0 so "bad liquidity" just flattens.
    """
    sig = liquidity_signal.sort_index().copy()

    if clip is not None:
        sig = sig.clip(-clip, clip)

    if long_only:
        sig = sig.clip(lower=0.0)

    if lag > 0:
        sig = sig.shift(lag)

    sig = sig.dropna()
    sig.name = "position"
    return sig


def build_position_matrix(
    signal: pd.Series,
    assets: list[str],
    lag: int = 1,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Turn a scalar liquidity signal into cross-sectional weights.
    Here: equal weight across assets when signal > 0; otherwise flat.
    """
    sig = signal.sort_index().copy()
    if long_only:
        sig = sig.clip(lower=0.0)
    if lag > 0:
        sig = sig.shift(lag)

    weights = []
    for _ in assets:
        weights.append(sig.copy())
    mat = pd.concat(weights, axis=1)
    mat.columns = assets

    # Normalize row-wise to sum to 1 when positive, else 0.
    row_sum = mat.sum(axis=1)
    mat = mat.div(row_sum.where(row_sum != 0.0, 1.0), axis=0)
    mat = mat.fillna(0.0)
    return mat


def run_backtest(
    price: pd.Series,
    signal: pd.Series,
    trading_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Simplified linear backtest:
    - position_t uses already-lagged signal
    - strategy return = position * asset return
    - turnover cost proportional to |Δposition|
    """
    price = price.sort_index().copy()
    ret = price.pct_change().fillna(0.0)

    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ret"] = ret

    df = df.join(signal, how="left")
    df["position"] = df["position"].ffill().fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(0.0)

    cost = trading_cost_bps / 10000.0
    df["strategy_ret_gross"] = df["position"] * df["ret"]
    df["strategy_ret_net"] = df["strategy_ret_gross"] - df["turnover"] * cost

    df["cum_strategy"] = (1.0 + df["strategy_ret_net"]).cumprod()
    df["cum_buyhold"] = (1.0 + df["ret"]).cumprod()

    return df


def run_cross_section_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    trading_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Cross-sectional backtest:
    - prices: wide DataFrame of asset prices
    - weights: wide DataFrame of target weights (already lagged)
    """
    prices = prices.sort_index().copy()
    rets = prices.pct_change().fillna(0.0)

    w = weights.reindex(rets.index).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)

    port_ret_gross = (w * rets).sum(axis=1)
    cost = trading_cost_bps / 10000.0
    port_ret_net = port_ret_gross - turnover * cost

    df = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "turnover": turnover,
        },
        index=rets.index,
    )

    df["cum_strategy"] = (1.0 + df["port_ret_net"]).cumprod()

    # Equal-weight buy & hold for benchmark
    n = prices.shape[1]
    eq_w = 1.0 / n if n else 0.0
    bh_ret = (rets * eq_w).sum(axis=1)
    df["cum_buyhold"] = (1.0 + bh_ret).cumprod()
    return df


@dataclass
class PerformanceStats:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    n_periods: int


def summarize(result: pd.DataFrame) -> PerformanceStats:
    if result.empty:
        raise ValueError("Backtest result is empty.")

    total_return = float(result["cum_strategy"].iloc[-1] - 1.0)
    n_periods = len(result)

    ann_return = (1.0 + total_return) ** (52.0 / n_periods) - 1.0 if n_periods > 1 else np.nan
    vol = float(result["strategy_ret_net"].std() * np.sqrt(52.0))
    sharpe = float(ann_return / vol) if vol not in (0.0, np.nan) else np.nan

    running_max = result["cum_strategy"].cummax()
    drawdown = (result["cum_strategy"] / running_max) - 1.0
    max_dd = float(drawdown.min())

    return PerformanceStats(
        total_return=total_return,
        annual_return=ann_return,
        annual_vol=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_periods=n_periods,
    )
