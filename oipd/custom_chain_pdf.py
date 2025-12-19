"""
Build an implied risk-neutral PDF from your own option-chain data.

The goal is similar to the `oipd` notebook example, but instead of auto-fetching
from Yahoo, you pass in your own quotes (CSV/API/etc.).

Expected input
-------------
- A pandas DataFrame with at least the following columns:
    * strike: option strike price (float)
    * type/right/option_type: call or put flag, anything starting with "C" or "P"
    * bid/ask or mid or last/close: prices in option premium terms
- If only calls are provided, that is sufficient. If you only have puts, you
  must also supply the spot price so calls can be synthesized via put-call parity.

What the code does
------------------
1) Clean and merge the raw chain into a call-price curve.
2) Fit a smoothing spline C(K) over strikes.
3) Apply Breeden-Litzenberger:  pdf(K) = exp(rT) * d^2 C / dK^2.
4) Plot both the call curve (raw vs smoothed) and the implied PDF.

Dependencies: numpy, pandas, scipy, plotly (already present in this repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline


@dataclass
class MarketInputs:
    valuation_date: date
    expiry_date: date
    risk_free_rate: float
    spot: Optional[float] = None  # optional, used to back out calls from puts

    @property
    def tau(self) -> float:
        """Year fraction to expiry."""
        days = (self.expiry_date - self.valuation_date).days
        if days <= 0:
            raise ValueError("expiry_date must be after valuation_date")
        return days / 365.25

    @property
    def discount_factor(self) -> float:
        return float(np.exp(-self.risk_free_rate * self.tau))

    @property
    def forward(self) -> Optional[float]:
        return (
            float(self.spot * np.exp(self.risk_free_rate * self.tau))
            if self.spot is not None
            else None
        )


def _standardize_columns(chain: pd.DataFrame) -> pd.DataFrame:
    """Normalize common option-chain column names into strike/type/mid."""
    df = chain.copy()
    rename_map = {
        "right": "type",
        "option_type": "type",
        "optionType": "type",
        "strike_price": "strike",
        "Strike": "strike",
        "lastPrice": "last",
        "close": "last",
        "mark": "mid",
        "Mark": "mid",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str[0]
    else:
        df["type"] = "C"

    if "mid" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
        elif "last" in df.columns:
            df["mid"] = df["last"]

    cols_to_keep = [c for c in ["strike", "type", "mid", "bid", "ask", "last"] if c in df.columns]
    return df[cols_to_keep].dropna(subset=["strike", "type", "mid"])


def _merge_calls_and_puts(
    df: pd.DataFrame, market: MarketInputs
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Return a DataFrame with strikes and call mid prices.
    If only puts exist, convert them to calls via put-call parity using the spot.
    Also try to infer a forward price if both calls and puts are present.
    """
    calls = df[df["type"] == "C"].copy()
    puts = df[df["type"] == "P"].copy()

    inferred_forward = None
    if not calls.empty and not puts.empty:
        merged = pd.merge(
            calls[["strike", "mid"]],
            puts[["strike", "mid"]],
            on="strike",
            how="inner",
            suffixes=("_call", "_put"),
        )
        if not merged.empty:
            inferred_forward = float(
                np.median(
                    merged["strike"]
                    + np.exp(market.risk_free_rate * market.tau)
                    * (merged["mid_call"] - merged["mid_put"])
                )
            )

    if calls.empty and not puts.empty:
        if market.spot is None:
            raise ValueError("Only puts provided; supply spot in MarketInputs to use parity.")
        calls = puts.copy()
        calls["mid"] = calls["mid"] + market.spot - calls["strike"] * market.discount_factor
        calls["type"] = "C"

    if calls.empty:
        raise ValueError("No call prices available after cleaning.")

    calls = calls.dropna(subset=["mid"]).copy()
    calls["strike"] = calls["strike"].astype(float)
    calls["mid"] = calls["mid"].astype(float)
    calls = calls.sort_values("strike")
    calls = calls.loc[calls["mid"] > 0]
    if calls.empty:
        raise ValueError("No positive call prices left after filtering.")

    return calls[["strike", "mid"]], inferred_forward


def estimate_pdf_from_chain(
    chain: pd.DataFrame,
    market: MarketInputs,
    smoothing: float = 1e-3,
    grid_points: int = 400,
) -> Dict[str, object]:
    """
    Compute the implied PDF and return data + Plotly figures.

    Returns a dict with:
        - strikes: np.ndarray of the evaluation grid
        - pdf: np.ndarray normalized to integrate to 1
        - call_curve: original call strikes/mids (pd.DataFrame)
        - call_spline: fitted spline object
        - forward: inferred forward (if any)
        - fig_call: Plotly Figure for call curve fit
        - fig_pdf: Plotly Figure for implied PDF
    """
    df = _standardize_columns(chain)
    calls, inferred_forward = _merge_calls_and_puts(df, market)

    x = calls["strike"].to_numpy()
    y = calls["mid"].to_numpy()
    if len(np.unique(x)) < 5:
        raise ValueError("Need at least ~5 distinct strikes to build a stable spline.")

    spline = UnivariateSpline(x, y, s=smoothing)
    grid = np.linspace(x.min(), x.max(), grid_points)
    c2 = spline.derivative(n=2)(grid)

    pdf = np.exp(market.risk_free_rate * market.tau) * c2
    pdf = np.maximum(pdf, 0.0)
    mass = np.trapz(pdf, grid)
    if mass > 0:
        pdf = pdf / mass

    fig_call = go.Figure()
    fig_call.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Call mid (input)"))
    fig_call.add_trace(go.Scatter(x=grid, y=spline(grid), mode="lines", name="Spline fit"))
    fig_call.update_layout(
        title="Call price curve (cleaned) vs strike",
        xaxis_title="Strike",
        yaxis_title="Call price",
        template="plotly_white",
    )

    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Scatter(x=grid, y=pdf, mode="lines", name="Implied PDF"))
    fig_pdf.update_layout(
        title="Implied risk-neutral PDF via Breeden-Litzenberger",
        xaxis_title="Price at expiry (strike space)",
        yaxis_title="Probability density",
        template="plotly_white",
    )

    return {
        "strikes": grid,
        "pdf": pdf,
        "call_curve": calls,
        "call_spline": spline,
        "forward": inferred_forward or market.forward,
        "fig_call": fig_call,
        "fig_pdf": fig_pdf,
    }


def example_usage() -> None:
    """Small self-contained demo using synthetic data."""
    np.random.seed(0)
    strikes = np.arange(60, 141, 5)
    true_forward = 100.0
    r = 0.02
    T = 30 / 365.25
    sigma = 0.35

    def bs_call_price(fwd, k, r_, t, vol):
        # Using forward form to keep it simple (undiscounted)
        from scipy.stats import norm

        if vol <= 0 or t <= 0:
            return max(fwd - k, 0.0) * np.exp(-r_ * t)
        d1 = (np.log(fwd / k) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
        d2 = d1 - vol * np.sqrt(t)
        undiscounted = fwd * norm.cdf(d1) - k * norm.cdf(d2)
        return np.exp(-r_ * t) * undiscounted

    call_mid = np.array(
        [bs_call_price(true_forward, k, r, T, sigma) for k in strikes]
    )
    noise = 0.02 * call_mid * np.random.randn(len(call_mid))
    call_mid = np.maximum(call_mid + noise, 0.01)

    chain = pd.DataFrame(
        {
            "strike": strikes,
            "type": ["C"] * len(strikes),
            "bid": call_mid * 0.99,
            "ask": call_mid * 1.01,
        }
    )

    market = MarketInputs(
        valuation_date=date.today(),
        expiry_date=date(2025, 12, 19),
        risk_free_rate=r,
        spot=np.exp(-r * T) * true_forward,
    )

    results = estimate_pdf_from_chain(chain, market, smoothing=1e-4)
    print("Forward used (spot * exp(rT) or inferred):", results["forward"])
    print("PDF integrates to:", np.trapz(results["pdf"], results["strikes"]))
    # To view the plots interactively:
    # results["fig_call"].show()
    # results["fig_pdf"].show()


if __name__ == "__main__":
    example_usage()
