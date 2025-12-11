from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Simple rolling z-score helper."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def build_net_liquidity_level(df: pd.DataFrame, level_window: int = 252) -> pd.DataFrame:
    """
    Net Liquidity level:
    NL_raw = RESERVE_BALANCE - TGA - RRP, then rolling z-score.
    """
    out = pd.DataFrame(index=df.index.copy())
    reserves = df["RESERVE_BALANCE"]
    tga = df["TGA"]
    rrp = df["RRP"]

    out["NL_level_raw"] = reserves - tga - rrp
    out["NL_level_z"] = rolling_zscore(out["NL_level_raw"], level_window)
    return out


def build_net_liquidity_flow(df: pd.DataFrame, k: int = 21, flow_window: int = 252) -> pd.DataFrame:
    """
    Net Liquidity flow:
    Flow_k = NL_raw - NL_raw.shift(k), then rolling z-score.
    """
    out = pd.DataFrame(index=df.index.copy())
    reserves = df["RESERVE_BALANCE"]
    tga = df["TGA"]
    rrp = df["RRP"]

    nl_raw = reserves - tga - rrp
    col_flow = f"NL_flow_{k}"
    out[col_flow] = nl_raw - nl_raw.shift(k)
    out["NL_flow_z"] = rolling_zscore(out[col_flow], flow_window)
    return out


def build_reserves_rrp_rotation(df: pd.DataFrame, k: int = 21, flow_window: int = 252) -> pd.DataFrame:
    """
    RRP -> Reserves rotation factor:
    Rot_k = ΔReserves_k - ΔRRP_k
    """
    out = pd.DataFrame(index=df.index.copy())
    reserves = df["RESERVE_BALANCE"]
    rrp = df["RRP"]

    d_res = reserves - reserves.shift(k)
    d_rrp = rrp - rrp.shift(k)

    col_rot = f"NL_rot_{k}"
    out[col_rot] = d_res - d_rrp
    out["NL_rot_z"] = rolling_zscore(out[col_rot], flow_window)
    return out


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_funding_stress(df: pd.DataFrame, window: int = 26) -> pd.DataFrame:
    """
    Funding stress block using FRED-available series:
    - TED spread (TEDRATE)
    - AA 30d CP minus Fed Funds (DCPN30 - DFF)
    - AA 30d CP minus SOFR (DCPN30 - SOFR)
    Returns z-scored components and a combined tanh score.
    """
    required = ["CP_30D_AA", "FED_FUNDS", "SOFR"]
    _ensure_columns(df, required)

    out = pd.DataFrame(index=df.index.copy())
    components = []
    weights = []

    if "TEDRATE" in df.columns and df["TEDRATE"].notna().sum() > window:
        out["FUND_ted_z"] = rolling_zscore(df["TEDRATE"], window).ffill()
        components.append(out["FUND_ted_z"])
        weights.append(0.5)

    cp_ff = df["CP_30D_AA"] - df["FED_FUNDS"]
    cp_sofr = df["CP_30D_AA"] - df["SOFR"]

    out["FUND_cp_ff_z"] = rolling_zscore(cp_ff, window).ffill()
    out["FUND_cp_sofr_z"] = rolling_zscore(cp_sofr, window).ffill()
    components.extend([out["FUND_cp_ff_z"], out["FUND_cp_sofr_z"]])
    weights.extend([0.25, 0.25])

    if not components:
        raise ValueError("No funding components available.")

    wsum = sum(weights)
    combo = sum(w * c for w, c in zip(weights, components)) / wsum
    out["FUND_stress"] = np.tanh(combo)
    return out


def build_market_liquidity(df: pd.DataFrame, window: int = 26) -> pd.DataFrame:
    """
    Market liquidity / credit risk block using FRED series:
    - VIXCLS
    - HY OAS (BAMLH0A0HYM2)
    - IG OAS (BAMLC0A0CM)
    Higher z-score => tighter liquidity / higher stress. Returns component z-scores and tanh score.
    """
    required = ["VIX", "HY_OAS", "IG_OAS"]
    _ensure_columns(df, required)

    out = pd.DataFrame(index=df.index.copy())
    out["LIQ_vix_z"] = rolling_zscore(df["VIX"], window).ffill()
    out["LIQ_hy_oas_z"] = rolling_zscore(df["HY_OAS"], window).ffill()
    out["LIQ_ig_oas_z"] = rolling_zscore(df["IG_OAS"], window).ffill()

    combo = 0.5 * out["LIQ_vix_z"] + 0.35 * out["LIQ_hy_oas_z"] + 0.15 * out["LIQ_ig_oas_z"]
    out["LIQ_stress"] = np.tanh(combo)
    return out


def build_deleveraging(df: pd.DataFrame, window: int = 26) -> pd.DataFrame:
    """
    Deleveraging proxy using CFTC S&P 500 futures positioning (FRED CFTC_ES_*):
    - Net = long - short
    - Stress grows when net positioning falls (diff < 0)
    Produces level/delta z-scores and a combined tanh score (higher => more deleveraging pressure).
    """
    required = ["CFTC_ES_LONG", "CFTC_ES_SHORT"]
    _ensure_columns(df, required)

    out = pd.DataFrame(index=df.index.copy())
    net = df["CFTC_ES_LONG"] - df["CFTC_ES_SHORT"]
    d_net = net.diff()

    out["DELEV_net_level_z"] = rolling_zscore(-net, window).ffill()  # negative net => deleveraging stress up
    out["DELEV_net_chg_z"] = rolling_zscore(-d_net, window).ffill()  # falling net => stress up

    combo = 0.5 * out["DELEV_net_level_z"] + 0.5 * out["DELEV_net_chg_z"]
    out["DELEV_stress"] = np.tanh(combo)
    return out



def combine_net_liquidity(
    level: pd.DataFrame,
    flow: pd.DataFrame | None = None,
    rotation: pd.DataFrame | None = None,
    metals: pd.DataFrame | None = None,
    w_level: float = 0.2,
    w_flow: float = 0.6,
    w_rotation: float = 0.2,
) -> pd.DataFrame:
    """
    Combine Net Liquidity level/flow/rotation into a liquidity signal.
    Returns original factor columns plus LIQ_signal_raw and LIQ_signal (tanh).
    """
    dfs = []
    if level is not None:
        dfs.append(level)
    if flow is not None:
        dfs.append(flow)
    if rotation is not None:
        dfs.append(rotation)
    if metals is not None:
        dfs.append(metals)

    if not dfs:
        raise ValueError("combine_net_liquidity() requires at least one non-empty factor DataFrame")

    out = pd.concat(dfs, axis=1).sort_index()

    level_z = out["NL_level_z"] if "NL_level_z" in out.columns else 0.0
    flow_z = out["NL_flow_z"] if "NL_flow_z" in out.columns else 0.0
    rot_z = out["NL_rot_z"] if "NL_rot_z" in out.columns else 0.0

    combo_raw = w_level * level_z + w_flow * flow_z + w_rotation * rot_z
    out["LIQ_signal_raw"] = combo_raw
    out["LIQ_signal"] = np.tanh(combo_raw)

    return out


def combine_macro_chain(
    funding: pd.DataFrame | None = None,
    liquidity: pd.DataFrame | None = None,
    deleveraging: pd.DataFrame | None = None,
    w_funding: float = 0.4,
    w_liquidity: float = 0.35,
    w_deleveraging: float = 0.25,
) -> pd.DataFrame:
    """
    Combine the three-block chain (funding -> market liquidity -> deleveraging)
    into a single stress score. Each block should include its *_stress column.
    """
    dfs = [d for d in (funding, liquidity, deleveraging) if d is not None]
    if not dfs:
        raise ValueError("combine_macro_chain() requires at least one non-empty block.")

    out = pd.concat(dfs, axis=1).sort_index()
    fund = out["FUND_stress"] if "FUND_stress" in out.columns else 0.0
    liq = out["LIQ_stress"] if "LIQ_stress" in out.columns else 0.0
    delev = out["DELEV_stress"] if "DELEV_stress" in out.columns else 0.0

    raw = w_funding * fund + w_liquidity * liq + w_deleveraging * delev
    out["MACRO_chain_raw"] = raw
    out["MACRO_chain"] = np.tanh(raw)
    return out


def combine_all_signals(
    net_liquidity: pd.DataFrame | None = None,
    macro_chain: pd.DataFrame | None = None,
    w_net_liquidity: float = 0.6,
    w_macro_chain: float = 0.4,
) -> pd.DataFrame:
    """
    Final combo: blend Fed plumbing (net liquidity) with the macro-chain stress block.
    Assumes net_liquidity has LIQ_signal, macro_chain has MACRO_chain.
    """
    dfs = [d for d in (net_liquidity, macro_chain) if d is not None]
    if not dfs:
        raise ValueError("combine_all_signals() requires at least one input DataFrame.")

    out = pd.concat(dfs, axis=1).sort_index()

    liq = out["LIQ_signal"] if "LIQ_signal" in out.columns else 0.0
    macro = out["MACRO_chain"] if "MACRO_chain" in out.columns else 0.0

    raw = w_net_liquidity * liq + w_macro_chain * macro
    out["ALL_signal_raw"] = raw
    out["ALL_signal"] = np.tanh(raw)
    return out
