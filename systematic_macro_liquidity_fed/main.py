from __future__ import annotations

import plotly.graph_objects as go

from config import settings
from data_fetch import DataFetcher
from factors import (
    build_deleveraging,
    build_funding_stress,
    build_market_liquidity,
    build_net_liquidity_flow,
    build_net_liquidity_level,
    build_reserves_rrp_rotation,
    combine_all_signals,
    combine_macro_chain,
    combine_net_liquidity,
)
from factor_tests import evaluate_factors
from strategy import (
    build_position_matrix,
    build_position_signal,
    run_backtest,
    run_cross_section_backtest,
    summarize,
)


def main() -> None:
    print("=== Fed Three-Bucket Liquidity Strategy ===")
    print(f"Date range: {settings.start_date} -> {settings.end_date}")
    fetcher = DataFetcher()

    # 1. Data (buckets + macro-chain inputs from FRED)
    buckets = fetcher.fetch_macro_chain_inputs(
        start=settings.start_date,
        end=settings.end_date,
        weekly_freq=settings.weekly_freq,
        include_sp500=True,
        use_cache=True,
    )
    print(f"Fetched weekly rows: {len(buckets)}")

    if "SP500" not in buckets.columns:
        raise ValueError("SP500 column missing from fetched data.")

    # Cross-sectional asset prices (pandas-datareader: stooq)
    asset_tickers = ["SPY", "GLD", "SLV"]
    asset_prices = fetcher.fetch_assets(
        tickers=asset_tickers,
        start=settings.start_date,
        end=settings.end_date,
        weekly_freq=settings.weekly_freq,
        source="stooq",
        use_cache=True,
    )

    # 2. Fed plumbing (net liquidity) factors
    nl_level = build_net_liquidity_level(buckets, level_window=252)
    nl_flow = build_net_liquidity_flow(buckets, k=21, flow_window=252)
    nl_rot = build_reserves_rrp_rotation(buckets, k=21, flow_window=252)
    net_liq = combine_net_liquidity(nl_level, nl_flow, nl_rot)

    # 3. Funding/Liquidity/Deleveraging macro chain factors
    funding = build_funding_stress(buckets, window=26)
    mkt_liq = build_market_liquidity(buckets, window=26)
    delev = None
    if {"CFTC_ES_LONG", "CFTC_ES_SHORT"}.issubset(buckets.columns):
        delev = build_deleveraging(buckets, window=26)
    macro_chain = combine_macro_chain(funding, mkt_liq, delev)

    # 4. Final combo across all blocks
    all_factors = combine_all_signals(net_liq, macro_chain)
    print("Factor columns:", list(all_factors.columns))

    diag = evaluate_factors(
        all_factors[
            [
                "NL_level_z",
                "NL_flow_z",
                "NL_rot_z",
                "LIQ_signal",
                "FUND_stress",
                "LIQ_stress",
                "MACRO_chain",
                "ALL_signal",
            ]
        ],
        price=buckets["SP500"],
        horizon=1,
        signal_lag=1,
    )
    print("\nFactor diagnostics (h=1, lag=1):")
    print(diag)

    liquidity_score = all_factors["ALL_signal"]

    # 5. Signal + backtest (cross-section: SPY/GLD/SLV equal-weighted when signal > 0)
    weights = build_position_matrix(liquidity_score, assets=asset_tickers, lag=1, long_only=True)
    result = run_cross_section_backtest(asset_prices, weights, trading_cost_bps=10.0)
    stats = summarize(
        result.rename(columns={"port_ret_net": "strategy_ret_net", "cum_strategy": "cum_strategy"})
    )

    print("\nPerformance stats (weekly):")
    print(f"Total return : {stats.total_return:.2%}")
    print(f"Annual return: {stats.annual_return:.2%}")
    print(f"Annual vol   : {stats.annual_vol:.2%}")
    print(f"Sharpe       : {stats.sharpe:.2f}")
    print(f"Max drawdown : {stats.max_drawdown:.2%}")
    print(f"Samples      : {stats.n_periods}")

    # 4. Plot with Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result["cum_strategy"],
            mode="lines",
            name="Strategy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result["cum_buyhold"],
            mode="lines",
            name="Buy & Hold",
        )
    )
    fig.update_layout(
        title="Fed Liquidity Strategy vs Buy & Hold (SP500)",
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    fig.show()


if __name__ == "__main__":
    main()
