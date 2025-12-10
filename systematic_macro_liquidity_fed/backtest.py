"""
Simple re-export layer to mirror the v5 architecture.
`strategy.py` contains the actual implementations; this module exposes
stable names so downstream notebooks can `from backtest import run_backtest`.
"""

from strategy import PerformanceStats, run_backtest, summarize

__all__ = [
    "PerformanceStats",
    "run_backtest",
    "summarize",
]
