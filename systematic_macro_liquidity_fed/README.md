# Fed Three-Bucket Macro Liquidity Framework

This mini-project mirrors the layout of `systematic_macro_liquidity_v5` while focusing
exclusively on the Federal Reserve “three buckets” of liquidity—Treasury General
Account (TGA), Overnight Reverse Repo (RRP), and Reserve Balances. The goal is to
turn those plumbing dynamics into a systematic SP500 trading signal.

## Project layout

```
systematic_macro_liquidity_fed/
├── config.py          # Global settings (date range, FRED key, default freq)
├── data_fetch.py      # FRED downloader + caching for the three buckets and SP500
├── factors.py         # Factor builders (F1 level, F2 flow, F3 rotation) + combiner
├── strategy.py        # Signal preparation + linear backtest + stats helper
├── backtest.py        # (Not needed here; logic lives in strategy.py)
├── main.py            # End-to-end script wiring the pieces together
├── data/              # CSV cache output directory
└── README.md
```

Run `python main.py` after installing the dependencies listed in `requirements.txt`.
Set `FRED_API_KEY` in your environment (or edit `config.py`) before the first run.
