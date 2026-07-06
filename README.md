# QuantLab

A quantitative research project exploring how markets move, how to price
derivatives on that randomness, and how to test trading strategies. Built as a
learning project alongside a Year-2 Physics degree, aimed at quantitative finance.

Built with AI assistance as a learning project.

## Status

In progress. Currently implemented:

- `core/` — shared utilities: downloading price data and computing returns
- `tests/` — unit tests for the returns calculations

Planned: empirical study of return distributions, a Monte Carlo option pricer
validated against Black–Scholes, and a strategy backtester.

## What's inside

- `core/data.py` — download adjusted closing prices (via yfinance)
- `core/returns.py` — log and simple returns
- `explore.py` — example script: downloads a stock, computes returns, plots them
- `tests/` — pytest unit tests

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -r requirements.txt
```

## Run

```
python explore.py     # prints return statistics and shows a chart
pytest -q             # runs the tests
```
