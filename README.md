# QuantLab

A quantitative research project exploring how markets move, how to price
derivatives on that randomness, and how to test trading strategies. Built as a
learning project alongside a physics degree, aimed at quantitative finance.

Built with AI assistance as a learning project.

## Status

In progress. Currently implemented:

- `core/` — shared utilities: downloading price data and computing returns
- `empirical/` — a study of the statistical "stylised facts" of real returns
  (fat tails and volatility clustering); see `reports/stylised_facts.md`
- `tests/` — unit tests for the returns calculations

Planned: stochastic price models (geometric Brownian motion, Heston), a Monte
Carlo option pricer validated against Black–Scholes, and a strategy backtester.

## Key findings so far

An empirical study of AAPL daily returns (2015–2024) shows they are strongly
non-Gaussian: excess kurtosis of 5.4 and a fitted Student-t with ~3.3 degrees of
freedom (fat tails), together with persistent autocorrelation in squared returns
(volatility clustering). Full write-up with figures in
`reports/stylised_facts.md`.

## What's inside

- `core/data.py` — download adjusted closing prices (via yfinance)
- `core/returns.py` — log and simple returns
- `empirical/stylised_facts.py` — distributional and volatility analysis of returns
- `explore.py` — example script: downloads a stock, computes returns, plots them
- `reports/` — written findings with figures
- `tests/` — pytest unit tests

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -r requirements.txt
```

## Run

```
python explore.py                    # return statistics and a chart
python -m empirical.stylised_facts   # the stylised-facts analysis
pytest -q                            # run the tests
```
