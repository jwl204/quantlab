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
- `models/` — geometric Brownian motion and the Heston stochastic-volatility
  model (Euler-Maruyama), plus calibration to historical data
- `pricing/` — the Black-Scholes formula and Monte Carlo option pricers
  (European, Asian), with variance reduction
- `backtest/` — a vectorised strategy backtester (no look-ahead, transaction
  costs, turnover) with Sharpe / drawdown metrics and bootstrap confidence intervals
- `tests/` — unit tests for returns, pricing, and the models

Planned: market microstructure, and extending the backtest to a basket of assets
with walk-forward validation.

## Key results so far

- **Empirical:** AAPL returns are strongly non-Gaussian — excess kurtosis 5.4,
  fitted Student-t with ~3.3 degrees of freedom, and persistent volatility
  clustering. See `reports/stylised_facts.md`.
- **Pricing:** the Monte Carlo European-call price agrees with Black-Scholes to
  within 0.8%, with error scaling as 1/sqrt(N) (Central Limit Theorem); antithetic
  variates give a ~2x variance reduction.
- **Modelling:** the Heston model reproduces the empirical stylised facts from
  first principles. See `reports/heston.md`.
- **Calibration:** GBM and Heston calibrated to AAPL. Calibrated GBM captures the
  volatility level but no fat tails; calibrated Heston reproduces the fat tails,
  while its negative skew is shown to be a discretisation artifact rather than a
  captured leverage effect (an honest limitation). See `reports/calibration.md`.
- **Backtesting:** a moving-average trend rule on AAPL beat buy-and-hold in- and
  out-of-sample, but the edge is within statistical noise (Sharpe difference 0.48,
  95% CI [-0.31, 1.33]) — a deliberately honest, rigorously-evaluated null result.
  See `reports/strategy_backtest.md`.

## What's inside

- `core/` — data loading and returns
- `empirical/stylised_facts.py` — distributional and volatility analysis
- `models/gbm.py`, `models/heston.py`, `models/calibration.py`
- `pricing/black_scholes.py`, `pricing/monte_carlo.py`
- scripts: `explore.py`, `simulate_paths.py`, `price_option.py`, `convergence.py`,
  `variance_reduction.py`, `heston_paths.py`, `heston_stylised_facts.py`,
  `calibrate.py`, `goodness_of_fit.py`, `validate_heston.py`
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
python -m empirical.stylised_facts   # stylised-facts analysis
python price_option.py               # Monte Carlo vs Black-Scholes, and an Asian option
python heston_paths.py               # stochastic-volatility price paths
python calibrate.py                  # fit GBM and Heston to real data
pytest -q                            # run the tests
```
