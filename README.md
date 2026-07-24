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
- `models/` — price models: geometric Brownian motion and the Heston
  stochastic-volatility model, both simulated via Euler-Maruyama
- `pricing/` — the Black-Scholes formula and Monte Carlo option pricers
  (European, Asian), with variance reduction
- `tests/` — unit tests for returns, pricing, and the models

Planned: calibrating the Heston model to market data, and a strategy backtester.

## Key results so far

- **Empirical:** AAPL returns are strongly non-Gaussian — excess kurtosis 5.4,
  fitted Student-t with ~3.3 degrees of freedom, and persistent volatility
  clustering. See `reports/stylised_facts.md`.
- **Pricing:** the Monte Carlo European-call price agrees with the exact
  Black-Scholes value to within 0.8%, and the error scales as 1/sqrt(N) as
  predicted by the Central Limit Theorem. Antithetic variates give a ~2x
  variance reduction.
- **Modelling:** the Heston model reproduces all three empirical stylised facts
  (fat tails, volatility clustering, negative skew) from first principles. See
  `reports/heston.md`.

## What's inside

- `core/` — data loading and returns
- `empirical/stylised_facts.py` — distributional and volatility analysis
- `models/gbm.py`, `models/heston.py` — the price simulators
- `pricing/black_scholes.py`, `pricing/monte_carlo.py` — option pricers
- `explore.py`, `simulate_paths.py`, `price_option.py`, `convergence.py`,
  `variance_reduction.py`, `heston_paths.py`, `heston_stylised_facts.py` — scripts
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
python -m empirical.stylised_facts   # the stylised-facts analysis
python price_option.py               # Monte Carlo vs Black-Scholes, and an Asian option
python heston_paths.py               # stochastic-volatility price paths
pytest -q                            # run the tests
```
