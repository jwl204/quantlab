# QuantLab

[![CI](https://github.com/jwl204/quantlab/actions/workflows/ci.yml/badge.svg)](https://github.com/jwl204/quantlab/actions/workflows/ci.yml)

A quantitative research project exploring how markets move, how to price
derivatives on that randomness, and how to test trading strategies. Built as a
learning project alongside a physics degree, aimed at quantitative finance.

Built with AI assistance as a learning project.

**Full write-up: [`reports/final_report.md`](reports/final_report.md).**

## Status

In progress. Currently implemented:

- `core/` — shared utilities: downloading price data and computing returns
- `empirical/` — a study of the statistical "stylised facts" of real returns
  (fat tails and volatility clustering); see `reports/stylised_facts.md`
- `models/` — geometric Brownian motion and the Heston stochastic-volatility
  model (arithmetic-Euler and log-Euler schemes, full truncation), plus
  historical parameter estimation
- `pricing/` — the Black-Scholes formula, Monte Carlo option pricers (European,
  Asian) with variance reduction, and a semi-analytic Heston pricer (from the
  characteristic function)
- `backtest/` — a holdings-based portfolio backtester (true buy-and-hold,
  rebalanced, and signal portfolios; no look-ahead; explicit costs and turnover)
  with a paired moving-block bootstrap for inference
- `microstructure/` — an event-driven limit order book (price-time priority) and
  an inventory-aware Avellaneda-Stoikov market maker, with latency and informed
  (toxic) order flow and an exact spread-vs-inventory P&L attribution; see
  `reports/microstructure.md`
- `tests/` — unit tests for returns, pricing, models, the semi-analytic Heston
  pricer, portfolio accounting, the order book, and the market maker (52 tests)

An expanding-window walk-forward is implemented; using point-in-time index
constituents (rather than the current fixed universe) is the main remaining
extension.

## Key results so far

- **Empirical:** AAPL returns are strongly non-Gaussian — excess kurtosis 5.4,
  fitted Student-t with ~3.3 degrees of freedom, and persistent volatility
  clustering. See `reports/stylised_facts.md`.
- **Pricing:** the Monte Carlo European-call price agrees with Black-Scholes to
  within 0.8%, with error scaling as 1/sqrt(N) (Central Limit Theorem); antithetic
  variates give a 2.1x variance reduction.
- **Modelling:** the Heston model reproduces the empirical stylised facts
  endogenously. See `reports/heston.md`.
- **Estimation:** GBM and Heston parameters estimated from AAPL returns (a
  physical-measure, method-of-moments estimate, not risk-neutral option calibration).
  Estimated GBM captures the volatility level but no fat tails; estimated Heston
  reproduces the fat tails, while its negative skew is shown to be a discretisation
  artifact rather than a captured leverage effect. See `reports/calibration.md`.
- **Backtesting:** a trend-following study on a 20-stock basket, using holdings-based
  portfolio accounting and a dependence-aware moving-block bootstrap, finds no
  statistically significant Sharpe advantage over a true buy-and-hold after costs
  (difference -0.003; 95% CI covers zero), a result that persists under an
  expanding-window walk-forward with per-fold retuning. An earlier draft's apparent
  edge was traced to a mislabelled rebalanced benchmark and an IID bootstrap, and
  corrected. See `reports/strategy_backtest.md`.
- **Microstructure:** an inventory-aware market maker on a simulated limit order
  book earns almost entirely from spread capture (inventory carry nets to ~0 under
  the Avellaneda-Stoikov skew). The exact P&L attribution separates two frictions:
  quoting **latency** drains *spread capture* (a stale-quote cost, total P&L down
  ~70% by 20 ticks of delay even as fills rise), while **informed (toxic) flow**
  drains *inventory carry* — genuine adverse selection. See
  `reports/microstructure.md`.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -e ".[dev]"
```

## Run

```
python -m empirical.stylised_facts   # the stylised-facts analysis
python price_option.py               # Monte Carlo vs Black-Scholes, and an Asian option
python calibrate.py                  # estimate GBM and Heston from data
python heston_convergence.py         # discretisation-error study vs the semi-analytic price
python run_basket.py                 # holdings-based portfolios: BH, rebalanced, strategy
python walk_forward.py               # expanding-window walk-forward validation
python etf_universe.py               # robustness on a fixed sector-ETF universe
python run_microstructure.py         # market-making P&L attribution: latency and toxic flow
python run_research.py               # regenerate the core headline metrics + provenance manifest
pytest -q                            # run the tests (52)
```
