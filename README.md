# QuantLab

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
  model (Euler-Maruyama), plus historical parameter estimation
- `pricing/` — the Black-Scholes formula and Monte Carlo option pricers
  (European, Asian), with variance reduction
- `backtest/` — a holdings-based portfolio backtester (true buy-and-hold,
  rebalanced, and signal portfolios; no look-ahead; explicit costs and turnover)
  with a paired moving-block bootstrap for inference
- `tests/` — unit tests for returns, pricing, models, and portfolio accounting

Planned: walk-forward with point-in-time constituents; market microstructure.

## Key results so far

- **Empirical:** AAPL returns are strongly non-Gaussian — excess kurtosis 5.4,
  fitted Student-t with ~3.3 degrees of freedom, and persistent volatility
  clustering. See `reports/stylised_facts.md`.
- **Pricing:** the Monte Carlo European-call price agrees with Black-Scholes to
  within 0.8%, with error scaling as 1/sqrt(N) (Central Limit Theorem); antithetic
  variates give a ~2x variance reduction.
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
  (difference -0.003; 95% CI covers zero). An earlier draft's apparent edge was
  traced to a mislabelled rebalanced benchmark and an IID bootstrap, and corrected.
  See `reports/strategy_backtest.md`.

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
python calibrate.py                  # estimate GBM and Heston from data
python run_basket.py                 # holdings-based portfolios: BH, rebalanced, strategy
python uncertainty_block.py          # paired moving-block bootstrap of the Sharpe difference
pytest -q                            # run the tests
```
