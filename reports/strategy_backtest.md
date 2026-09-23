# Trend-Following on a US Equity Basket: An Honest Backtest

**Revision note (September 2026).** An earlier version of this study used the daily
cross-sectional mean of returns as the benchmark — which is a *daily-rebalanced*
equal-weight portfolio, not buy-and-hold — and estimated confidence intervals with
an IID bootstrap, which understates the serial dependence in returns. Following a
methodological review, the benchmark was reimplemented with explicit holdings-based
portfolio accounting (true buy-and-hold, and a separately-labelled rebalanced
benchmark), and the inference was replaced with a paired moving-block bootstrap.
The results below use the corrected methods. The correction *strengthened* the
conclusion. The earlier code remains in the Git history.

## Research question

Does a simple moving-average trend rule on a basket of US equities produce
risk-adjusted returns that beat buy-and-hold, out-of-sample and after realistic
transaction costs?

## Method

- **Universe:** 20 large-cap US stocks; **200-day** trend window fixed *a priori*
  (not tuned), applied identically to every stock; **5 bps** cost per unit turnover.
- **Portfolio accounting:** an explicit holdings/cash engine (`backtest/portfolio.py`)
  tracks shares, cash, NAV, turnover and cost. Three portfolios are built from the
  same engine:
  - *True buy-and-hold* — buy equal weight once; weights then drift with prices.
  - *Monthly-rebalanced equal weight* — reset to equal weight each month (this is
    what the earlier mislabelled benchmark approximated).
  - *Trend strategy* — hold equal weight among stocks currently in an uptrend,
    signals lagged one day (no look-ahead), rebalanced monthly.
- **Inference:** paired moving-block bootstrap of the Sharpe difference, with
  block-length sensitivity.

## Corrected results (2015-2024, 5 bps costs)

| Portfolio | Sharpe | Turnover / yr | Growth of $1 |
|---|---|---|---|
| True buy-and-hold | 1.062 | 0.11 | 8.68x |
| Monthly-rebalanced equal weight | 1.000 | 0.64 | 4.54x |
| Trend strategy | 1.059 | 4.26 | 4.70x |

Two important observations. First, the earlier "buy-and-hold" number (Sharpe ~0.99)
matches the *monthly-rebalanced* portfolio here (1.000), confirming the original
mislabel. Second, *true* buy-and-hold is a much tougher benchmark: buying once lets
winners compound, nearly doubling the final wealth versus the rebalanced version.

Against the correct benchmark, the trend strategy shows **no Sharpe advantage**
(1.059 vs 1.062), delivers barely half the wealth (4.70x vs 8.68x), and trades about
**40x more** (turnover 4.26 vs 0.11 per year).

## Statistical inference (paired moving-block bootstrap)

Strategy Sharpe 1.059, true buy-and-hold 1.062, difference -0.003.

| Block length | Median diff | 95% CI | P(strategy better) |
|---|---|---|---|
| 5 days | +0.001 | [-0.378, +0.393] | 0.50 |
| 10 days | +0.000 | [-0.388, +0.417] | 0.50 |
| 20 days | +0.009 | [-0.383, +0.433] | 0.52 |
| 40 days | +0.010 | [-0.379, +0.461] | 0.52 |

The Sharpe difference is indistinguishable from zero at every block length, with the
probability of the strategy being better close to a coin flip. The intervals widen
with block length, which is exactly why the earlier IID bootstrap (effectively a
one-day block) understated the uncertainty.

## Walk-forward validation (stateful)

A single train/test split is fragile, so the strategy was also evaluated with a
stateful expanding-window walk-forward: one continuous portfolio is carried across
four sequential out-of-sample folds, and in each fold the moving-average window is
re-selected (from 50-250 days) using only prior data. A single live portfolio
persists across folds; when a fold selects a new window, that model becomes active
at the **next scheduled monthly rebalance**, where the resulting transition trade
and its cost are charged. The portfolio trades only on month-ends, not on the fold
boundary itself.

| Test period | Selected window | Strategy Sharpe | Buy-and-hold Sharpe | Difference |
|---|---|---|---|---|
| 2016-10 to 2018-08 | 150 | 2.00 | 2.33 | -0.33 |
| 2018-08 to 2020-05 | 150 | 0.64 | 0.48 | +0.16 |
| 2020-05 to 2022-03 | 100 | 1.32 | 1.20 | +0.13 |
| 2022-03 to 2023-12 | 100 | 0.60 | 0.89 | -0.30 |

The selected window adapts and the folds are mixed. Stitched across all folds, the
continuous strategy scores Sharpe 0.987 versus buy-and-hold's 1.004 (difference
-0.017; paired block-bootstrap 95% CI [-0.278, +0.258], probability the strategy is
better 0.48). Even with per-fold retuning and transition costs, there is no
out-of-sample edge.

## Robustness and sensitivity

The primary result is robust to the main modelling choices, and the small edge does
not survive realistic assumptions.

Transaction costs -- the strategy only matches buy-and-hold at zero cost:

| Cost (bps) | Strategy Sharpe | Buy-and-hold Sharpe | Difference |
|---|---|---|---|
| 0 | 1.071 | 1.062 | +0.009 |
| 5 | 1.059 | 1.062 | -0.003 |
| 10 | 1.047 | 1.062 | -0.015 |
| 20 | 1.022 | 1.061 | -0.039 |
| 50 | 0.949 | 1.060 | -0.110 |

Execution lag and rebalancing -- a two-day rather than one-day execution lag worsens
the difference (-0.003 to -0.048), as does quarterly rather than monthly rebalancing
(-0.045). The result does not survive more conservative execution assumptions.

Inference method -- the paired moving-block and stationary bootstraps agree closely
(95% CI [-0.383, +0.436] vs [-0.382, +0.450]), so the interval is not an artifact of
the fixed block length.

Multiple testing -- across the five candidate windows, the Deflated Sharpe Ratio of
the best in-sample active return is 0.065 (far below 0.95): once the number of trials
is accounted for, the best-of-several result is not statistically significant.

## Universe and survivorship bias (limitation)

The 20 tickers are large, liquid companies selected by hand at the present day. This
introduces survivorship and selection bias: firms that failed or were delisted over
2015-2024 are excluded, which flatters any long-biased result. A stronger design
would use point-in-time index constituents or a rules-based universe declared at the
sample start. This study should therefore be read as exploratory evidence on a
surviving large-cap universe, not a general claim about US equities.

**Robustness universe.** As a cross-check on a universe that substantially reduces
single-stock survivorship and selection bias, the same strategy was run on a
rules-based universe of 9 SPDR sector ETFs (sectors are not subject to single-name
delisting). This is not a complete point-in-time investable universe -- it is still a
retrospectively chosen set of funds that existed over the sample -- but it removes the
single-name survivorship of the hand-picked basket. There the strategy Sharpe was
0.702 versus buy-and-hold 0.661 (difference +0.041; 95% CI [-0.236, +0.317],
P(strategy better) 0.61). Notably the sign of the (still insignificant) edge flips: on
the hand-picked survivors buy-and-hold benefits from the exceptional compounding of the
selected winners, so trend-following trails; on the sector universe trend-following
marginally leads. The conclusion of no statistically significant edge holds on both
universes, but the point estimate's sign depends on the universe -- which is exactly
why the selection matters and is disclosed.

## Conclusion

On a hand-selected 20-stock basket, over 2015-2024, after 5 bps costs, a 200-day
trend rule shows **no statistically significant Sharpe advantage over a true
buy-and-hold** (difference -0.003; 95% CI covers zero for all block lengths). It also
delivers materially less total wealth and far higher turnover. The apparent
single-stock "win" in the earlier draft was overfitting; the earlier basket "win" was
an artifact of a mislabelled rebalanced benchmark and an over-confident IID bootstrap.
Corrected, the result is a clean, well-supported null. The null also holds under a walk-forward evaluation that re-tunes the window each fold (a stateful walk-forward out-of-sample Sharpe difference of -0.017, 95% CI [-0.278, +0.258]). It is robust to cost, execution-lag, rebalance and bootstrap-method choices, and the best-of-five-windows result is not significant after a deflated-Sharpe multiple-testing correction (0.065).

## Limitations and next steps

- Survivorship/selection bias in the hand-picked universe (see above).
- Sensitive to the cost assumption; the strategy has high turnover.
- Walk-forward with per-fold parameter refitting and a multiple-testing correction
  (deflated Sharpe) are both implemented (see above). The main remaining extension
  is **point-in-time index constituents** — a survivorship-free, as-of-date
  membership list — rather than the current fixed universe.

## How to reproduce

```
python build_snapshot.py     # build the price snapshots (basket + sector ETFs)
pytest -q                    # 54 tests incl. leakage, accounting and walk-forward invariants
python run_basket.py         # three portfolios: true BH, rebalanced EW, strategy
python uncertainty_block.py  # paired moving-block bootstrap (single split)
python walk_forward.py       # stateful walk-forward with per-fold retuning
python oos_uncertainty.py    # bootstrap of the stitched out-of-sample difference
python sensitivity.py        # cost / execution-lag / rebalance sensitivity grids
python robustness.py         # stationary bootstrap + deflated Sharpe (multiple testing)
python etf_universe.py       # robustness on a fixed sector-ETF universe
python run_research.py       # regenerate the core headline metrics + a provenance manifest
```
