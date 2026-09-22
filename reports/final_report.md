# QuantLab — Final Report

*A quantitative research project: how markets behave, how to model and price that
behaviour, and whether a simple strategy can beat the market.*

## Summary

Real equity returns are not normally distributed — they have fat tails and
volatility clustering. This project documents those facts empirically, builds and
validates an option-pricing engine, develops a stochastic-volatility model that
reproduces the facts from first principles, calibrates it to real data, and finally
evaluates a trend-following trading strategy with full statistical rigour. Every
result is checked against a known answer or a defensible argument. The strategy
study reaches a deliberately honest conclusion: no statistically significant edge
over buy-and-hold after costs.

## 1. Motivation and approach

The aim was one coherent, defensible project rather than many disconnected
exercises, following the loop a real quant desk uses: understand the data, model
it, price contracts on it, and test strategies. The guiding discipline throughout
is to **validate everything** — against theoretical values, closed-form
benchmarks, or out-of-sample data. The work is in Python (NumPy, pandas, SciPy,
statsmodels), tested with pytest, and version-controlled on GitHub.

## 2. How markets really behave

Using nine years of Apple daily returns, the textbook normal-distribution
assumption fails in two systematic ways.

**Fat tails.** Excess kurtosis is 5.4 (a normal distribution gives 0), and a
fitted Student-t distribution has just ~3.3 degrees of freedom — very heavy tails.
Extreme days (like the −13.8% COVID crash) happen far more often than a bell curve
allows.

![Returns vs normal and Student-t](figures/6_normal_vs_t.png)

**Volatility clustering.** Turbulent periods cluster in time. Raw returns have
almost no autocorrelation (direction is unpredictable), but *squared* returns are
strongly, persistently autocorrelated — volatility is predictable even though
direction is not.

![Autocorrelation of squared returns](figures/5_acf_squared_returns.png)

## 3. Modelling and option pricing

A geometric Brownian motion simulator (integrated with the Euler–Maruyama scheme)
was validated against its exact expected value, then used to price a European call
option by Monte Carlo. The result agreed with the exact Black–Scholes price to
within **0.82%**, and the error was shown to scale as 1/√N, exactly as the Central
Limit Theorem predicts.

![Monte Carlo convergence](figures/mc_convergence.png)

Antithetic variates gave a **2.1× variance reduction**, and a path-dependent Asian
option (which has no closed-form price) was priced by simulation — demonstrating
the method where no formula exists.

## 4. A stochastic-volatility model, and its calibration

The Heston model makes volatility itself random and mean-reverting. It reproduces
all three empirical stylised facts (fat tails, volatility clustering, negative
skew) purely from first principles. Calibrated to Apple returns, it captures the
fat tails well.

![Real vs calibrated Heston](figures/heston_goodness_of_fit.png)

An important honest finding: the calibrated model's negative skew turned out to be
a **discretisation artifact** of the simulation scheme, not a captured leverage
effect — because the leverage parameter cannot be reliably estimated from returns
(it is far better recovered from option prices). Recognising that an output can
look right for the wrong reason is central to the project.

## 5. Does a trading strategy work?

A moving-average trend-following rule was evaluated with the full rigour of the
research checklist: a look-ahead-safe backtester, realistic transaction costs and
turnover, and comparison against a buy-and-hold benchmark.

A parameter sweep revealed the classic overfitting trap — one window (20-day)
appeared to beat the market as an isolated spike, while its neighbours collapsed.

![Trend Sharpe vs window](figures/trend_param_sweep.png)

Under a proper train/test split the 20-day rule did survive out-of-sample, but a
bootstrap confidence interval showed the edge was not statistically significant.
Extending to a 20-stock basket with a conventional window chosen a priori, the edge
shrank to +0.11 in Sharpe with a 95% confidence interval of [−0.31, +0.55] — again
not significant. A subtle but important lesson: the precision of a Sharpe ratio is
governed by the length of the track record in *time*, not the number of assets.

**Conclusion:** no statistically significant edge for trend-following over
buy-and-hold after costs — a rigorously-evaluated null result.

## 6. What this project demonstrates

- **Statistical rigour:** letting the data speak, and quantifying uncertainty
  rather than quoting a single Sharpe ratio.
- **Sound modelling:** models built on theory and validated against known answers.
- **Intellectual honesty:** reporting what failed and why — the discretisation
  artifact, the overfitting spike, the insignificant edge.
- **Clean, reproducible code:** a small shared core, independent modules, tests,
  and a public repository.

## 7. Reproducibility

```
python -m venv .venv && pip install -r requirements.txt
python -m empirical.stylised_facts   # the stylised facts
python price_option.py               # pricing vs Black-Scholes
python calibrate.py                  # model calibration
python portfolio.py                  # the basket strategy study
pytest -q                            # all tests
```

## Detailed stage reports

- Empirical stylised facts: `reports/stylised_facts.md`
- Heston model: `reports/heston.md`
- Calibration: `reports/calibration.md`
- Strategy backtest: `reports/strategy_backtest.md`
