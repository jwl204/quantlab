# The Heston Stochastic-Volatility Model

## Motivation

Geometric Brownian motion (GBM) assumes a single, constant volatility. But real
returns have fat tails and volatility clustering (see `stylised_facts.md`). The
Heston model fixes GBM's central flaw by letting the variance itself follow a
random, mean-reverting process.

## The model

Two coupled stochastic differential equations:

```
dS = mu*S dt + sqrt(v)*S dW1
dv = kappa*(theta - v) dt + xi*sqrt(v) dW2,    corr(dW1, dW2) = rho
```

- **v** — the instantaneous variance (volatility squared), now time-varying.
- **kappa, theta** — the speed and long-run level of the variance's mean reversion.
- **xi** — the volatility of volatility, which controls tail fatness.
- **rho** — the correlation between price and variance shocks; negative rho gives
  the leverage effect (volatility rises when prices fall).

Simulated with Euler-Maruyama using full truncation (the variance is floored at
zero, since variance cannot be negative).

## Stochastic volatility in action

![Heston paths](figures/heston_paths.png)

The volatility (lower panel) is not constant: it wanders and is pulled back toward
its long-run level. Where volatility is high, the price paths (upper panel) are
more violent, and falling prices coincide with rising volatility — the leverage
effect (rho = -0.7).

## Does it reproduce the empirical stylised facts?

From a 10-year simulated daily path (v0 = theta = 0.04, kappa = 3, xi = 0.5,
rho = -0.7):

| Property | Heston result | GBM / normal |
|---|---|---|
| Excess kurtosis (fat tails) | 2.20 | ~0 |
| Skewness (leverage) | -0.11 | ~0 |
| Squared-return autocorrelation | positive, persistent | ~0 |

![Heston squared-return ACF](figures/heston_acf_squared.png)

Remarkably, the fat tails emerge purely from random volatility, even though every
individual time-step is Gaussian. The model therefore regenerates all three
stylised facts originally found in Apple's returns — the empirics and the model
meet in the middle.

## Next

Calibration: fitting the Heston parameters to real market data.

## How to reproduce

```
python heston_paths.py
python heston_stylised_facts.py
```
