import numpy as np
import pandas as pd

from backtest.bootstrap import sharpe
from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.snapshots import load_snapshot

prices = load_snapshot("basket")
dates = prices.index
n = prices.shape[1]
ew = pd.DataFrame(1.0 / n, index=dates, columns=prices.columns)


def reb(freq):
    return pd.Series(index=dates, data=dates).resample(freq).last().dropna().values


def evaluate(cost_bps=5.0, lag=1, freq="ME", window=200):
    raw = pd.DataFrame({t: ma_trend_signal(prices[t], window) for t in prices.columns})
    w = raw.shift(lag).fillna(0.0)
    w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    strat = simulate_portfolio(prices, w, reb(freq), cost_bps=cost_bps)["net_return"]
    bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=cost_bps)["net_return"]
    return sharpe(strat), sharpe(bh)


print("Cost sensitivity (monthly, 1-day lag):")
print(f"  {'cost bps':>8s} {'strat':>7s} {'BH':>7s} {'diff':>7s}")
for c in [0, 5, 10, 20, 50]:
    s, b = evaluate(cost_bps=c)
    print(f"  {c:>8d} {s:7.3f} {b:7.3f} {s - b:+7.3f}")

print("\nExecution-lag sensitivity (5 bps, monthly):")
for lag in [1, 2]:
    s, b = evaluate(lag=lag)
    print(f"  lag {lag}d: strat {s:.3f}  BH {b:.3f}  diff {s - b:+.3f}")

print("\nRebalance sensitivity (5 bps, 1-day lag):")
for freq, name in [("ME", "monthly"), ("QE", "quarterly")]:
    s, b = evaluate(freq=freq)
    print(f"  {name:9s}: strat {s:.3f}  BH {b:.3f}  diff {s - b:+.3f}")
