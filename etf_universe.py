import numpy as np
import pandas as pd

from backtest.bootstrap import block_bootstrap_sharpe_diff, sharpe
from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.snapshots import load_snapshot

prices = load_snapshot("etf")
dates = prices.index
n = prices.shape[1]
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values

ew = pd.DataFrame(1.0 / n, index=dates, columns=prices.columns)
bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=5.0)["net_return"]
raw = (
    pd.DataFrame({t: ma_trend_signal(prices[t], 200) for t in prices.columns}).shift(1).fillna(0.0)
)
w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
strat = simulate_portfolio(prices, w, month_ends, cost_bps=5.0)["net_return"]

s, b = sharpe(strat.to_numpy()), sharpe(bh.to_numpy())
d = block_bootstrap_sharpe_diff(strat.to_numpy(), bh.to_numpy(), block=20, n=3000, seed=0)
lo, hi = np.percentile(d, [2.5, 97.5])
print("Sector-ETF universe (9 SPDR sector ETFs, 200-day trend, 5 bps, monthly):")
print(f"  strategy Sharpe {s:.3f}, buy-and-hold {b:.3f}, diff {s - b:+.3f}")
print(f"  block-bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  P(strategy better) {np.mean(d > 0):.2f}")
