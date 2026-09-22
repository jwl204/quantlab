import numpy as np
import pandas as pd

from backtest.bootstrap import (
    block_bootstrap_sharpe_diff,
    sharpe,
    stationary_bootstrap_sharpe_diff,
)
from backtest.diagnostics import deflated_sharpe_ratio
from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.snapshots import load_snapshot

prices = load_snapshot("basket")
dates = prices.index
n = prices.shape[1]
cost_bps = 5.0
windows = [50, 100, 150, 200, 250]
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values
ew = pd.DataFrame(1.0 / n, index=dates, columns=prices.columns)
bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=cost_bps)["net_return"]


def strat_ret(window):
    raw = pd.DataFrame({t: ma_trend_signal(prices[t], window) for t in prices.columns})
    raw = raw.shift(1).fillna(0.0)
    w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return simulate_portfolio(prices, w, month_ends, cost_bps=cost_bps)["net_return"]


srets = {w: strat_ret(w) for w in windows}
main = srets[200].to_numpy()
bh_np = bh.to_numpy()

d_block = block_bootstrap_sharpe_diff(main, bh_np, block=20, n=3000, seed=0)
d_stat = stationary_bootstrap_sharpe_diff(main, bh_np, expected_block=20, n=3000, seed=0)
print("Primary strategy (200-day) vs buy-and-hold:")
print(f"  observed Sharpe diff: {sharpe(main) - sharpe(bh_np):+.3f}")
lo, hi = np.percentile(d_block, [2.5, 97.5])
print(f"  moving-block  95% CI [{lo:+.3f}, {hi:+.3f}]  P(better) {np.mean(d_block > 0):.2f}")
lo, hi = np.percentile(d_stat, [2.5, 97.5])
print(f"  stationary    95% CI [{lo:+.3f}, {hi:+.3f}]  P(better) {np.mean(d_stat > 0):.2f}")

active = {w: (srets[w] - bh).to_numpy() for w in windows}
active_sr = {w: active[w].mean() / active[w].std() for w in windows}
best_w = max(active_sr, key=active_sr.get)
dsr = deflated_sharpe_ratio(active[best_w], list(active_sr.values()), n_trials=len(windows))
print(f"\nMultiple-testing diagnostic across {len(windows)} windows:")
print(f"  best in-sample window: {best_w}")
print(f"  Deflated Sharpe Ratio of its active return: {dsr:.3f}")
print("  (well below 0.95 -> the best-of-several active Sharpe is not significant)")
