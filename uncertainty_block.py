import numpy as np
import pandas as pd

from backtest.bootstrap import block_bootstrap_sharpe_diff, sharpe

strat = pd.read_csv("strat_ret.csv", index_col=0).iloc[:, 0]
bench = pd.read_csv("bh_ret.csv", index_col=0).iloc[:, 0]
common = strat.index.intersection(bench.index)
strat, bench = strat.loc[common].to_numpy(), bench.loc[common].to_numpy()

print(
    f"Strategy Sharpe {sharpe(strat):.3f} | buy-and-hold Sharpe {sharpe(bench):.3f} | diff {sharpe(strat) - sharpe(bench):+.3f}"
)
print("\nPaired moving-block bootstrap of the Sharpe difference (block-length sensitivity):")
for block in [5, 10, 20, 40]:
    d = block_bootstrap_sharpe_diff(strat, bench, block=block, n=5000, seed=0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(
        f"  block {block:3d}d:  diff {np.median(d):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   P(diff>0) = {(d > 0).mean():.2f}"
    )
