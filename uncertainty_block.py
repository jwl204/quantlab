import numpy as np
import pandas as pd

from backtest.bootstrap import block_bootstrap_sharpe_diff, sharpe

strat = pd.read_csv("strat_ret.csv", index_col=0).iloc[:, 0]
bench = pd.read_csv("bh_ret.csv", index_col=0).iloc[:, 0]
common = strat.index.intersection(bench.index)
strat, bench = strat.loc[common].to_numpy(), bench.loc[common].to_numpy()

print("Strategy Sharpe {:.3f} | buy-and-hold Sharpe {:.3f} | diff {:+.3f}".format(
    sharpe(strat), sharpe(bench), sharpe(strat) - sharpe(bench)))
print("\nPaired moving-block bootstrap of the Sharpe difference (block-length sensitivity):")
for block in [5, 10, 20, 40]:
    d = block_bootstrap_sharpe_diff(strat, bench, block=block, n=5000, seed=0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("  block {:3d}d:  diff {:+.3f}   95% CI [{:+.3f}, {:+.3f}]   P(diff>0) = {:.2f}".format(
        block, np.median(d), lo, hi, (d > 0).mean()))