import numpy as np
import pandas as pd

from backtest.bootstrap import block_bootstrap_sharpe_diff, sharpe

strat = pd.read_csv("oos_ret.csv", index_col=0).iloc[:, 0]
bench = pd.read_csv("bh_oos_ret.csv", index_col=0).iloc[:, 0]
common = strat.index.intersection(bench.index)
strat, bench = strat.loc[common].to_numpy(), bench.loc[common].to_numpy()

d = block_bootstrap_sharpe_diff(strat, bench, block=20, n=5000, seed=0)
lo, hi = np.percentile(d, [2.5, 97.5])
print(
    f"Walk-forward OOS Sharpe diff {sharpe(strat) - sharpe(bench):+.3f}  "
    f"95% CI [{lo:+.3f}, {hi:+.3f}]  P(strategy better) = {(d > 0).mean():.2f}"
)
