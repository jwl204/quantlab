import numpy as np
import pandas as pd

strat = pd.read_csv("strat_port.csv", index_col=0).iloc[:, 0]
bench = pd.read_csv("bench_port.csv", index_col=0).iloc[:, 0]
common = strat.index.intersection(bench.index)
strat = strat.loc[common].to_numpy()
bench = bench.loc[common].to_numpy()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def boot_ci(r, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [sharpe(rng.choice(r, len(r), replace=True)) for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


print("Trend      Sharpe {:.2f}  95% CI [{:.2f}, {:.2f}]".format(sharpe(strat), *boot_ci(strat)))
print("Buy-hold   Sharpe {:.2f}  95% CI [{:.2f}, {:.2f}]".format(sharpe(bench), *boot_ci(bench, seed=1)))

rng = np.random.default_rng(2)
idx = np.arange(len(strat))
diffs = []
for _ in range(5000):
    t = rng.choice(idx, len(idx), replace=True)
    diffs.append(sharpe(strat[t]) - sharpe(bench[t]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print("Difference Sharpe {:.2f}  95% CI [{:.2f}, {:.2f}]".format(np.mean(diffs), lo, hi))