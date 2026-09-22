import numpy as np

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.strategies import ma_trend_signal

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = simple_returns(prices)
test_mask = rets.index >= "2020-01-01"

sig = ma_trend_signal(prices, 20).reindex(rets.index).fillna(0.0)
strat_r = backtest(rets, sig, cost_bps=5.0)["strategy_return"][test_mask].to_numpy()
bench_r = rets[test_mask].to_numpy()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def boot_ci(r, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [sharpe(rng.choice(r, len(r), replace=True)) for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


print(
    "Strategy  Sharpe {:.2f}  95% CI [{:.2f}, {:.2f}]".format(
        sharpe(strat_r), *boot_ci(strat_r)
    )
)
print(
    "Benchmark Sharpe {:.2f}  95% CI [{:.2f}, {:.2f}]".format(
        sharpe(bench_r), *boot_ci(bench_r, seed=1)
    )
)

rng = np.random.default_rng(2)
idx = np.arange(len(strat_r))
diffs = []
for _ in range(5000):
    take = rng.choice(idx, len(idx), replace=True)
    diffs.append(sharpe(strat_r[take]) - sharpe(bench_r[take]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(
    "Sharpe difference {:.2f}  95% CI [{:.2f}, {:.2f}]".format(np.mean(diffs), lo, hi)
)
