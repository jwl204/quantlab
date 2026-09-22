import numpy as np
import pandas as pd

from backtest.portfolio import simulate_portfolio
from backtest.walk_forward import run_walk_forward, sharpe
from core.snapshots import load_snapshot

prices = load_snapshot("basket")
dates = prices.index
n = prices.shape[1]

portfolio, folds = run_walk_forward(prices, [50, 100, 150, 200, 250], n_folds=5, cost_bps=5.0)
strat = portfolio["net_return"]

first_test = folds[0]["test_start"]
oos = dates >= np.datetime64(first_test)
oos_start = dates[oos][0]
ew = pd.DataFrame(1.0 / n, index=dates, columns=prices.columns)
bh = simulate_portfolio(prices, ew, [oos_start], cost_bps=5.0)["net_return"]

print(f"{'Test period':24s} {'window':>6s} {'strat':>6s} {'BH':>6s} {'diff':>7s}")
for f in folds:
    m = (dates >= f["test_start"]) & (dates < f["test_end"])
    print(
        f"{f['test_start'].date()!s} to {f['test_end'].date()!s}  {f['selected_window']:>6d} "
        f"{sharpe(strat[m]):6.2f} {sharpe(bh[m]):6.2f} {sharpe(strat[m]) - sharpe(bh[m]):+7.2f}"
    )

so, bo = strat[oos], bh[oos]
print(
    f"\nStateful stitched OOS: strategy Sharpe {sharpe(so):.3f}, "
    f"buy-and-hold {sharpe(bo):.3f}, diff {sharpe(so) - sharpe(bo):+.3f}"
)
so.to_csv("oos_ret.csv")
bo.to_csv("bh_oos_ret.csv")
