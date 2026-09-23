"""Regenerate the *core* headline metrics and a provenance manifest, offline.

Covers the pricing validation, the AAPL stylised-fact moments and the basket
strategy-vs-buy-and-hold comparison with block-bootstrap inference -- the numbers
quoted in the README and final report. It deliberately does not re-run every study:
the walk-forward, sensitivity grids, ETF robustness universe, Heston convergence
and microstructure experiments have their own scripts (walk_forward.py,
sensitivity.py, etf_universe.py, heston_convergence.py, run_microstructure.py).
"""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels

from backtest.bootstrap import block_bootstrap_sharpe_diff, sharpe
from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.returns import log_returns
from core.snapshots import load_snapshot
from pricing.black_scholes import bs_european_call
from pricing.monte_carlo import mc_call_terminal

CONFIG = {"seed": 0, "snapshot": "basket", "window": 200, "cost_bps": 5.0}
results = {}

# 1. pricing validation (no data required)
p = {"s0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0}
mc = mc_call_terminal(**p, n_paths=200000, seed=CONFIG["seed"], antithetic=True)
bs = bs_european_call(**p)
results["mc_price"], results["bs_price"] = round(mc, 4), round(bs, 4)
results["pricing_rel_error_pct"] = round(abs(mc - bs) / bs * 100, 3)

# data from the immutable snapshot (offline-reproducible)
prices = load_snapshot(CONFIG["snapshot"])
tickers = list(prices.columns)
n = len(tickers)
dates = prices.index

# 2. empirical stylised facts
rets = log_returns(prices["AAPL"])
results["aapl_excess_kurtosis"] = round(float(rets.kurtosis()), 3)
results["aapl_skewness"] = round(float(rets.skew()), 3)

# 3. basket strategy vs true buy-and-hold
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values
ew = pd.DataFrame(1.0 / n, index=dates, columns=tickers)
bh_ret = simulate_portfolio(prices, ew, [dates[0]], cost_bps=CONFIG["cost_bps"])["net_return"]
raw = (
    pd.DataFrame({t: ma_trend_signal(prices[t], CONFIG["window"]) for t in tickers})
    .shift(1)
    .fillna(0.0)
)
w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
strat_ret = simulate_portfolio(prices, w, month_ends, cost_bps=CONFIG["cost_bps"])["net_return"]
d = block_bootstrap_sharpe_diff(
    strat_ret.to_numpy(), bh_ret.to_numpy(), block=20, n=5000, seed=CONFIG["seed"]
)
lo, hi = np.percentile(d, [2.5, 97.5])
results["strategy_sharpe"] = round(float(sharpe(strat_ret.to_numpy())), 3)
results["buy_and_hold_sharpe"] = round(float(sharpe(bh_ret.to_numpy())), 3)
results["sharpe_diff"] = round(results["strategy_sharpe"] - results["buy_and_hold_sharpe"], 3)
results["sharpe_diff_ci_low"] = float(round(lo, 3))
results["sharpe_diff_ci_high"] = float(round(hi, 3))
results["p_strategy_better"] = round(float((d > 0).mean()), 3)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


snap_manifest = json.loads((Path("snapshots") / f"{CONFIG['snapshot']}_manifest.json").read_text())
manifest = {
    "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    # The commit whose code produced these results. It is the commit that was HEAD
    # when the script ran, NOT the commit that stores this file -- committing the
    # manifest necessarily advances HEAD past it, so the two can never be identical.
    # Regenerate only when the code or data that affects the results changes.
    "results_source_commit": git_commit(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": {m.__name__: m.__version__ for m in [np, pd, scipy, statsmodels, matplotlib]},
    "data_snapshot": CONFIG["snapshot"],
    "data_snapshot_sha256": snap_manifest.get("sha256"),
    "config": CONFIG,
    "results": results,
}
with open("reports/run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
pd.Series(results).to_csv("reports/results.csv")
print(json.dumps(results, indent=2))
print("\nWrote reports/run_manifest.json and reports/results.csv (offline, from snapshot)")
