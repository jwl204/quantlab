import numpy as np
import pandas as pd

from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.data import load_prices

tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "KO",
    "WMT",
    "NVDA",
    "META",
    "V",
    "HD",
    "DIS",
    "INTC",
    "CSCO",
    "PFE",
    "BA",
    "MCD",
]
start, end, cost_bps = "2015-01-01", "2024-01-01", 5.0
candidate_windows = [50, 100, 150, 200, 250]

prices = pd.DataFrame({tk: load_prices(tk, start, end) for tk in tickers}).dropna()
dates = prices.index
n = len(tickers)
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values


def strategy_returns(window):
    raw = pd.DataFrame({tk: ma_trend_signal(prices[tk], window) for tk in tickers})
    raw = raw.shift(1).fillna(0.0)
    w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return simulate_portfolio(prices, w, month_ends, cost_bps=cost_bps)["net_return"]


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


strat_by_w = {w: strategy_returns(w) for w in candidate_windows}
ew = pd.DataFrame(1.0 / n, index=dates, columns=tickers)
bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=cost_bps)["net_return"]

fold_edges = pd.date_range(dates[0], dates[-1], periods=6)
oos = []
print(f"{'Test period':24s} {'window':>6s} {'strat':>6s} {'BH':>6s} {'diff':>7s}")
for i in range(1, 6):
    test_start, test_end = fold_edges[i - 1], fold_edges[i]
    train_mask = dates < test_start
    test_mask = (dates >= test_start) & (dates < test_end)
    if train_mask.sum() < 252:
        continue
    best = max(candidate_windows, key=lambda w: sharpe(strat_by_w[w][train_mask]))
    test_r = strat_by_w[best][test_mask]
    oos.append(test_r)
    print(
        f"{test_start.date()!s} to {test_end.date()!s}  {best:>6d} "
        f"{sharpe(test_r):6.2f} {sharpe(bh[test_mask]):6.2f} {sharpe(test_r) - sharpe(bh[test_mask]):+7.2f}"
    )

oos_series = pd.concat(oos)
bh_oos = bh[oos_series.index]
print(
    f"\nStitched out-of-sample: strategy Sharpe {sharpe(oos_series):.3f}, "
    f"buy-and-hold {sharpe(bh_oos):.3f}, diff {sharpe(oos_series) - sharpe(bh_oos):+.3f}"
)
oos_series.to_csv("oos_ret.csv")
bh_oos.to_csv("bh_oos_ret.csv")
