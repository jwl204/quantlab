import numpy as np
import pandas as pd

from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal
from core.snapshots import load_snapshot

window, cost_bps = 200, 5.0
prices = load_snapshot("basket")
dates = prices.index
tickers = list(prices.columns)
n = len(tickers)
years = len(dates) / 252
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values

ew = pd.DataFrame(1.0 / n, index=dates, columns=tickers)
bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=cost_bps)
reb = simulate_portfolio(prices, ew, month_ends, cost_bps=cost_bps)
raw = pd.DataFrame({t: ma_trend_signal(prices[t], window) for t in tickers}).shift(1).fillna(0.0)
w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
strat = simulate_portfolio(prices, w, month_ends, cost_bps=cost_bps)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


print(f"{'Portfolio':26s} {'Sharpe':>7s} {'turnover/yr':>12s} {'final NAV':>10s}")
for name, res in [
    ("True buy-and-hold", bh),
    ("Monthly-rebalanced EW", reb),
    ("Trend strategy", strat),
]:
    r = res["net_return"]
    print(
        f"{name:26s} {sharpe(r):7.3f} {res['turnover'].sum() / years:12.2f} {res['nav'].iloc[-1]:10.2f}"
    )

bh["net_return"].to_csv("bh_ret.csv")
reb["net_return"].to_csv("reb_ret.csv")
strat["net_return"].to_csv("strat_ret.csv")
