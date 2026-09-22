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
start, end, window, cost_bps = "2015-01-01", "2024-01-01", 200, 5.0

prices = pd.DataFrame({tk: load_prices(tk, start, end) for tk in tickers}).dropna()
dates = prices.index
n = len(tickers)
years = len(dates) / 252
month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values

ew = pd.DataFrame(1.0 / n, index=dates, columns=tickers)

# 1. true buy-and-hold: buy equal weight once, never trade again
bh = simulate_portfolio(prices, ew, [dates[0]], cost_bps=cost_bps)

# 2. monthly-rebalanced equal weight
reb = simulate_portfolio(prices, ew, month_ends, cost_bps=cost_bps)

# 3. trend strategy: equal weight among stocks in an uptrend, lagged, monthly
raw = pd.DataFrame({tk: ma_trend_signal(prices[tk], window) for tk in tickers}).shift(1).fillna(0.0)
w = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
strat = simulate_portfolio(prices, w, month_ends, cost_bps=cost_bps)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


print("{:26s} {:>7s} {:>10s} {:>10s}".format("Portfolio", "Sharpe", "turnover/yr", "final NAV"))
for name, res in [
    ("True buy-and-hold", bh),
    ("Monthly-rebalanced EW", reb),
    ("Trend strategy", strat),
]:
    r = res["net_return"]
    print(
        "{:26s} {:7.3f} {:10.2f} {:10.2f}".format(
            name, sharpe(r), res["turnover"].sum() / years, res["nav"].iloc[-1]
        )
    )

bh["net_return"].to_csv("bh_ret.csv")
reb["net_return"].to_csv("reb_ret.csv")
strat["net_return"].to_csv("strat_ret.csv")
