import numpy as np
import pandas as pd

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.strategies import ma_trend_signal

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG", "KO",
           "WMT", "NVDA", "META", "V", "HD", "DIS", "INTC", "CSCO", "PFE", "BA", "MCD"]

start, end, window, cost_bps = "2015-01-01", "2024-01-01", 200, 5.0

strat_rets, bench_rets = {}, {}
for tk in tickers:
    prices = load_prices(tk, start, end)
    rets = simple_returns(prices)
    sig = ma_trend_signal(prices, window).reindex(rets.index).fillna(0.0)
    strat_rets[tk] = backtest(rets, sig, cost_bps=cost_bps)["strategy_return"]
    bench_rets[tk] = rets
    print("done", tk)

strat_port = pd.DataFrame(strat_rets).mean(axis=1).dropna()
bench_port = pd.DataFrame(bench_rets).mean(axis=1).dropna()

# save for the next step
strat_port.to_csv("strat_port.csv")
bench_port.to_csv("bench_port.csv")


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


print("\n20-stock equal-weight portfolio, 200-day trend:")
print("  Trend strategy Sharpe: {:.3f}".format(sharpe(strat_port)))
print("  Buy-and-hold  Sharpe: {:.3f}".format(sharpe(bench_port)))