import pandas as pd

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.metrics import performance
from backtest.strategies import ma_trend_signal

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = simple_returns(prices)

signal = ma_trend_signal(prices, window=200).reindex(rets.index).fillna(0.0)

strat = backtest(rets, signal, cost_bps=5.0)
bench = backtest(rets, pd.Series(1.0, index=rets.index), cost_bps=0.0)

print("200-day trend strategy (5 bps cost):")
for k, v in performance(strat).items():
    print("  {:16s}: {:.3f}".format(k, v))
print("\nBuy-and-hold benchmark:")
for k, v in performance(bench).items():
    print("  {:16s}: {:.3f}".format(k, v))
