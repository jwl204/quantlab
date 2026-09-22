import numpy as np
import pandas as pd

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.metrics import performance

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = simple_returns(prices)

signal = pd.Series(1.0, index=rets.index)  # always fully long
result = backtest(rets, signal, cost_bps=0.0)

# From the second day on, the position is 1, so the strategy return
# must exactly equal the asset return.
match = np.allclose(result["strategy_return"].iloc[1:], rets.iloc[1:])
print("Strategy reproduces the asset (from day 2):", match)

print("Final strategy equity:", result["equity"].iloc[-1])
print("Final buy-and-hold:   ", (1 + rets).cumprod().iloc[-1])

print("\nBuy-and-hold Apple performance:")
for name, value in performance(result).items():
    print("  {:16s}: {:.3f}".format(name, value))
