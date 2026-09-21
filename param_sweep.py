import pandas as pd
import matplotlib.pyplot as plt

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.metrics import performance
from backtest.strategies import ma_trend_signal

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = simple_returns(prices)

bench = performance(backtest(rets, pd.Series(1.0, index=rets.index)))["sharpe"]

windows = [20, 50, 100, 150, 200, 250]
sharpes = []
for w in windows:
    sig = ma_trend_signal(prices, w).reindex(rets.index).fillna(0.0)
    s = performance(backtest(rets, sig, cost_bps=5.0))["sharpe"]
    sharpes.append(s)
    print("window {:4d}: Sharpe {:.3f}".format(w, s))

plt.plot(windows, sharpes, "o-", label="Trend strategy")
plt.axhline(bench, color="red", linestyle="--", label="Buy-and-hold")
plt.xlabel("Moving-average window (days)")
plt.ylabel("Sharpe ratio")
plt.title("Trend strategy Sharpe vs window length (AAPL)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/trend_param_sweep.png", dpi=120)
plt.show()