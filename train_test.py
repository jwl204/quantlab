import numpy as np

from backtest.engine import backtest
from backtest.strategies import ma_trend_signal
from core.data import load_prices
from core.returns import simple_returns

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = simple_returns(prices)
split = "2020-01-01"
train_mask = rets.index < split
test_mask = rets.index >= split


def strat_sharpe(window, mask):
    sig = ma_trend_signal(prices, window).reindex(rets.index).fillna(0.0)
    r = backtest(rets, sig, cost_bps=5.0)["strategy_return"][mask]
    return r.mean() / r.std() * np.sqrt(252)


windows = [20, 50, 100, 150, 200, 250]
train_sharpes = {w: strat_sharpe(w, train_mask) for w in windows}
best = max(train_sharpes, key=train_sharpes.get)

print(
    "Chosen on TRAIN (2015-2019):  window",
    best,
    f"  Sharpe {train_sharpes[best]:.3f}",
)
print(f"Same window on TEST (2020-23): Sharpe {strat_sharpe(best, test_mask):.3f}")

bench = rets[test_mask]
print(f"Buy-and-hold on TEST:          Sharpe {bench.mean() / bench.std() * np.sqrt(252):.3f}")
