import numpy as np

from core.data import load_prices
from core.returns import simple_returns
from backtest.engine import backtest
from backtest.strategies import ma_trend_signal

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
    "  Sharpe {:.3f}".format(train_sharpes[best]),
)
print(
    "Same window on TEST (2020-23): Sharpe {:.3f}".format(strat_sharpe(best, test_mask))
)

bench = rets[test_mask]
print(
    "Buy-and-hold on TEST:          Sharpe {:.3f}".format(
        bench.mean() / bench.std() * np.sqrt(252)
    )
)
