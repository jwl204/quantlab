import numpy as np
import pandas as pd

from backtest.engine import backtest


def test_buy_and_hold_reproduces_asset():
    rets = pd.Series([0.01, -0.02, 0.03, 0.005])
    signal = pd.Series(1.0, index=rets.index)
    result = backtest(rets, signal, cost_bps=0.0)
    # from the second day on, the position is 1, so returns match the asset
    assert np.allclose(result["strategy_return"].iloc[1:], rets.iloc[1:])


def test_costs_reduce_returns():
    rets = pd.Series([0.01, 0.01, 0.01, 0.01])
    signal = pd.Series([1.0, 0.0, 1.0, 0.0], index=rets.index)  # trades a lot
    gross = backtest(rets, signal, cost_bps=0.0)["strategy_return"].sum()
    net = backtest(rets, signal, cost_bps=10.0)["strategy_return"].sum()
    assert net < gross
