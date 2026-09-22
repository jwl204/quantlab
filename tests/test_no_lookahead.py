import numpy as np
import pandas as pd

from backtest.engine import backtest
from backtest.strategies import ma_trend_signal


def test_future_prices_do_not_change_the_past():
    rng = np.random.default_rng(0)
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 500)),
        index=pd.date_range("2020-01-01", periods=500),
    )
    cut = 300

    def run(px):
        rets = px.pct_change().dropna()
        sig = ma_trend_signal(px, window=50).reindex(rets.index).fillna(0.0)
        return backtest(rets, sig, cost_bps=5.0)["strategy_return"]

    base = run(prices)

    mutated = prices.copy()
    mutated.iloc[cut:] *= rng.normal(1.0, 0.2, len(mutated) - cut)  # scramble the future
    mut = run(mutated)

    # everything comfortably before the cutoff must be identical
    assert np.allclose(base.iloc[: cut - 5], mut.iloc[: cut - 5])
