import itertools

import numpy as np
import pandas as pd

from backtest.walk_forward import run_walk_forward


def _prices(seed=0, n=600, k=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {f"S{j}": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for j in range(k)},
        index=idx,
    )


def test_folds_ordered_and_non_overlapping():
    _, folds = run_walk_forward(_prices(), [20, 50], n_folds=3, cost_bps=5.0, min_train=120)
    for a, b in itertools.pairwise(folds):
        assert a["test_end"] <= b["test_start"]
        assert a["test_start"] < a["test_end"]


def test_stitched_nav_matches_compounded_returns():
    portfolio, _ = run_walk_forward(_prices(), [20, 50], n_folds=3, cost_bps=5.0, min_train=120)
    compounded = (1.0 + portfolio["net_return"]).cumprod()
    assert np.allclose(portfolio["nav"].to_numpy(), compounded.to_numpy())


def test_future_mutation_does_not_change_the_past():
    prices = _prices()
    port, _ = run_walk_forward(prices, [20, 50], n_folds=3, cost_bps=5.0, min_train=120)
    cut = int(len(prices) * 0.8)
    cutoff = prices.index[cut]
    rng = np.random.default_rng(9)
    mutated = prices.copy()
    mutated.iloc[cut:] *= rng.normal(1.0, 0.2, (len(prices) - cut, prices.shape[1]))
    port2, _ = run_walk_forward(mutated, [20, 50], n_folds=3, cost_bps=5.0, min_train=120)
    r1 = port["net_return"][port.index < cutoff].to_numpy()
    r2 = port2["net_return"][port2.index < cutoff].to_numpy()
    k = len(r1) - 5
    assert np.allclose(r1[:k], r2[:k])
