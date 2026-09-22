import numpy as np
import pytest

from backtest.bootstrap import (
    block_bootstrap_sharpe_diff,
    sharpe,
    stationary_bootstrap_sharpe_diff,
)


def _pair(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)


def test_block_bootstrap_is_deterministic_and_shaped():
    a, b = _pair()
    d1 = block_bootstrap_sharpe_diff(a, b, block=20, n=1000, seed=1)
    d2 = block_bootstrap_sharpe_diff(a, b, block=20, n=1000, seed=1)
    assert d1.shape == (1000,)
    assert np.allclose(d1, d2)


def test_invalid_block_raises():
    a, b = _pair(n=50)
    with pytest.raises(ValueError):
        block_bootstrap_sharpe_diff(a, b, block=100, n=10, seed=0)


def test_stationary_bootstrap_paired_alignment():
    a, _ = _pair()
    d = stationary_bootstrap_sharpe_diff(a, a.copy(), expected_block=15, n=300, seed=0)
    assert np.allclose(d, 0.0)  # identical series -> difference exactly zero


def test_sharpe_zero_variance_not_finite():
    with np.errstate(invalid="ignore", divide="ignore"):
        val = sharpe(np.zeros(100))
    assert not np.isfinite(val)
