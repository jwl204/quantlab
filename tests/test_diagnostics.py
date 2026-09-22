import numpy as np

from backtest.diagnostics import deflated_sharpe_ratio, probabilistic_sharpe_ratio


def test_psr_increases_with_stronger_returns():
    rng = np.random.default_rng(0)
    weak = rng.normal(0.0001, 0.01, 1000)
    strong = rng.normal(0.0015, 0.01, 1000)
    assert probabilistic_sharpe_ratio(strong) > probabilistic_sharpe_ratio(weak)


def test_deflated_sharpe_in_unit_interval():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    trials = [rng.normal(0, 0.01, 1000).mean() / 0.01 for _ in range(10)]
    dsr = deflated_sharpe_ratio(r, trials, n_trials=10)
    assert 0.0 <= dsr <= 1.0
