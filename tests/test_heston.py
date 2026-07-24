import numpy as np
from scipy import stats

from models.heston import simulate_heston


def test_variance_reverts_to_theta():
    S, v = simulate_heston(s0=100, v0=0.04, mu=0.0, kappa=3.0, theta=0.04,
                           xi=0.5, rho=-0.7, T=20.0, n_steps=5040,
                           n_paths=200, seed=0)
    long_run = v[:, 2520:].mean()
    assert abs(long_run - 0.04) < 0.01


def test_heston_returns_have_fat_tails():
    S, v = simulate_heston(s0=100, v0=0.04, mu=0.0, kappa=3.0, theta=0.04,
                           xi=0.5, rho=-0.7, T=10.0, n_steps=2520,
                           n_paths=1, seed=1)
    rets = np.log(S[0][1:] / S[0][:-1])
    assert stats.kurtosis(rets) > 0.5