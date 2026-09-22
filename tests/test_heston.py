import numpy as np
import pytest

from models.heston import feller_ratio, simulate_heston


def test_variance_reverts_to_theta():
    _S, v = simulate_heston(
        s0=100,
        v0=0.04,
        mu=0.0,
        kappa=3.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        T=20.0,
        n_steps=5040,
        n_paths=200,
        seed=0,
    )
    assert abs(v[:, 2520:].mean() - 0.04) < 0.01


def test_heston_returns_have_fat_tails():
    from scipy import stats

    S, _v = simulate_heston(
        s0=100,
        v0=0.04,
        mu=0.0,
        kappa=3.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        T=10.0,
        n_steps=2520,
        n_paths=1,
        seed=1,
    )
    rets = np.log(S[0][1:] / S[0][:-1])
    assert stats.kurtosis(rets) > 0.5


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        simulate_heston(
            s0=100,
            v0=0.04,
            mu=0.0,
            kappa=-1.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            T=1.0,
            n_steps=10,
            n_paths=5,
        )


def test_log_euler_keeps_prices_positive():
    S, _v = simulate_heston(
        s0=100,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.9,
        rho=-0.7,
        T=1.0,
        n_steps=100,
        n_paths=500,
        seed=0,
        scheme="log-euler",
    )
    assert (S > 0).all()


def test_feller_ratio():
    assert feller_ratio(2.0, 0.04, 0.4) == pytest.approx(2 * 2.0 * 0.04 / 0.16)
