import math

import numpy as np
import pytest

from pricing.heston_analytic import _char_func, heston_call_price, heston_put_price


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(s0, k, r, T, sig):
    d1 = (math.log(s0 / k) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return s0 * _ncdf(d1) - k * math.exp(-r * T) * _ncdf(d2)


def test_reduces_to_black_scholes_as_volofvol_vanishes():
    # xi -> 0 with v0 = theta collapses Heston to Black-Scholes at vol sqrt(theta)
    price = heston_call_price(100, 100, 0.04, 0.05, 1.0, 2.0, 0.04, 1e-5, -0.7)
    assert price == pytest.approx(bs_call(100, 100, 0.05, 1.0, 0.2), abs=1e-3)


def test_forward_identity():
    # characteristic function at u = -i equals the forward E[S_T] = s0 e^{rT}
    fwd = _char_func(np.array([-1j]), 100, 0.04, 0.05, 1.0, 2.0, 0.04, 0.5, -0.7)[0]
    assert fwd.real == pytest.approx(100 * math.exp(0.05), abs=1e-6)
    assert abs(fwd.imag) < 1e-6


def test_put_call_parity():
    args = (100, 100, 0.04, 0.05, 1.0, 2.0, 0.04, 0.5, -0.7)
    c = heston_call_price(*args)
    p = heston_put_price(*args)
    assert c - p == pytest.approx(100 - 100 * math.exp(-0.05), abs=1e-6)


def test_quadrature_is_converged():
    args = (100, 100, 0.04, 0.05, 1.0, 2.0, 0.04, 0.5, -0.7)
    assert heston_call_price(*args, n_nodes=128) == pytest.approx(
        heston_call_price(*args, n_nodes=256), abs=1e-6
    )
