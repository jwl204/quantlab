"""Semi-analytic European option prices under the Heston model.

Uses the Heston (1993) characteristic function in the numerically stable
"little Heston trap" form (Albrecher et al., 2007) and the two-probability
decomposition C = S0 * P1 - K exp(-rT) * P2. The probability integrals are
evaluated by fixed Gauss-Legendre quadrature on a truncated frequency domain,
so the pricer depends only on NumPy and is exact up to the quadrature tolerance
(no Monte Carlo error). It serves as the benchmark for the discretisation-error
study in ``heston_convergence.py``.
"""

from __future__ import annotations

import numpy as np


def _char_func(u, s0, v0, r, T, kappa, theta, xi, rho):
    """Characteristic function of ln(S_T): E[exp(i u ln S_T)] (little-trap form)."""
    i = 1j
    xi2 = xi * xi
    b = kappa - rho * xi * i * u
    d = np.sqrt(b**2 + xi2 * (i * u + u * u))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    C = r * i * u * T + (kappa * theta / xi2) * (
        (b - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
    )
    D = ((b - d) / xi2) * ((1.0 - edt) / (1.0 - g * edt))
    return np.exp(C + D * v0 + i * u * np.log(s0))


def heston_call_price(s0, k, v0, r, T, kappa, theta, xi, rho, u_max=200.0, n_nodes=256):
    """Semi-analytic European call price under Heston via Gauss-Legendre quadrature."""
    i = 1j
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * u_max * (x + 1.0)  # map [-1,1] -> [0, u_max]
    wu = 0.5 * u_max * w
    lnk = np.log(k)

    phi = _char_func(u, s0, v0, r, T, kappa, theta, xi, rho)
    phi_shift = _char_func(u - i, s0, v0, r, T, kappa, theta, xi, rho)
    forward = s0 * np.exp(r * T)  # equals _char_func(-i, ...)

    integ2 = np.real(np.exp(-i * u * lnk) * phi / (i * u))
    p2 = 0.5 + np.sum(wu * integ2) / np.pi
    integ1 = np.real(np.exp(-i * u * lnk) * phi_shift / (i * u * forward))
    p1 = 0.5 + np.sum(wu * integ1) / np.pi

    return float(s0 * p1 - k * np.exp(-r * T) * p2)


def heston_put_price(s0, k, v0, r, T, kappa, theta, xi, rho, u_max=200.0, n_nodes=256):
    """European put via put-call parity."""
    call = heston_call_price(s0, k, v0, r, T, kappa, theta, xi, rho, u_max, n_nodes)
    return call - s0 + k * np.exp(-r * T)
