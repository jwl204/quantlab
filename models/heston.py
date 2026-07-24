import numpy as np


def simulate_heston(s0, v0, mu, kappa, theta, xi, rho, T,
                    n_steps, n_paths, seed=None):
    """Simulate Heston stochastic-volatility paths via Euler-Maruyama.

    s0: start price. v0: start variance. mu: drift.
    kappa: mean-reversion speed. theta: long-run variance.
    xi: vol-of-vol. rho: price/variance correlation.
    Returns (S, v): price paths and variance paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = s0
    v[:, 0] = v0
    for t in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dw1 = np.sqrt(dt) * z1
        dw2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
        v_prev = np.maximum(v[:, t - 1], 0.0)
        S[:, t] = S[:, t - 1] + mu * S[:, t - 1] * dt \
            + np.sqrt(v_prev) * S[:, t - 1] * dw1
        v[:, t] = v[:, t - 1] + kappa * (theta - v_prev) * dt \
            + xi * np.sqrt(v_prev) * dw2
    return S, v