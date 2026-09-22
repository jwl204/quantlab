import numpy as np


def simulate_gbm(s0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Simulate geometric Brownian motion paths via Euler-Maruyama.

    s0: starting price. mu: annual drift. sigma: annual volatility.
    T: horizon in years. n_steps: number of time steps. n_paths: number of paths.
    Returns an array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        prev = paths[:, t - 1]
        paths[:, t] = prev + mu * prev * dt + sigma * prev * np.sqrt(dt) * z
    return paths
