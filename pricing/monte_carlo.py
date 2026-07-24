import numpy as np

from models.gbm import simulate_gbm


def mc_european_call(s0, K, r, sigma, T, n_steps, n_paths, seed=None):
    """Price a European call option by Monte Carlo under GBM.

    s0: spot price. K: strike. r: risk-free rate. sigma: volatility.
    T: time to maturity (years). Returns the option price today.
    """
    paths = simulate_gbm(s0, r, sigma, T, n_steps, n_paths, seed)
    final_prices = paths[:, -1]
    payoffs = np.maximum(final_prices - K, 0.0)
    return np.exp(-r * T) * payoffs.mean()


def mc_call_terminal(s0, K, r, sigma, T, n_paths, seed=None, antithetic=False):
    """Price a European call by sampling the terminal price directly.
    Set antithetic=True to use variance reduction."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = rng.standard_normal(n_paths // 2)
        z = np.concatenate([half, -half])
    else:
        z = rng.standard_normal(n_paths)
    sT = s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(sT - K, 0.0)
    return np.exp(-r * T) * payoffs.mean()

def mc_asian_call(s0, K, r, sigma, T, n_steps, n_paths, seed=None):
    """Price an arithmetic-average Asian call by Monte Carlo (no closed form)."""
    paths = simulate_gbm(s0, r, sigma, T, n_steps, n_paths, seed)
    avg_price = paths[:, 1:].mean(axis=1)
    payoffs = np.maximum(avg_price - K, 0.0)
    return np.exp(-r * T) * payoffs.mean()