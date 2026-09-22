import numpy as np


def sharpe(r, periods=252):
    """Annualised Sharpe ratio of a return series."""
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(periods)


def block_bootstrap_sharpe_diff(strat, bench, block=20, n=5000, seed=0):
    """Paired moving-block bootstrap of the Sharpe difference (strat - bench).

    Resamples the SAME contiguous blocks for both series, preserving serial
    dependence and keeping the two portfolios aligned in time.
    """
    strat, bench = np.asarray(strat), np.asarray(bench)
    T = len(strat)
    if block < 1 or block > T:
        raise ValueError("block length must be between 1 and the series length")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        diffs[i] = sharpe(strat[idx]) - sharpe(bench[idx])
    return diffs


def stationary_bootstrap_sharpe_diff(strat, bench, expected_block=20, n=5000, seed=0):
    """Paired stationary bootstrap (Politis-Romano) of the Sharpe difference.

    Uses randomly-sized contiguous blocks (geometric length, mean `expected_block`)
    wrapped circularly. A robustness check alongside the fixed-length block bootstrap.
    """
    strat, bench = np.asarray(strat), np.asarray(bench)
    T = len(strat)
    if expected_block < 1 or expected_block > T:
        raise ValueError("expected_block must be between 1 and the series length")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = []
        while len(idx) < T:
            start = int(rng.integers(0, T))
            length = int(rng.geometric(1.0 / expected_block))
            idx.extend(((start + np.arange(length)) % T).tolist())
        idx = np.asarray(idx[:T])
        diffs[i] = sharpe(strat[idx]) - sharpe(bench[idx])
    return diffs
