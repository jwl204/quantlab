import numpy as np


def sharpe(r, periods=252):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(periods)


def block_bootstrap_sharpe_diff(strat, bench, block=20, n=5000, seed=0):
    """Paired moving-block bootstrap of the Sharpe difference (strat - bench).

    Resamples the SAME contiguous blocks for both series, preserving serial
    dependence and keeping the two portfolios aligned in time.
    """
    rng = np.random.default_rng(seed)
    strat, bench = np.asarray(strat), np.asarray(bench)
    T = len(strat)
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        diffs[i] = sharpe(strat[idx]) - sharpe(bench[idx])
    return diffs