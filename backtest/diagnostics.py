"""Multiple-testing diagnostics: the Probabilistic and Deflated Sharpe Ratios.

The Deflated Sharpe Ratio (Bailey and Lopez de Prado) adjusts an observed Sharpe for
the number of strategies tried, non-normal returns and sample length, deflating an
in-sample Sharpe to reflect the chance of finding it by luck.
"""

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def _sr_moments(returns):
    r = np.asarray(returns, dtype=float)
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd  # per-period (non-annualised) Sharpe
    g3 = float(((r - mu) ** 3).mean() / sd**3)
    g4 = float(((r - mu) ** 4).mean() / sd**4)
    return sr, g3, g4, T


def probabilistic_sharpe_ratio(returns, sr_star=0.0):
    """P(true per-period Sharpe > sr_star), accounting for skew, kurtosis and length."""
    sr, g3, g4, T = _sr_moments(returns)
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2)
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / denom))


def deflated_sharpe_ratio(returns, trial_sharpes, n_trials=None):
    """Deflated Sharpe: PSR of `returns` against the expected best Sharpe under the null.

    `trial_sharpes` are the per-period Sharpe ratios of every strategy variant tried.
    """
    sr_trials = np.asarray(trial_sharpes, dtype=float)
    n = n_trials or len(sr_trials)
    var_sr = sr_trials.var(ddof=1)
    sr0 = np.sqrt(var_sr) * (
        (1.0 - EULER) * norm.ppf(1.0 - 1.0 / n) + EULER * norm.ppf(1.0 - 1.0 / (n * np.e))
    )
    return probabilistic_sharpe_ratio(returns, sr_star=sr0)
