import numpy as np


def calibrate_gbm(log_rets, periods_per_year=252):
    """Calibrate GBM parameters (mu, sigma) to log returns by maximum likelihood.

    Returns annualised (mu, sigma).
    """
    m = log_rets.mean()
    s = log_rets.std(ddof=0)          # MLE variance divides by N (ddof=0)
    sigma = s * np.sqrt(periods_per_year)
    mu = m * periods_per_year + 0.5 * sigma ** 2
    return mu, sigma

def calibrate_heston(log_rets, window=21, periods_per_year=252):
    """Simplified Heston calibration from historical returns via realized variance."""
    dt = 1.0 / periods_per_year
    rv = (log_rets.rolling(window).std(ddof=0) ** 2 * periods_per_year).dropna()

    theta = rv.mean()                    # long-run variance level
    v0 = rv.iloc[-1]                     # current variance
    acf1 = rv.autocorr(lag=1)           # how persistent the variance is
    kappa = -np.log(acf1) / dt          # mean-reversion speed
    xi = np.sqrt(2 * kappa * rv.var(ddof=0) / theta)   # vol of vol (CIR moment)

    vol_change = np.sqrt(rv).diff().dropna()
    aligned_rets = log_rets.loc[vol_change.index]
    rho = np.corrcoef(aligned_rets, vol_change)[0, 1]  # leverage correlation

    return dict(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)