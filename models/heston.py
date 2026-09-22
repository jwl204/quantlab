import numpy as np


def feller_ratio(kappa, theta, xi):
    """2*kappa*theta / xi^2. A value >= 1 (the Feller condition) means the CIR
    variance process stays strictly positive; below 1 it can hit zero."""
    return 2.0 * kappa * theta / xi**2


def simulate_heston(
    s0, v0, mu, kappa, theta, xi, rho, T, n_steps, n_paths, seed=None, scheme="euler"
):
    """Simulate Heston stochastic-volatility paths with full truncation of the variance.

    scheme:
        "euler"     - arithmetic price update S += mu*S*dt + sqrt(v)*S*dW (educational;
                      can distort the skew of log returns and let prices go negative).
        "log-euler" - log-price update d(logS) = (mu - v/2)*dt + sqrt(v)*dW, which keeps
                      prices positive and removes the arithmetic-scheme skew artifact.

    Returns (S, v): price and variance paths, each of shape (n_paths, n_steps + 1).
    """
    if kappa <= 0 or theta <= 0 or xi <= 0 or v0 < 0 or not (-1.0 <= rho <= 1.0):
        raise ValueError(
            "invalid Heston parameters: need kappa,theta,xi > 0, v0 >= 0, -1 <= rho <= 1"
        )
    if scheme not in ("euler", "log-euler"):
        raise ValueError("scheme must be 'euler' or 'log-euler'")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = s0
    v[:, 0] = v0
    log_s = np.full(n_paths, np.log(s0))

    for t in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dw1 = sqrt_dt * z1
        dw2 = sqrt_dt * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
        v_prev = np.maximum(v[:, t - 1], 0.0)
        vol = np.sqrt(v_prev)
        if scheme == "log-euler":
            log_s = log_s + (mu - 0.5 * v_prev) * dt + vol * dw1
            S[:, t] = np.exp(log_s)
        else:
            S[:, t] = S[:, t - 1] + mu * S[:, t - 1] * dt + vol * S[:, t - 1] * dw1
        v[:, t] = np.maximum(v_prev + kappa * (theta - v_prev) * dt + xi * vol * dw2, 0.0)

    return S, v
