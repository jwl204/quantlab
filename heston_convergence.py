"""Heston discretisation-error study with common random numbers.

Weak (discretisation) error and Monte Carlo sampling error are separated cleanly:

* The benchmark is the *semi-analytic* Heston price (``pricing.heston_analytic``),
  which carries no Monte Carlo error at all.
* Every time-step grid is driven by the SAME underlying Brownian path: fine-grid
  increments are generated once and summed into the coarser grids' increments
  (nested / common random numbers). The discretisation error of a coarse grid is
  then estimated from the *paired* difference against the finest grid, whose
  standard error is tiny because the shared randomness cancels.
* For contrast the table also shows the ordinary (unpaired) Monte Carlo standard
  error of each price estimate -- the noise you would be fighting *without* common
  random numbers, and the reason an earlier version's "biases" were indistinguishable
  from sampling noise.

Run:  python heston_convergence.py
"""

from __future__ import annotations

import numpy as np

from models.heston import feller_ratio
from pricing.heston_analytic import heston_call_price

PARAMS = {"s0": 100.0, "v0": 0.04, "mu": 0.05, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
K, T, R = 100.0, 1.0, 0.05  # risk-neutral pricing: mu = r
N_PATHS = 40000
N_FINE = 1000
GRIDS = [25, 50, 100, 125, 250, 500, 1000]  # all divide N_FINE


def build_fine_increments(n_fine, n_paths, rho, dt_fine, seed):
    """Correlated Brownian increments on the finest grid, shape (n_paths, n_fine)."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_paths, n_fine))
    z2 = rng.standard_normal((n_paths, n_fine))
    dw1 = np.sqrt(dt_fine) * z1
    dw2 = np.sqrt(dt_fine) * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
    return dw1, dw2


def coarsen(dw_fine, n_coarse):
    """Aggregate fine increments into n_coarse increments by summing blocks."""
    n_paths, n_fine = dw_fine.shape
    block = n_fine // n_coarse
    return dw_fine.reshape(n_paths, n_coarse, block).sum(axis=2)


def simulate_terminal(dw1, dw2, scheme, params, T):
    """Terminal price under the same full-truncation scheme as models.heston."""
    n_paths, n_steps = dw1.shape
    dt = T / n_steps
    v = np.full(n_paths, params["v0"])
    if scheme == "log-euler":
        x = np.full(n_paths, np.log(params["s0"]))
    else:
        s = np.full(n_paths, params["s0"])
    for t in range(n_steps):
        v_prev = np.maximum(v, 0.0)
        vol = np.sqrt(v_prev)
        if scheme == "log-euler":
            x = x + (params["mu"] - 0.5 * v_prev) * dt + vol * dw1[:, t]
        else:
            s = s + params["mu"] * s * dt + vol * s * dw1[:, t]
        v = np.maximum(v_prev + params["kappa"] * (params["theta"] - v_prev) * dt
                       + params["xi"] * vol * dw2[:, t], 0.0)
    return np.exp(x) if scheme == "log-euler" else np.maximum(s, 0.0)


def discounted_payoff(terminal):
    return np.exp(-R * T) * np.maximum(terminal - K, 0.0)


def main() -> None:
    import matplotlib.pyplot as plt

    dt_fine = T / N_FINE
    fr = feller_ratio(PARAMS["kappa"], PARAMS["theta"], PARAMS["xi"])
    print(f"Feller ratio 2*kappa*theta/xi^2 = {fr:.3f} "
          f"({'satisfied' if fr >= 1 else 'violated: truncation active'})")

    exact = heston_call_price(PARAMS["s0"], K, PARAMS["v0"], R, T,
                              PARAMS["kappa"], PARAMS["theta"], PARAMS["xi"], PARAMS["rho"])
    print(f"Semi-analytic benchmark price: {exact:.4f}\n")

    dw1_fine, dw2_fine = build_fine_increments(N_FINE, N_PATHS, PARAMS["rho"], dt_fine, seed=12345)

    disc_abs = {}
    for scheme in ["euler", "log-euler"]:
        # finest-grid payoffs (shared path) anchor the paired discretisation error
        pay_fine = discounted_payoff(simulate_terminal(dw1_fine, dw2_fine, scheme, PARAMS, T))
        print(f"{scheme} scheme:")
        print(f"  {'steps':>5} {'price':>8} {'bias_vs_exact':>13} "
              f"{'disc_vs_finest':>15} {'95% CI':>17} {'MC_SE':>7}")
        disc_abs[scheme] = []
        for n in GRIDS:
            pay = discounted_payoff(
                simulate_terminal(coarsen(dw1_fine, n), coarsen(dw2_fine, n), scheme, PARAMS, T)
            )
            price = pay.mean()
            mc_se = pay.std(ddof=1) / np.sqrt(N_PATHS)          # unpaired MC error
            diff = pay - pay_fine                                 # common random numbers
            disc = diff.mean()
            disc_se = diff.std(ddof=1) / np.sqrt(N_PATHS)         # tiny, thanks to CRN
            ci = 1.96 * disc_se
            disc_abs[scheme].append(abs(disc) if n != N_FINE else np.nan)
            print(f"  {n:>5d} {price:8.4f} {price - exact:+13.4f} "
                  f"{disc:+15.4f} {f'+/-{ci:.4f}':>17} {mc_se:7.4f}")
        fine_bias = pay_fine.mean() - exact
        fine_se = pay_fine.std(ddof=1) / np.sqrt(N_PATHS)
        print(f"  finest ({N_FINE}) bias vs exact: {fine_bias:+.4f} +/- {1.96 * fine_se:.4f} "
              f"(MC 95% CI)\n")

    # --- plot: |discretisation error vs finest| with CRN, both schemes ---
    grids_plot = [n for n in GRIDS if n != N_FINE]
    for scheme in ["euler", "log-euler"]:
        y = disc_abs[scheme][: len(grids_plot)]
        plt.loglog(grids_plot, y, "o-", label=scheme)
    plt.xlabel("time steps")
    plt.ylabel("|discretisation error vs finest grid| (common random numbers)")
    plt.title("Heston weak convergence (Monte Carlo noise removed by CRN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/heston_convergence.png", dpi=120)
    print("Wrote reports/figures/heston_convergence.png")

    # --- scheme-dependent skew at rho = 0 (the true skew is small, NOT zero) ---
    # At rho = 0 the finite-horizon log return is still mildly asymmetric: the
    # -1/2 int v dt drift acts on a right-skewed integrated variance. So we do not
    # assume a zero truth; instead we show the arithmetic-Euler skew is inflated at
    # coarse steps and converges toward the log-Euler value as the step shrinks,
    # identifying the bulk of it as a discretisation artifact.
    p0 = dict(PARAMS, rho=0.0)
    print("\nDaily-return skew at rho=0 by scheme and intraday resolution "
          "(common random numbers):")
    base_days, n_paths_sk = 252, 4000
    for mult in (1, 2, 4, 8):
        n_steps = base_days * mult
        dw1, dw2 = build_fine_increments(n_steps, n_paths_sk, 0.0, T / n_steps, seed=7)
        row = []
        for scheme in ["euler", "log-euler"]:
            # record daily prices (every `mult` steps) and pool daily log returns
            v = np.full(n_paths_sk, p0["v0"])
            x = np.full(n_paths_sk, np.log(p0["s0"]))
            s = np.full(n_paths_sk, p0["s0"])
            daily = [np.full(n_paths_sk, p0["s0"])]
            dt = T / n_steps
            for t in range(n_steps):
                v_prev = np.maximum(v, 0.0)
                vol = np.sqrt(v_prev)
                if scheme == "log-euler":
                    x = x + (p0["mu"] - 0.5 * v_prev) * dt + vol * dw1[:, t]
                    cur = np.exp(x)
                else:
                    s = s + p0["mu"] * s * dt + vol * s * dw1[:, t]
                    cur = np.maximum(s, 0.0)
                v = np.maximum(v_prev + p0["kappa"] * (p0["theta"] - v_prev) * dt
                               + p0["xi"] * vol * dw2[:, t], 0.0)
                if (t + 1) % mult == 0:
                    daily.append(cur)
            prices = np.stack(daily, axis=1)
            rets = np.log(prices[:, 1:] / prices[:, :-1]).ravel()
            rets = rets[np.isfinite(rets)]
            m = rets.mean()
            skew = np.mean((rets - m) ** 3) / rets.std() ** 3
            row.append(skew)
        print(f"  {mult}x/day ({n_steps:>4} steps/yr): "
              f"euler {row[0]:+.3f}   log-euler {row[1]:+.3f}")


if __name__ == "__main__":
    main()
