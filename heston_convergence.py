import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from models.heston import feller_ratio, simulate_heston

PARAMS = {"s0": 100, "v0": 0.04, "mu": 0.05, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
K, T, r = 100.0, 1.0, 0.05  # risk-neutral: mu = r
N_PATHS = 10000
STEPS = [25, 50, 100, 252, 500, 1000]

fr = feller_ratio(PARAMS["kappa"], PARAMS["theta"], PARAMS["xi"])
print(
    f"Feller ratio 2*kappa*theta/xi^2 = {fr:.3f} "
    f"({'satisfied' if fr >= 1 else 'violated: variance hits zero, truncation active'})"
)


def price(n_steps, scheme, seed=0):
    t0 = time.perf_counter()
    S, v = simulate_heston(
        **PARAMS, T=T, n_steps=n_steps, n_paths=N_PATHS, seed=seed, scheme=scheme
    )
    runtime = time.perf_counter() - t0
    payoff = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0)
    return payoff.mean(), payoff.std() / np.sqrt(N_PATHS), runtime, float((v == 0).mean())


ref, _, _, _ = price(1000, "log-euler", seed=1)
print(f"\nReference call price (log-euler, 1000 steps): {ref:.4f}\n")

bias = {}
for scheme in ["euler", "log-euler"]:
    print(f"{scheme} scheme:")
    print(f"  {'steps':>6s} {'price':>8s} {'SE':>7s} {'bias':>8s} {'trunc%':>7s} {'sec':>6s}")
    bias[scheme] = []
    for n in STEPS:
        pr, se, rt, tr = price(n, scheme)
        bias[scheme].append(abs(pr - ref))
        print(f"  {n:>6d} {pr:8.4f} {se:7.4f} {pr - ref:+8.4f} {100 * tr:7.2f} {rt:6.2f}")
    print()

for scheme in ["euler", "log-euler"]:
    plt.loglog(STEPS, bias[scheme], "o-", label=f"{scheme}")
plt.xlabel("time steps")
plt.ylabel("|price bias| vs reference")
plt.title("Heston discretisation: bias vs number of time steps")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/heston_convergence.png", dpi=120)
print("Wrote reports/figures/heston_convergence.png")

# --- The scheme's real effect: the skew of log returns. With rho = 0 the true model
# skew is zero, so any nonzero skew is a discretisation artifact. ---
p0 = dict(PARAMS)
p0["rho"] = 0.0
print("\nSkew of daily log returns with rho = 0 (true model skew is 0):")
for scheme in ["euler", "log-euler"]:
    S_s, _ = simulate_heston(**p0, T=10.0, n_steps=2520, n_paths=50, seed=2, scheme=scheme)
    rets = np.log(S_s[:, 1:] / S_s[:, :-1]).ravel()
    print(f"  {scheme:9s}: skew {stats.skew(rets):+.3f}  (positive prices: {(S_s > 0).all()})")
