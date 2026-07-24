import matplotlib.pyplot as plt
import numpy as np

from models.heston import simulate_heston

S, v = simulate_heston(s0=100, v0=0.04, mu=0.10, kappa=3.0, theta=0.04,
                       xi=0.5, rho=-0.7, T=1.0, n_steps=252, n_paths=5, seed=42)

vol = np.sqrt(v)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

for path in S:
    ax1.plot(path, linewidth=1)
ax1.set_title("Heston price paths")
ax1.set_ylabel("Price")

for path in vol:
    ax2.plot(path, linewidth=1)
ax2.set_title("Corresponding volatility sqrt(v) over time")
ax2.set_ylabel("Volatility")
ax2.set_xlabel("Trading day")

plt.tight_layout()
plt.savefig("reports/figures/heston_paths.png", dpi=120)
plt.show()