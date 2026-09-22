import matplotlib.pyplot as plt
import numpy as np

from models.gbm import simulate_gbm

paths = simulate_gbm(
    s0=100, mu=0.10, sigma=0.20, T=1.0, n_steps=252, n_paths=20, seed=42
)

for path in paths:
    plt.plot(path, linewidth=1)

plt.title("Simulated GBM price paths (1 year)")
plt.xlabel("Trading day")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("reports/figures/gbm_paths.png", dpi=120)
plt.show()

many = simulate_gbm(
    s0=100, mu=0.10, sigma=0.20, T=1.0, n_steps=252, n_paths=20000, seed=1
)
final_prices = many[:, -1]

print("Simulated mean final price: ", final_prices.mean())
print("Theoretical mean (S0*e^muT):", 100 * np.exp(0.10 * 1.0))
