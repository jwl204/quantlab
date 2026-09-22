import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from core.data import load_prices
from core.returns import log_returns
from models.heston import simulate_heston
from models.calibration import calibrate_gbm, calibrate_heston

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = log_returns(prices)
mu, sigma = calibrate_gbm(rets)
hp = calibrate_heston(rets)

n = len(rets)
S, v = simulate_heston(
    s0=100,
    v0=hp["v0"],
    mu=mu,
    kappa=hp["kappa"],
    theta=hp["theta"],
    xi=hp["xi"],
    rho=hp["rho"],
    T=n / 252,
    n_steps=n,
    n_paths=20,
    seed=1,
)
heston_rets = np.log(S[:, 1:] / S[:, :-1]).flatten()

print("Real AAPL   excess kurtosis:", stats.kurtosis(rets))
print("Heston      excess kurtosis:", stats.kurtosis(heston_rets))
print("Real AAPL   skewness:       ", stats.skew(rets))
print("Heston      skewness:       ", stats.skew(heston_rets))

plt.hist(rets, bins=100, density=True, alpha=0.5, label="Real AAPL returns")
plt.hist(
    heston_rets, bins=200, density=True, alpha=0.5, label="Calibrated Heston returns"
)
plt.xlim(-0.15, 0.15)
plt.title("Real vs calibrated-Heston return distributions")
plt.xlabel("Daily log return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/heston_goodness_of_fit.png", dpi=120)
plt.show()
