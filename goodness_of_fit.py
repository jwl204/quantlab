import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from core.data import load_prices
from core.returns import log_returns
from models.gbm import simulate_gbm
from models.calibration import calibrate_gbm

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = log_returns(prices)
mu, sigma = calibrate_gbm(rets)

n = len(rets)
S = simulate_gbm(s0=100, mu=mu, sigma=sigma, T=n / 252,
                 n_steps=n, n_paths=1, seed=1)
gbm_rets = np.log(S[0][1:] / S[0][:-1])

print("Real AAPL excess kurtosis:      ", stats.kurtosis(rets))
print("Calibrated-GBM excess kurtosis: ", stats.kurtosis(gbm_rets))

plt.hist(rets, bins=100, density=True, alpha=0.5, label="Real AAPL returns")
plt.hist(gbm_rets, bins=100, density=True, alpha=0.5, label="Calibrated GBM returns")
plt.title("Real vs calibrated-GBM return distributions")
plt.xlabel("Daily log return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/gbm_goodness_of_fit.png", dpi=120)
plt.show()