import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from models.heston import simulate_heston

S, v = simulate_heston(s0=100, v0=0.04, mu=0.05, kappa=3.0, theta=0.04,
                       xi=0.5, rho=-0.7, T=10.0, n_steps=2520, n_paths=1, seed=1)

prices = S[0]
rets = np.log(prices[1:] / prices[:-1])

print("Heston excess kurtosis:", stats.kurtosis(rets))
print("Heston skewness:       ", stats.skew(rets))
print("(GBM / normal would give ~0 for both)")

plot_acf(rets ** 2, lags=40,
         title="Heston: ACF of squared returns (volatility clustering)")
plt.tight_layout()
plt.savefig("reports/figures/heston_acf_squared.png", dpi=120)
plt.show()