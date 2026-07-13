import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from core.data import load_prices
from core.returns import log_returns

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = log_returns(prices)

print("Observations:   ", len(rets))
print("Mean (daily):   ", rets.mean())
print("Std (daily):    ", rets.std())
print("Skewness:       ", rets.skew())
print("Excess kurtosis:", rets.kurtosis())

mu, sigma = rets.mean(), rets.std()
x = np.linspace(rets.min(), rets.max(), 200)
normal_curve = stats.norm.pdf(x, mu, sigma)

plt.hist(rets, bins=100, density=True, alpha=0.6, label="Actual AAPL returns")
plt.plot(x, normal_curve, "r-", linewidth=2, label="Normal distribution")
plt.title("AAPL daily returns vs a normal distribution")
plt.xlabel("Daily log return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/1_hist_vs_normal.png", dpi=120) 
plt.show()

plt.figure()
stats.probplot(rets, dist="norm", plot=plt)
plt.title("Q-Q plot: AAPL returns vs normal")
plt.tight_layout()
plt.savefig("reports/figures/2_qq_plot.png", dpi=120)  
plt.show()

plt.figure()
rets.plot(title="AAPL daily returns over time (volatility clustering)")
plt.xlabel("Date")
plt.ylabel("Daily log return")
plt.axhline(0, color="black", linewidth=0.6)
plt.tight_layout()
plt.savefig("reports/figures/3_returns_timeseries.png", dpi=120) 
plt.show()

plot_acf(rets, lags=40, title="ACF of returns (can we predict direction?)")
plt.tight_layout()
plt.savefig("reports/figures/4_acf_returns.png", dpi=120)
plt.show()

plot_acf(rets**2, lags=40, title="ACF of squared returns ( volatility?)")
plt.tight_layout()
plt.savefig("reports/figures/5_acf_squared_returns.png", dpi=120)
plt.show()

t_params = stats.t.fit(rets)
print("Student-t fit (df, loc, scale):", t_params)
print("Degrees of freedom:", t_params[0])

t_curve = stats.t.pdf(x, *t_params)

plt.figure()
plt.hist(rets, bins=100, density=True, alpha=0.6, label="Actual AAPL returns")
plt.plot(x, normal_curve, "r-", linewidth=2, label="Normal")
plt.plot(x, t_curve, "g-", linewidth=2, label="Student-t (fitted)")
plt.title("AAPL returns: normal vs Student-t")
plt.xlabel("Daily log return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/6_normal_vs_t.png", dpi=120)   
plt.show()