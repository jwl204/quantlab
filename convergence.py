import numpy as np
import matplotlib.pyplot as plt

from pricing.monte_carlo import mc_european_call
from pricing.black_scholes import bs_european_call

params = dict(s0=100, K=100, r=0.05, sigma=0.20, T=1.0)
bs = bs_european_call(**params)

path_counts = [100, 300, 1000, 3000, 10000, 30000]
rms_errors = []
for n in path_counts:
    errs = [
        mc_european_call(**params, n_steps=100, n_paths=n, seed=s) - bs
        for s in range(10)
    ]
    rms_errors.append(np.sqrt(np.mean(np.square(errs))))

plt.loglog(path_counts, rms_errors, "o-", label="RMS Monte Carlo error")
reference = rms_errors[0] * np.sqrt(path_counts[0] / np.array(path_counts))
plt.loglog(path_counts, reference, "r--", label="1/sqrt(N) reference")
plt.xlabel("Number of paths N")
plt.ylabel("RMS pricing error")
plt.title("Monte Carlo convergence: error proportional to 1/sqrt(N)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/mc_convergence.png", dpi=120)
plt.show()
