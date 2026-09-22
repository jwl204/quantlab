import numpy as np

from pricing.black_scholes import bs_european_call
from pricing.monte_carlo import mc_call_terminal

params = {"s0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0}
bs = bs_european_call(**params)
N = 10000

plain = [mc_call_terminal(**params, n_paths=N, seed=s) - bs for s in range(300)]
anti = [mc_call_terminal(**params, n_paths=N, seed=s, antithetic=True) - bs for s in range(300)]

plain_rms = np.sqrt(np.mean(np.square(plain)))
anti_rms = np.sqrt(np.mean(np.square(anti)))

print("Plain RMS error:      ", plain_rms)
print("Antithetic RMS error: ", anti_rms)
print(f"Variance reduction factor: {(plain_rms / anti_rms) ** 2:.2f}x")
