from pricing.black_scholes import bs_european_call
from pricing.monte_carlo import mc_asian_call, mc_european_call

params = {"s0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0}

mc = mc_european_call(**params, n_steps=252, n_paths=50000, seed=7)
bs = bs_european_call(**params)

print("Monte Carlo price:  ", mc)
print("Black-Scholes price:", bs)
print("Difference:         ", abs(mc - bs))
print(f"Relative error:      {abs(mc - bs) / bs:.2%}")

asian = mc_asian_call(s0=100, K=100, r=0.05, sigma=0.20, T=1.0, n_steps=252, n_paths=50000, seed=7)
print("Monte Carlo Asian call price:", asian)
