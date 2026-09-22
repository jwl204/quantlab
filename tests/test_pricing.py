from pricing.black_scholes import bs_european_call
from pricing.monte_carlo import mc_call_terminal, mc_asian_call


def test_mc_matches_black_scholes():
    params = dict(s0=100, K=100, r=0.05, sigma=0.20, T=1.0)
    mc = mc_call_terminal(**params, n_paths=200000, seed=0, antithetic=True)
    bs = bs_european_call(**params)
    assert abs(mc - bs) < 0.1


def test_asian_is_cheaper_than_european():
    params = dict(s0=100, K=100, r=0.05, sigma=0.20, T=1.0)
    euro = bs_european_call(**params)
    asian = mc_asian_call(**params, n_steps=100, n_paths=50000, seed=0)
    assert asian < euro
