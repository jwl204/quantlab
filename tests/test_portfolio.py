import numpy as np
import pandas as pd

from backtest.portfolio import simulate_portfolio


def test_constant_prices_give_constant_nav():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0] * 5, "B": [20.0] * 5}, index=dates)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=dates)
    res = simulate_portfolio(prices, weights, [dates[0]], initial_cash=100.0, cost_bps=0.0)
    assert np.allclose(res["nav"], 100.0)


def test_single_asset_matches_the_asset():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.0, 13.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0] * 5}, index=dates)
    res = simulate_portfolio(prices, weights, [dates[0]], initial_cash=100.0, cost_bps=0.0)
    assert np.allclose(res["net_return"], prices["A"].pct_change().fillna(0.0))


def test_rebalance_to_same_weights_has_zero_turnover():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0] * 4, "B": [10.0] * 4}, index=dates)  # no drift
    weights = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=dates)
    res = simulate_portfolio(prices, weights, dates, initial_cash=100.0, cost_bps=0.0)
    assert np.allclose(res["turnover"].iloc[1:], 0.0)


def test_higher_cost_lowers_terminal_nav():
    dates = pd.date_range("2020-01-01", periods=10)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10)),
                           "B": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10))}, index=dates)
    weights = pd.DataFrame({"A": [0.5] * 10, "B": [0.5] * 10}, index=dates)
    low = simulate_portfolio(prices, weights, dates, 100.0, cost_bps=0.0)["nav"].iloc[-1]
    high = simulate_portfolio(prices, weights, dates, 100.0, cost_bps=50.0)["nav"].iloc[-1]
    assert high < low