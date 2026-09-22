import numpy as np
import pandas as pd
import pytest

from backtest.portfolio import simulate_portfolio


def _rw(seed, n):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0, 0.01, n))


def test_initial_investment_identity():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0] * 3, "B": [20.0] * 3}, index=dates)
    w = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=dates)
    res = simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0, cost_bps=10.0)
    # post-trade holdings + cash + paid cost == initial capital
    assert np.isclose(res["nav"].iloc[0] + res["cost"].iloc[0], 100.0)


def test_no_leverage_cash_nonnegative():
    dates = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"A": _rw(0, 6), "B": _rw(1, 6)}, index=dates)
    w = pd.DataFrame({"A": [0.5] * 6, "B": [0.5] * 6}, index=dates)
    res = simulate_portfolio(prices, w, dates, initial_cash=100.0, cost_bps=20.0)
    assert (res["cash"] >= -1e-9).all()


def test_entry_cost_appears_once_in_returns():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0] * 3}, index=dates)
    w = pd.DataFrame({"A": [1.0] * 3}, index=dates)
    res = simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0, cost_bps=50.0)
    assert res["net_return"].iloc[0] < 0  # the entry cost
    assert np.allclose(res["net_return"].iloc[1:], 0.0)  # constant prices, no more trades


def test_single_asset_matches_asset():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.0, 13.0]}, index=dates)
    w = pd.DataFrame({"A": [1.0] * 5}, index=dates)
    res = simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0, cost_bps=0.0)
    assert np.allclose(res["net_return"], prices["A"].pct_change().fillna(0.0))


def test_rebalance_same_weights_zero_turnover():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0] * 4, "B": [10.0] * 4}, index=dates)
    w = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=dates)
    res = simulate_portfolio(prices, w, dates, initial_cash=100.0, cost_bps=0.0)
    assert np.allclose(res["turnover"].iloc[1:], 0.0)


def test_higher_cost_lowers_terminal_nav():
    dates = pd.date_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"A": _rw(0, 10), "B": _rw(1, 10)}, index=dates)
    w = pd.DataFrame({"A": [0.5] * 10, "B": [0.5] * 10}, index=dates)
    low = simulate_portfolio(prices, w, dates, 100.0, cost_bps=0.0)["nav"].iloc[-1]
    high = simulate_portfolio(prices, w, dates, 100.0, cost_bps=50.0)["nav"].iloc[-1]
    assert high < low


def test_two_asset_hand_calculation():
    dates = pd.date_range("2020-01-01", periods=2)
    prices = pd.DataFrame({"A": [10.0, 10.0], "B": [20.0, 20.0]}, index=dates)
    w = pd.DataFrame({"A": [0.5] * 2, "B": [0.5] * 2}, index=dates)
    res = simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0, cost_bps=0.0)
    assert np.isclose(res["nav"].iloc[0], 100.0)
    assert np.isclose(res["cash"].iloc[0], 0.0)  # fully invested, no cash left


def test_missing_weights_raise():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0] * 3}, index=dates)
    w = pd.DataFrame({"A": [np.nan, 1.0, 1.0]}, index=dates)
    with pytest.raises(ValueError):
        simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0)


def test_bad_prices_raise():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, np.nan, 10.0]}, index=dates)
    w = pd.DataFrame({"A": [1.0] * 3}, index=dates)
    with pytest.raises(ValueError):
        simulate_portfolio(prices, w, [dates[0]], initial_cash=100.0)
