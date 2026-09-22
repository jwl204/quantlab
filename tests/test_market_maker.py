import pytest

from microstructure.market_maker import (
    AvellanedaStoikovQuoter,
    simulate_market_making,
)


def test_reservation_price_skews_against_inventory():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.5, k=1.5)
    long = q.reservation_price(mid=100.0, inventory=5.0, tau=1.0)
    flat = q.reservation_price(mid=100.0, inventory=0.0, tau=1.0)
    short = q.reservation_price(mid=100.0, inventory=-5.0, tau=1.0)
    assert long < flat < short  # long inventory pushes quotes down to encourage selling


def test_half_spread_grows_with_time_remaining():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.5, k=1.5)
    assert q.half_spread(2.0) > q.half_spread(0.5) > 0


def test_quotes_are_ordered():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.5, k=1.5)
    bid, ask = q.quotes(mid=100.0, inventory=0.0, tau=1.0)
    assert bid < 100.0 < ask


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        AvellanedaStoikovQuoter(gamma=-1.0, sigma=0.5, k=1.5)


def test_pnl_attribution_reconciles():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.05, k=20.0)
    res = simulate_market_making(
        quoter=q,
        n_steps=1500,
        background_half_spread=0.15,
        arrival_rate=1.0,
        seed=3,
    )
    assert res.n_fills > 0
    # spread capture + inventory carry must equal total P&L by construction
    assert res.reconciliation_error < 1e-8


def test_latency_reduces_profit_on_average():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.05, k=20.0)

    def mean_total_pnl(latency):
        vals = [
            simulate_market_making(
                quoter=q,
                n_steps=1500,
                background_half_spread=0.15,
                arrival_rate=1.0,
                latency=latency,
                seed=s,
            ).total_pnl
            for s in range(20)
        ]
        return sum(vals) / len(vals)

    # stale quotes get adversely selected: market-making profit falls with latency
    assert mean_total_pnl(latency=10) < mean_total_pnl(latency=0)
