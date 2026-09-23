import pytest

from microstructure.market_maker import (
    AvellanedaStoikovQuoter,
    clip_to_book,
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


def test_toxicity_out_of_range_raises():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.05, k=20.0)
    with pytest.raises(ValueError):
        simulate_market_making(quoter=q, n_steps=10, toxicity=1.5)


def test_clip_to_book_prevents_crossing():
    # a bid above the background ask is pulled a tick below it; an ask below the
    # background bid is pushed a tick above it; valid quotes are left untouched
    bid, ask = clip_to_book(bid=101.0, ask=99.0, bg_bid=99.9, bg_ask=100.1, tick=0.01)
    assert bid == pytest.approx(100.09) and ask == pytest.approx(99.91)
    assert bid < 100.1 and ask > 99.9
    bid2, ask2 = clip_to_book(99.95, 100.05, 99.9, 100.1, 0.01)
    assert bid2 == pytest.approx(99.95) and ask2 == pytest.approx(100.05)


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


def test_toxic_flow_reduces_profit_via_inventory():
    q = AvellanedaStoikovQuoter(gamma=0.1, sigma=0.05, k=20.0)

    def means(toxicity):
        runs = [
            simulate_market_making(
                quoter=q,
                n_steps=1500,
                background_half_spread=0.15,
                arrival_rate=1.0,
                toxicity=toxicity,
                seed=s,
            )
            for s in range(20)
        ]
        return (
            sum(r.total_pnl for r in runs) / len(runs),
            sum(r.inventory_pnl for r in runs) / len(runs),
        )

    total_control, inv_control = means(0.0)
    total_toxic, inv_toxic = means(0.8)
    # informed flow is genuine adverse selection: lower total profit, driven by a
    # much more negative inventory-carry term than the random-flow control
    assert total_toxic < total_control
    assert inv_toxic < inv_control
