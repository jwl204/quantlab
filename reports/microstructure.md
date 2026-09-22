# Market Microstructure: Inventory-Aware Market Making

## Motivation

The rest of QuantLab treats price as a single number that arrives once per day.
Real trading happens one order at a time against a *limit order book*, and the
price you actually pay depends on who is resting liquidity and who is crossing
the spread. This module drops down to that level: it builds an order book, runs
an inventory-aware market-making strategy on it, and measures where the money
comes from — and how quickly it disappears when the maker is slow.

## The limit order book

`microstructure/orderbook.py` is a small matching engine with **price-time
priority**, the rule used by essentially every modern exchange:

- A **limit order** rests as passive liquidity at a chosen price.
- A **market order** walks the opposite side, consuming the best-priced,
  earliest-arriving orders first.

The engine tracks the best bid, best ask, mid and spread, and supports cancels.
It is deliberately small enough to unit-test exactly (`tests/test_orderbook.py`
checks price priority, FIFO time priority within a level, partial fills, cancels
and depletion).

## Inventory-aware quoting (Avellaneda-Stoikov)

`microstructure/market_maker.py` implements the Avellaneda & Stoikov (2008)
quoter. Rather than quoting symmetrically around the mid, it forms a
**reservation price** shifted against its current inventory:

```
reservation = mid - inventory * gamma * sigma^2 * tau
half_spread = 0.5 * gamma * sigma^2 * tau + (1/gamma) * ln(1 + gamma/k)
bid, ask    = reservation - half_spread, reservation + half_spread
```

- **gamma** — risk aversion. Higher gamma quotes wider and skews harder.
- **sigma** — mid volatility; more volatility means more inventory risk, so wider quotes.
- **tau** — time left to the horizon; the maker quotes wider early and tightens as it runs out of time to offload inventory.
- **k** — order-book liquidity; a deeper book (large k) lets the maker quote tighter for the same fill rate.

The intuition: when the maker is long, it lowers *both* quotes so it is keener to
sell and more reluctant to buy, steering inventory back toward flat. When flat,
it quotes symmetrically and simply earns the spread.

## The simulation

`simulate_market_making` runs the quoter tick by tick. The true mid follows an
arithmetic random walk. Each tick the maker posts a bid and ask alongside
**background liquidity** from noise traders a fixed distance either side of the
mid, then a Poisson number of market orders arrive and walk the book through the
matching engine. The maker is filled only when its quote *betters* the
background — so tighter quotes win more flow, at the cost of thinner edge.

## P&L attribution

Total profit is decomposed exactly into two economically distinct pieces:

```
total P&L = spread capture + inventory carry
spread capture = sum over fills of (mid_at_fill - fill_price) * signed_qty
inventory carry = sum over ticks of inventory_held * (mid_change)
```

The first term is the edge earned by buying below and selling above the mid; the
second is the mark-to-market gain or loss on the position held while the mid
drifts. These reconcile to total P&L *by construction* (a summation-by-parts
identity), and the test suite asserts the residual is numerically zero.

Averaged over 50 seeds of 1,500 ticks (gamma = 0.1, sigma = 0.05, k = 20,
background half-spread = 0.15), with no latency:

| Component | Mean P&L |
|---|---|
| Spread capture | +30.9 |
| Inventory carry | −0.3 |
| **Total** | **+30.6** |

Almost all the profit is spread capture; the inventory-carry term is close to
zero because the reservation-price skew keeps the position from drifting far.
The maximum reconciliation error across all runs is ~1e-13.

## The cost of latency (adverse selection)

Latency is modelled by having the maker quote off a mid observed `latency` ticks
in the past. Stale quotes sit at prices that are favourable to *takers* just
before the mid moves, so informed flow picks them off — classic adverse
selection. Averaging over 50 seeds:

| Latency (ticks) | Total P&L | Spread capture | Inventory carry | Fills |
|---|---|---|---|---|
| 0 | 30.6 | 30.9 | −0.3 | 494 |
| 1 | 23.2 | 23.5 | −0.2 | 597 |
| 2 | 19.4 | 19.6 | −0.2 | 613 |
| 5 | 13.7 | 13.6 | +0.1 | 623 |
| 10 | 8.8 | 8.7 | +0.1 | 618 |
| 20 | 4.6 | 4.8 | −0.2 | 613 |

The striking part is that **fills go up while profit collapses**. A slow maker
gets *more* trades — its stale quotes are attractive precisely when they are
mispriced — but each fill is worse, so total P&L falls by ~85% from zero to
twenty ticks of delay. This is the quantitative signature of adverse selection,
and it is why real market makers spend heavily on speed.

## Reproduce

```
python run_microstructure.py
```

Fully offline and seeded — no market data, deterministic output. Both the order
book and the market maker are covered by `tests/test_orderbook.py` and
`tests/test_market_maker.py`.

## Limitations

The mid is an exogenous random walk, so there is no feedback from the maker's own
trades into the price (no permanent market impact). Background liquidity is a
single level rather than a full depth profile, and the fill mechanism is
competition against that level rather than a calibrated intensity curve. These
are deliberate simplifications: the goal is a transparent, testable illustration
of inventory management, spread capture and adverse selection, not a production
market-making engine.
