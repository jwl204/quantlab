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

To keep the book well-formed, the maker's quotes are **clipped to a tick inside
the opposite background quote** (`clip_to_book`): a large inventory skew can push
the reservation price far enough that a raw quote would cross, and the passive
`add_limit` does not match crossing orders, so clipping prevents a bid ever
resting at or above the best ask (and vice versa).

Order flow is a controllable mix (`toxicity` in [0, 1]). A fraction `toxicity` of
orders are **informed**: their direction matches the sign of the *next* mid move,
so they systematically trade just ahead of the price. The rest are uninformed
(equally likely to buy or sell). `toxicity = 0` is the random-flow control. This
matters because latency and informed flow are *different* loss channels, and the
P&L attribution below separates them.

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
background half-spread = 0.15), random flow with no latency:

| Component | Mean P&L |
|---|---|
| Spread capture | +31.3 |
| Inventory carry | −0.2 |
| **Total** | **+31.1** |

Almost all the profit is spread capture; the inventory-carry term is close to
zero because the reservation-price skew keeps the position from drifting far.
The maximum reconciliation error across all runs is ~1e-13.

## Two loss channels, cleanly separated

The value of an *exact* attribution is that different frictions hit different
terms. The simulation isolates two.

**Latency — a stale-quote loss (hits spread capture).** With latency the maker
quotes off a mid observed `latency` ticks in the past. Even against purely random
flow this loses money, because the quote lags the walk and fills happen at worse
prices relative to the current mid. Averaging over 50 seeds (random flow):

| Latency (ticks) | Total P&L | Spread capture | Inventory carry | Fills |
|---|---|---|---|---|
| 0 | 31.1 | 31.3 | −0.2 | 499 |
| 1 | 23.6 | 23.7 | −0.1 | 596 |
| 2 | 20.1 | 20.1 | 0.0 | 615 |
| 5 | 15.7 | 15.4 | +0.3 | 627 |
| 10 | 12.0 | 11.7 | +0.4 | 617 |
| 20 | 9.8 | 9.5 | +0.3 | 612 |

Note that **fills go up while profit falls**: a slow maker gets *more* trades
(its stale quotes are attractive precisely when mispriced) but each is worse, and
the loss lands almost entirely in **spread capture** while inventory carry stays
near zero. This is a stale-quote / latency cost, not adverse selection in the
informed-trader sense — the flow here is random, so it is worth naming precisely.

**Toxic flow — genuine adverse selection (hits inventory carry).** Setting
`toxicity > 0` makes a fraction of orders informed, trading in the direction of
the next mid move. Now the maker is filled precisely on the wrong side just before
the price moves, so it accumulates adverse inventory. With no latency at all,
averaging over 50 seeds:

| Toxicity | Total P&L | Spread capture | Inventory carry | Fills |
|---|---|---|---|---|
| 0.00 | 31.1 | 31.3 | −0.2 | 499 |
| 0.25 | 26.1 | 30.2 | −4.0 | 489 |
| 0.50 | 19.3 | 27.4 | −8.1 | 462 |
| 0.75 | 13.0 | 23.8 | −10.7 | 425 |
| 0.90 | 7.7 | 21.1 | −13.4 | 396 |

Here spread capture stays high but the **inventory-carry term collapses** — the
signature of adverse selection, and mechanically distinct from the latency loss
above. The exact attribution is what lets the two be told apart: latency drains
spread capture, informed flow drains inventory carry.

## Reproduce

```
python run_microstructure.py
```

Fully offline and seeded — no market data, deterministic output. Both the order
book and the market maker are covered by `tests/test_orderbook.py` and
`tests/test_market_maker.py`.

## Limitations

The mid is an exogenous random walk, so there is no feedback from the maker's own
trades into the price (no permanent market impact). The informed-flow model uses a
one-step look-ahead (direction matches the very next mid move) as a stylised proxy
for toxicity rather than a fully specified informed-trader model. Background
liquidity is a single level rather than a full depth profile, and the fill
mechanism is competition against that level rather than a calibrated intensity
curve. These are deliberate simplifications: the goal is a transparent, testable
illustration of inventory management, spread capture, latency cost and adverse
selection, not a production market-making engine.
