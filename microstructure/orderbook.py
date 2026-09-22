"""A minimal event-driven limit order book with price-time priority.

The book keeps two sides, each a mapping from price level to a FIFO queue of
resting orders. A limit order rests as passive liquidity; a market order walks
the opposite side, consuming the best-priced, earliest-arriving orders first
(price-time priority, the rule used by essentially every modern exchange).

The matching engine is deliberately small so its behaviour can be reasoned
about and unit-tested exactly; it is fast enough for the tick-level Monte Carlo
simulations in ``market_maker.py`` but is not an industrial low-latency engine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

_EPS = 1e-12


@dataclass
class Order:
    """A single resting order. ``side`` is 'buy' or 'sell'."""

    id: int
    side: str
    price: float
    qty: float


@dataclass(frozen=True)
class Fill:
    """One trade. ``aggressor_side`` is the side of the incoming market order."""

    price: float
    qty: float
    aggressor_side: str
    maker_id: int
    taker_id: int


class LimitOrderBook:
    """Price-time-priority limit order book.

    Bids and asks are ``dict`` mappings from price to a ``deque`` of orders in
    arrival order. The best bid is the highest bid price; the best ask is the
    lowest ask price. Order ids must be unique while resting.
    """

    def __init__(self) -> None:
        self.bids: dict[float, deque[Order]] = {}
        self.asks: dict[float, deque[Order]] = {}
        self._loc: dict[int, tuple[str, float]] = {}

    # --- top of book -----------------------------------------------------
    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else 0.5 * (b + a)

    @property
    def spread(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    # --- mutation --------------------------------------------------------
    def add_limit(self, order: Order) -> None:
        """Rest a passive order. The caller is responsible for not crossing."""
        if order.qty <= _EPS:
            raise ValueError("order quantity must be positive")
        if order.id in self._loc:
            raise ValueError(f"order id {order.id} already resting")
        book = self.bids if order.side == "buy" else self.asks
        book.setdefault(order.price, deque()).append(order)
        self._loc[order.id] = (order.side, order.price)

    def cancel(self, order_id: int) -> bool:
        """Remove a resting order by id. Returns True if it was present."""
        loc = self._loc.pop(order_id, None)
        if loc is None:
            return False
        side, price = loc
        book = self.bids if side == "buy" else self.asks
        queue = book.get(price)
        if queue is not None:
            for o in queue:
                if o.id == order_id:
                    queue.remove(o)
                    break
            if not queue:
                del book[price]
        return True

    def market_order(self, side: str, qty: float, taker_id: int = -1) -> list[Fill]:
        """Execute a market order that consumes liquidity on the opposite side.

        ``side`` is the aggressor's side: a 'buy' lifts asks, a 'sell' hits bids.
        Returns the list of fills, walking levels by price then arrival time.
        Any quantity beyond the resting depth is left unfilled (no fill emitted).
        """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        book = self.asks if side == "buy" else self.bids
        remaining = qty
        fills: list[Fill] = []
        while remaining > _EPS and book:
            best = min(book) if side == "buy" else max(book)
            queue = book[best]
            while remaining > _EPS and queue:
                maker = queue[0]
                traded = min(remaining, maker.qty)
                fills.append(Fill(best, traded, side, maker.id, taker_id))
                maker.qty -= traded
                remaining -= traded
                if maker.qty <= _EPS:
                    queue.popleft()
                    self._loc.pop(maker.id, None)
            if not queue:
                del book[best]
        return fills
