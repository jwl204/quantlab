"""Inventory-aware market making (Avellaneda-Stoikov) on the limit order book.

The quoter follows Avellaneda & Stoikov (2008). Around the mid it forms a
*reservation price* that is shifted against its inventory: a long inventory
lowers both quotes so the maker is keener to sell, a short inventory raises
them. The total quoted spread widens with risk aversion, volatility and time
remaining, and narrows when the book is deep (large ``k``). The maker earns the
half-spread on round trips but carries inventory risk between them; with
non-zero ``latency`` its quotes are stale, so informed flow picks them off just
before the mid moves -- classic adverse selection.

``simulate_market_making`` runs the quoter as the only strategic liquidity in a
book that also holds background liquidity from noise traders, with Poisson
market-order arrivals routed through the price-time-priority matching engine in
``orderbook.py``. It returns an exact profit-and-loss attribution splitting the
result into spread capture (edge earned versus the mid at each fill) and
inventory carry (mark-to-market of the held position as the mid drifts). The two
terms reconcile to total P&L by construction; ``tests/test_market_maker.py``
asserts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .orderbook import LimitOrderBook, Order


@dataclass
class AvellanedaStoikovQuoter:
    """Inventory-aware quoter.

    gamma: risk aversion (>0). sigma: mid volatility in price units per sqrt-time.
    k: order-book liquidity / intensity-decay parameter (>0); larger k => the
    market maker can quote tighter for the same fill rate.
    """

    gamma: float
    sigma: float
    k: float

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.sigma < 0 or self.k <= 0:
            raise ValueError("require gamma>0, sigma>=0, k>0")

    def reservation_price(self, mid: float, inventory: float, tau: float) -> float:
        """Mid shifted against inventory over the remaining horizon ``tau``."""
        return mid - inventory * self.gamma * self.sigma**2 * tau

    def half_spread(self, tau: float) -> float:
        """Half of the optimal total spread gamma*sigma^2*tau + (2/gamma)ln(1+gamma/k)."""
        return 0.5 * self.gamma * self.sigma**2 * tau + (1.0 / self.gamma) * math.log(
            1.0 + self.gamma / self.k
        )

    def quotes(self, mid: float, inventory: float, tau: float) -> tuple[float, float]:
        """Return (bid, ask) around the inventory-adjusted reservation price."""
        r = self.reservation_price(mid, inventory, tau)
        h = self.half_spread(tau)
        return r - h, r + h


@dataclass
class MarketMakingResult:
    total_pnl: float
    spread_pnl: float
    inventory_pnl: float
    n_fills: int
    final_inventory: float
    inventory_std: float
    max_abs_inventory: float
    mids: np.ndarray
    inventory_path: np.ndarray

    @property
    def reconciliation_error(self) -> float:
        """|total - (spread + inventory)|; exact attribution => ~0."""
        return abs(self.total_pnl - (self.spread_pnl + self.inventory_pnl))


_BID_ID = 1
_ASK_ID = 2


def simulate_market_making(
    *,
    quoter: AvellanedaStoikovQuoter,
    n_steps: int = 2000,
    dt: float = 1.0,
    s0: float = 100.0,
    order_size: float = 1.0,
    background_half_spread: float = 0.10,
    arrival_rate: float = 1.0,
    latency: int = 0,
    horizon: float | None = None,
    seed: int = 0,
) -> MarketMakingResult:
    """Tick-level market-making simulation on the limit order book.

    The true mid follows an arithmetic random walk with per-step standard
    deviation ``quoter.sigma * sqrt(dt)``. Each tick the maker observes the mid
    delayed by ``latency`` ticks, quotes a bid and ask, and posts them alongside
    background liquidity a fixed ``background_half_spread`` either side of the
    *true* mid. A Poisson(``arrival_rate``) number of market orders then arrive,
    each an equally likely buy or sell of ``order_size``, and walk the book; the
    maker is filled only when its quote betters the background, so tighter quotes
    win flow at the cost of thinner edge and more adverse selection under latency.
    """
    if latency < 0:
        raise ValueError("latency must be non-negative")
    rng = np.random.default_rng(seed)
    total_time = horizon if horizon is not None else n_steps * dt
    step_sd = quoter.sigma * math.sqrt(dt)

    mids = np.empty(n_steps + 1)
    mids[0] = s0
    mids[1:] = s0 + np.cumsum(step_sd * rng.standard_normal(n_steps))

    inventory = 0.0
    cash = 0.0
    inv_path = np.empty(n_steps)
    fills: list[tuple[int, float, float]] = []  # (tick, price, signed_qty)
    bg_id = 1000
    taker_id = -1

    for t in range(n_steps):
        tau = max(total_time - t * dt, dt)
        obs_mid = mids[max(0, t - latency)]
        bid_px, ask_px = quoter.quotes(obs_mid, inventory, tau)

        book = LimitOrderBook()
        true_mid = mids[t]
        book.add_limit(Order(bg_id, "buy", round(true_mid - background_half_spread, 4), 100.0))
        book.add_limit(Order(bg_id + 1, "sell", round(true_mid + background_half_spread, 4), 100.0))
        bg_id += 2
        if bid_px < ask_px:
            book.add_limit(Order(_BID_ID, "buy", round(bid_px, 4), order_size))
            book.add_limit(Order(_ASK_ID, "sell", round(ask_px, 4), order_size))

        for _ in range(int(rng.poisson(arrival_rate))):
            side = "buy" if rng.random() < 0.5 else "sell"
            for f in book.market_order(side, order_size, taker_id=taker_id):
                if f.maker_id == _BID_ID:  # a market sell hit the maker's bid: maker buys
                    inventory += f.qty
                    cash -= f.price * f.qty
                    fills.append((t, f.price, f.qty))
                elif f.maker_id == _ASK_ID:  # a market buy lifted the maker's ask: maker sells
                    inventory -= f.qty
                    cash += f.price * f.qty
                    fills.append((t, f.price, -f.qty))
            taker_id -= 1

        inv_path[t] = inventory

    final_mid = float(mids[n_steps])
    total_pnl = cash + inventory * final_mid
    spread_pnl = sum((mids[t] - price) * sq for t, price, sq in fills)
    inventory_pnl = float(np.sum(inv_path * np.diff(mids)))

    return MarketMakingResult(
        total_pnl=float(total_pnl),
        spread_pnl=float(spread_pnl),
        inventory_pnl=inventory_pnl,
        n_fills=len(fills),
        final_inventory=float(inventory),
        inventory_std=float(inv_path.std()),
        max_abs_inventory=float(np.abs(inv_path).max()) if n_steps else 0.0,
        mids=mids,
        inventory_path=inv_path,
    )
