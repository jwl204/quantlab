"""Market-making experiment: P&L attribution and the cost of latency.

Runs the Avellaneda-Stoikov quoter through the limit order book over many seeds
and reports (1) how total P&L splits into spread capture versus inventory carry,
and (2) how profit decays as quoting latency rises. Fully offline and
reproducible -- it is a Monte Carlo simulation with fixed seeds, no market data.

Run:  python run_microstructure.py
"""

from __future__ import annotations

import numpy as np

from microstructure.market_maker import AvellanedaStoikovQuoter, simulate_market_making

GAMMA, SIGMA, K = 0.1, 0.05, 20.0
N_STEPS, BG_HALF_SPREAD, ARRIVAL = 1500, 0.15, 1.0
SEEDS = range(50)


def average_over_seeds(latency: int) -> dict[str, float]:
    quoter = AvellanedaStoikovQuoter(GAMMA, SIGMA, K)
    runs = [
        simulate_market_making(
            quoter=quoter,
            n_steps=N_STEPS,
            background_half_spread=BG_HALF_SPREAD,
            arrival_rate=ARRIVAL,
            latency=latency,
            seed=s,
        )
        for s in SEEDS
    ]
    return {
        "total": float(np.mean([r.total_pnl for r in runs])),
        "spread": float(np.mean([r.spread_pnl for r in runs])),
        "inventory": float(np.mean([r.inventory_pnl for r in runs])),
        "fills": float(np.mean([r.n_fills for r in runs])),
        "inv_std": float(np.mean([r.inventory_std for r in runs])),
        "recon": float(np.max([r.reconciliation_error for r in runs])),
    }


def main() -> None:
    base = average_over_seeds(latency=0)
    print(f"Averages over {len(SEEDS)} seeds, {N_STEPS} ticks each\n")
    print("P&L attribution (no latency):")
    print(f"  spread capture   {base['spread']:8.3f}")
    print(f"  inventory carry  {base['inventory']:8.3f}")
    print(f"  total P&L        {base['total']:8.3f}")
    print(f"  (max reconciliation error {base['recon']:.1e})\n")

    print("Cost of latency:")
    print(f"  {'latency':>7}  {'total':>8}  {'spread':>8}  {'inventory':>9}  {'fills':>6}")
    for lat in (0, 1, 2, 5, 10, 20):
        m = average_over_seeds(latency=lat)
        print(
            f"  {lat:>7}  {m['total']:>8.3f}  {m['spread']:>8.3f}  "
            f"{m['inventory']:>9.3f}  {m['fills']:>6.0f}"
        )


if __name__ == "__main__":
    main()
