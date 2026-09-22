import numpy as np
import pandas as pd


def simulate_portfolio(
    prices,
    target_weights,
    rebalance_dates,
    initial_cash=1.0,
    cost_bps=0.0,
    cash_rate=0.0,
    tol=1e-12,
):
    """Self-financing portfolio simulator with explicit holdings, cash, turnover and costs.

    Each rebalance is treated as a self-financing transaction: the post-trade NAV is
    solved so that post-trade holdings plus post-trade cash equal pre-trade NAV minus
    the transaction cost incurred on the trades themselves. This avoids the small
    negative-cash / over-exposure artifact of charging costs after sizing.

    prices:          DataFrame (dates x assets) of finite, positive prices.
    target_weights:  DataFrame (dates x assets) of desired weights, read only on
                     rebalance dates (must already be lagged to avoid look-ahead).
    rebalance_dates: dates on which to trade toward the target weights.
    cost_bps:        transaction cost in basis points per unit of traded notional.
    cash_rate:       per-period return earned on uninvested cash (0 by default).
    Returns a DataFrame indexed by date: nav, cash, turnover, cost, net_return.
    The first net_return reflects the initial entry cost exactly once.
    """
    arr = prices.to_numpy()
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("prices must be finite and positive")

    assets = list(prices.columns)
    reb = set(pd.to_datetime(list(rebalance_dates)))
    shares = pd.Series(0.0, index=assets)
    cash = float(initial_cash)
    cost_rate = cost_bps / 1e4

    rows = []
    for t in prices.index:
        px = prices.loc[t]
        cash *= 1.0 + cash_rate
        nav_pre = cash + float((shares * px).sum())
        turnover = cost = 0.0
        nav = nav_pre
        if t in reb:
            w = target_weights.loc[t]
            if w.isna().any():
                raise ValueError(f"missing target weights on {t}")
            # self-financing fixed point: post_nav = nav_pre - cost_rate * traded_notional(post_nav)
            post_nav = nav_pre
            for _ in range(100):
                target_shares = (post_nav * w) / px
                traded = float(((target_shares - shares).abs() * px).sum())
                new_post = nav_pre - cost_rate * traded
                if abs(new_post - post_nav) < tol:
                    post_nav = new_post
                    break
                post_nav = new_post
            target_shares = (post_nav * w) / px
            traded = float(((target_shares - shares).abs() * px).sum())
            cost = cost_rate * traded
            turnover = traded / nav_pre if nav_pre else 0.0
            shares = target_shares
            cash = post_nav - float((shares * px).sum())
            nav = post_nav
        if nav < -tol:
            raise ValueError(f"negative NAV on {t}")
        rows.append((t, nav, float(cash), turnover, cost))

    out = pd.DataFrame(rows, columns=["date", "nav", "cash", "turnover", "cost"]).set_index("date")
    prev = out["nav"].shift(1)
    prev.iloc[0] = initial_cash  # entry cost appears once, in the first return
    out["net_return"] = (out["nav"] / prev - 1.0).fillna(0.0)
    return out
