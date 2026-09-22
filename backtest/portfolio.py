import pandas as pd


def simulate_portfolio(
    prices, target_weights, rebalance_dates, initial_cash=1.0, cost_bps=0.0
):
    """Portfolio simulator with explicit holdings, cash, turnover and costs.

    prices:          DataFrame (dates x assets) of prices.
    target_weights:  DataFrame (dates x assets) of desired weights, read only on
                     rebalance dates (must already be lagged to avoid look-ahead).
    rebalance_dates: the dates on which to trade toward the target weights.
    Returns a DataFrame indexed by date with: nav, cash, turnover, cost, net_return.
    """
    assets = list(prices.columns)
    reb = set(pd.to_datetime(list(rebalance_dates)))
    shares = pd.Series(0.0, index=assets)
    cash = float(initial_cash)

    rows = []
    for t in prices.index:
        px = prices.loc[t]
        nav = cash + float((shares * px).sum())  # mark to market
        turnover = cost = 0.0
        if t in reb:
            target_shares = (nav * target_weights.loc[t]) / px
            traded_notional = float(((target_shares - shares).abs() * px).sum())
            cost = traded_notional * (cost_bps / 1e4)
            turnover = traded_notional / nav
            shares = target_shares
            cash = nav - float((shares * px).sum()) - cost
            nav = cash + float((shares * px).sum())  # after paying cost
        rows.append((t, nav, cash, turnover, cost))

    out = pd.DataFrame(
        rows, columns=["date", "nav", "cash", "turnover", "cost"]
    ).set_index("date")
    out["net_return"] = out["nav"].pct_change().fillna(0.0)
    return out
