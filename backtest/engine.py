import pandas as pd


def backtest(returns, signal, cost_bps=0.0):
    """Backtest a trading signal on an asset's returns.

    returns: daily returns of the asset (Series).
    signal:  desired position each day (+1 long, 0 flat, -1 short) (Series).
    cost_bps: transaction cost in basis points per unit of turnover.
    Returns a DataFrame with the position held, strategy returns, and equity curve.
    """
    position = signal.shift(1).fillna(0.0)  # trade on yesterday's signal
    turnover = position.diff().abs().fillna(0.0)  # how much the position changes
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = position * returns - cost
    equity = (1.0 + strat_ret).cumprod()
    return pd.DataFrame({"position": position, "strategy_return": strat_ret, "equity": equity})
