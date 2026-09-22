import numpy as np


def performance(result, periods_per_year=252):
    """Compute headline performance metrics from a backtest result."""
    r = result["strategy_return"]
    sharpe = r.mean() / r.std() * np.sqrt(periods_per_year)

    equity = result["equity"]
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()

    turnover = result["position"].diff().abs().mean() * periods_per_year
    total_return = equity.iloc[-1] - 1.0

    return dict(
        sharpe=sharpe,
        max_drawdown=max_dd,
        annual_turnover=turnover,
        total_return=total_return,
    )
