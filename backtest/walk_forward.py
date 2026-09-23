import numpy as np
import pandas as pd

from backtest.portfolio import simulate_portfolio
from backtest.strategies import ma_trend_signal


def sharpe(r, periods=252):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(periods)


def trend_weights(prices, window):
    """Equal weight among stocks in an uptrend, lagged one day (no look-ahead)."""
    raw = pd.DataFrame({t: ma_trend_signal(prices[t], window) for t in prices.columns})
    raw = raw.shift(1).fillna(0.0)
    return raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def run_walk_forward(prices, candidate_windows, n_folds=5, cost_bps=5.0, min_train=252):
    """Stateful expanding-window walk-forward validation.

    In each sequential test fold the moving-average window is re-selected using only
    data prior to that fold. A single continuous portfolio is carried across folds and
    rebalances only on month-ends, so a newly selected window becomes active at the
    next scheduled monthly rebalance, where the transition trade and its cost are
    charged (not at the fold boundary itself). Returns (portfolio_result, folds_info).
    """
    dates = prices.index
    month_ends = pd.Series(index=dates, data=dates).resample("ME").last().dropna().values
    cand_w = {w: trend_weights(prices, w) for w in candidate_windows}
    cand_ret = {
        w: simulate_portfolio(prices, cand_w[w], month_ends, cost_bps=cost_bps)["net_return"]
        for w in candidate_windows
    }

    fold_edges = pd.date_range(dates[0], dates[-1], periods=n_folds + 1)
    combined = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    folds_info = []
    for i in range(1, n_folds + 1):
        test_start, test_end = fold_edges[i - 1], fold_edges[i]
        train_mask = dates < test_start
        test_mask = (dates >= test_start) & (dates < test_end)
        if train_mask.sum() < min_train:
            continue
        scores = {w: sharpe(cand_ret[w][train_mask]) for w in candidate_windows}
        best = max(scores, key=scores.get)
        combined.loc[test_mask] = cand_w[best].loc[test_mask]
        folds_info.append({"test_start": test_start, "test_end": test_end, "selected_window": best})

    if not folds_info:
        raise ValueError("no fold had enough training history; reduce min_train or n_folds")
    first_test = folds_info[0]["test_start"]
    reb_dates = [d for d in month_ends if d >= np.datetime64(first_test)]
    portfolio = simulate_portfolio(prices, combined, reb_dates, cost_bps=cost_bps)
    return portfolio, folds_info
