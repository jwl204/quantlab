import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns: P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna()
