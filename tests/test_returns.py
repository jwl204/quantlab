import numpy as np
import pandas as pd

from core.returns import log_returns


def test_log_returns_length():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert len(log_returns(prices)) == 2


def test_log_returns_value():
    prices = pd.Series([100.0, 110.0])
    result = log_returns(prices).iloc[0]
    assert np.isclose(result, np.log(1.1))
