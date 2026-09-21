def ma_trend_signal(prices, window=200):
    """Long (1) when price is above its moving average, flat (0) otherwise."""
    ma = prices.rolling(window).mean()
    return (prices > ma).astype(float)