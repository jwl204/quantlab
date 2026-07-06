import pandas as pd
import yfinance as yf

def load_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted closing prices for one ticker as a clean Series."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()