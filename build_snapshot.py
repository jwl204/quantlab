from core.snapshots import build_snapshot

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "KO",
    "WMT",
    "NVDA",
    "META",
    "V",
    "HD",
    "DIS",
    "INTC",
    "CSCO",
    "PFE",
    "BA",
    "MCD",
]

prices = build_snapshot(TICKERS, "2015-01-01", "2024-01-01", name="basket")
print(f"Built snapshot: snapshots/basket.csv ({prices.shape[0]} rows x {prices.shape[1]} tickers)")
