from core.snapshots import SNAPSHOT_DIR, build_snapshot

# Hand-picked large caps (survivors) — the exploratory universe.
BASKET = [
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

# 9 SPDR sector ETFs — a rules-based universe with no single-name survivorship bias.
ETF = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

for name, tickers in [("basket", BASKET), ("etf", ETF)]:
    if (SNAPSHOT_DIR / f"{name}.csv").exists():
        print(f"snapshot '{name}' already exists, skipping (delete the file to rebuild)")
        continue
    prices = build_snapshot(tickers, "2015-01-01", "2024-01-01", name=name)
    print(
        f"Built snapshot: snapshots/{name}.csv ({prices.shape[0]} rows x {prices.shape[1]} tickers)"
    )
