"""Versioned, checksummed price snapshots for offline-reproducible research.

Live data providers can rate-limit requests or silently revise adjusted histories,
so headline results should be regenerated from an immutable dated snapshot rather
than a live response. Build a snapshot once (with network), then load it offline.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from core.data import load_prices

SNAPSHOT_DIR = Path("snapshots")


def build_snapshot(tickers, start, end, name="basket", provider="yfinance"):
    """Download an aligned price matrix and store it as an immutable, checksummed snapshot."""
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    frames = {}
    for t in tickers:
        s = load_prices(t, start, end)
        if s.empty:
            raise ValueError(f"no data returned for {t}")
        frames[t] = s
    prices = pd.DataFrame(frames).dropna()
    csv_path = SNAPSHOT_DIR / f"{name}.csv"
    prices.to_csv(csv_path)
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "name": name,
        "provider": provider,
        "adjustment": "auto_adjust=True (dividends and splits)",
        "tickers": list(tickers),
        "start": start,
        "end": end,
        "rows": int(prices.shape[0]),
        "cols": int(prices.shape[1]),
        "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
        "sha256": checksum,
    }
    (SNAPSHOT_DIR / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return prices


def load_snapshot(name="basket", verify=True):
    """Load a stored snapshot offline, verifying its SHA-256 checksum against the manifest."""
    csv_path = SNAPSHOT_DIR / f"{name}.csv"
    man_path = SNAPSHOT_DIR / f"{name}_manifest.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"snapshot '{name}' not found; run build_snapshot first")
    if verify and man_path.exists():
        expected = json.loads(man_path.read_text())["sha256"]
        actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"snapshot '{name}' checksum mismatch (file changed since build)")
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)
