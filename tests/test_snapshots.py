import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from core import snapshots


def _write(tmp_path, name, df, checksum):
    df.to_csv(tmp_path / f"{name}.csv")
    (tmp_path / f"{name}_manifest.json").write_text(json.dumps({"sha256": checksum}))


def test_load_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", tmp_path)
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.date_range("2020-01-01", periods=3))
    df.to_csv(tmp_path / "x.csv")
    checksum = hashlib.sha256((tmp_path / "x.csv").read_bytes()).hexdigest()
    (tmp_path / "x_manifest.json").write_text(json.dumps({"sha256": checksum}))
    loaded = snapshots.load_snapshot("x")
    assert np.allclose(loaded["A"].to_numpy(), [1.0, 2.0, 3.0])


def test_checksum_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", tmp_path)
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    _write(tmp_path, "x", df, "deadbeef")
    with pytest.raises(ValueError):
        snapshots.load_snapshot("x")


def test_missing_snapshot_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        snapshots.load_snapshot("missing")
