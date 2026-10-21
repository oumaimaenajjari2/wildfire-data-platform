"""Extract: find the raw FIRMS CSV files, fingerprint them, and read them.

Raw files are never modified. The manifest (name, size, SHA-256, row count)
makes every run traceable to the exact input files.
"""
import hashlib
from pathlib import Path

import pandas as pd


class ExtractError(Exception):
    """Raw input is missing."""


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_raw_files(raw_dir, pattern="*.csv") -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(pattern)) if raw_dir.is_dir() else []
    if not files:
        raise ExtractError(f"No files matching {pattern} in {raw_dir}")
    return files


def read_raw(path) -> pd.DataFrame:
    # acq_time must stay text: "0251" means 02:51, not the number 251
    return pd.read_csv(path, dtype={"acq_time": str})


def build_manifest(files, frames) -> list[dict]:
    return [
        {"file": f.name, "size_bytes": f.stat().st_size, "sha256": sha256_of(f), "rows": len(df)}
        for f, df in zip(files, frames)
    ]