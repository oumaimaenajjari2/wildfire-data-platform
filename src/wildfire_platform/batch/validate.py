"""Validation of raw NASA FIRMS detections.

Two kinds of problems:
- critical (empty file, missing columns): raise ValidationError, the pipeline must stop
- row-level (bad coordinates, bad date/time, duplicates): the row goes to `rejected`
  with a reason, and the rest continues.
"""
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time"]
DEDUP_KEYS = ["latitude", "longitude", "acq_date", "acq_time"]


class ValidationError(Exception):
    """A critical assumption is broken; the pipeline must fail."""


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame
    report: dict


def validate(df: pd.DataFrame) -> ValidationResult:
    # --- critical checks ---
    if df is None or len(df) == 0:
        raise ValidationError("Input is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    df = df.copy()
    df["reject_reason"] = None

    def mark(mask, reason):
        # only label rows that don't already have a reason (first problem wins)
        df.loc[mask & df["reject_reason"].isna(), "reject_reason"] = reason

    # --- row-level checks ---
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    bad_coords = ~(lat.between(-90, 90) & lon.between(-180, 180))  # NaN -> False -> bad
    mark(bad_coords, "invalid_coordinates")

    date = pd.to_datetime(df["acq_date"], errors="coerce")
    mark(date.isna(), "invalid_date")

    t = df["acq_time"].astype(str).str.zfill(4)  # "251" -> "0251"
    hh = pd.to_numeric(t.str[:2], errors="coerce")
    mm = pd.to_numeric(t.str[2:], errors="coerce")
    good_time = t.str.fullmatch(r"\d{4}") & (hh < 24) & (mm < 60)
    mark(~good_time, "invalid_time")

    mark(df.duplicated(subset=DEDUP_KEYS, keep="first"), "duplicate")

    # --- split and report ---
    is_bad = df["reject_reason"].notna()
    clean = df.loc[~is_bad].drop(columns="reject_reason").reset_index(drop=True)
    rejected = df.loc[is_bad].reset_index(drop=True)

    report = {
        "rows_in": len(df),
        "rows_clean": len(clean),
        "rows_rejected": len(rejected),
        "rejected_by_reason": rejected["reject_reason"].value_counts().to_dict(),
    }

    if clean.empty:
        raise ValidationError(f"All rows rejected: {report['rejected_by_reason']}")

    return ValidationResult(clean=clean, rejected=rejected, report=report)