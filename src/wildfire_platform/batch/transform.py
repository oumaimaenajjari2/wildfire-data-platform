"""Transform validated FIRMS detections into an analysis-ready table.

Design choices (also listed in docs/decisions.md):
- Grid cells are coarse (default 0.5 degrees, about 55 km): a cell is an
  aggregation unit, not a precise location.
- All rows are kept. The `type` field is turned into a flag so queries can
  filter, instead of silently deleting data.
- Optional source columns may be missing in some years; they are filled with
  nulls so the output schema is always the same.
"""
import hashlib

import numpy as np
import pandas as pd

OPTIONAL_COLUMNS = [
    "satellite", "instrument", "confidence", "frp", "daynight",
    "type", "bright_ti4", "bright_ti5", "version",
]

OUTPUT_COLUMNS = [
    "detection_id", "acq_datetime", "date", "year", "month", "hour",
    "latitude", "longitude", "cell_id",
    "satellite", "instrument", "confidence", "confidence_label",
    "frp", "bright_ti4", "bright_ti5", "daynight", "version",
    "type", "is_presumed_vegetation_fire",
]

CONFIDENCE_LABELS = {"l": "low", "n": "nominal", "h": "high"}


def _grid_cell(values: pd.Series, cell_size: float) -> pd.Series:
    """Lower edge of the cell containing each value (floor works for negatives too)."""
    return np.floor(values / cell_size) * cell_size


def transform(df: pd.DataFrame, cell_size: float = 0.5) -> pd.DataFrame:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # timestamp (UTC): acq_time is text like "0251" = 02:51
    t = out["acq_time"].astype(str).str.zfill(4)
    out["acq_datetime"] = pd.to_datetime(
        out["acq_date"].astype(str).str[:10] + " " + t.str[:2] + ":" + t.str[2:],
        format="%Y-%m-%d %H:%M",
    )
    out["date"] = out["acq_datetime"].dt.date
    out["year"] = out["acq_datetime"].dt.year
    out["month"] = out["acq_datetime"].dt.month
    out["hour"] = out["acq_datetime"].dt.hour

    # geography
    lat = pd.to_numeric(out["latitude"])
    lon = pd.to_numeric(out["longitude"])
    out["latitude"], out["longitude"] = lat, lon
    cell_lat = _grid_cell(lat, cell_size)
    cell_lon = _grid_cell(lon, cell_size)
    out["cell_id"] = cell_lat.map("{:.2f}".format) + "_" + cell_lon.map("{:.2f}".format)

    # confidence: VIIRS uses l / n / h
    out["confidence_label"] = (
        out["confidence"].astype("string").str.lower().map(CONFIDENCE_LABELS).fillna("unknown")
    )

    # type 0 = presumed vegetation fire; other values are kept but flagged False
    type_num = pd.to_numeric(out["type"], errors="coerce")
    out["is_presumed_vegetation_fire"] = (type_num == 0).astype("boolean").where(type_num.notna(), pd.NA)

    # stable id: same detection -> same id on every run
    key = (
        lat.round(5).astype(str) + "|" + lon.round(5).astype(str) + "|"
        + out["acq_datetime"].dt.strftime("%Y-%m-%dT%H:%M") + "|" + out["satellite"].astype(str)
    )
    out["detection_id"] = key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])

    return out[OUTPUT_COLUMNS].reset_index(drop=True)