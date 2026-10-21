"""Batch pipeline: extract -> validate -> transform -> load -> report.

The steps are plain functions so they can be tested without Prefect.
flows/batch_flow.py wraps the same functions as Prefect tasks.
"""
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from wildfire_platform.batch.extract import build_manifest, list_raw_files, read_raw
from wildfire_platform.batch.load import build_warehouse, write_parquet
from wildfire_platform.batch.transform import transform
from wildfire_platform.batch.validate import ValidationError, validate


def extract(raw_dir):
    files = list_raw_files(raw_dir)
    frames = [read_raw(f) for f in files]
    return files, frames, build_manifest(files, frames)


def validate_all(files, frames):
    """Validate each file on its own, so an error names the file."""
    clean, rejected, reports = [], [], []
    for f, df in zip(files, frames):
        try:
            result = validate(df)
        except ValidationError as exc:
            raise ValidationError(f"{f.name}: {exc}") from exc
        clean.append(result.clean)
        if len(result.rejected):
            rejected.append(result.rejected.assign(source_file=f.name))
        reports.append(result.report)
    return clean, rejected, reports


def transform_all(clean, cell_size=0.5):
    df = transform(pd.concat(clean, ignore_index=True), cell_size=cell_size)
    before = len(df)
    df = df.drop_duplicates(subset="detection_id").reset_index(drop=True)
    return df, before - len(df)  # second value: duplicates found across files


def load_all(df, rejected, manifest, out_dir, db_path, cell_size=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df, out_dir / "fire_detections.parquet")
    tables = build_warehouse(df, db_path, cell_size=cell_size)
    rejected_path = out_dir / "rejected_rows.csv"
    if rejected:
        pd.concat(rejected, ignore_index=True).to_csv(rejected_path, index=False)
    else:
        rejected_path.unlink(missing_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int))
    return tables


def build_report(manifest, reports, cross_file_duplicates, df, tables, started):
    reasons = Counter()
    for r in reports:
        reasons.update(r["rejected_by_reason"])
    report = {
        "files": len(manifest),
        "rows_in": sum(m["rows"] for m in manifest),
        "rows_rejected": sum(r["rows_rejected"] for r in reports),
        "rejected_by_reason": dict(reasons),
        "cross_file_duplicates": cross_file_duplicates,
        "rows_out": len(df),
        "tables": tables,
        "duration_seconds": round(time.perf_counter() - started, 2),
    }
    expected = report["rows_rejected"] + cross_file_duplicates + report["rows_out"]
    if report["rows_in"] != expected:
        raise RuntimeError(f"Row counts do not add up: in={report['rows_in']}, accounted={expected}")
    return report


def write_report(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, default=int))


def run_batch(raw_dir="data/raw", out_dir="data/processed",
              db_path="data/warehouse/wildfire.duckdb", cell_size=0.5) -> dict:
    started = time.perf_counter()
    files, frames, manifest = extract(raw_dir)
    clean, rejected, reports = validate_all(files, frames)
    df, dupes = transform_all(clean, cell_size)
    tables = load_all(df, rejected, manifest, out_dir, db_path, cell_size)
    report = build_report(manifest, reports, dupes, df, tables, started)
    write_report(report, out_dir)
    return report