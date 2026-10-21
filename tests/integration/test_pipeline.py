import json

import pandas as pd
import pytest

from wildfire_platform.batch.extract import ExtractError
from wildfire_platform.batch.pipeline import run_batch
from wildfire_platform.batch.validate import ValidationError

HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight,type\n"
)


def line(lat, lon, date, time):
    return f"{lat},{lon},300.0,0.4,0.6,{date},{time},N,VIIRS,n,2,280.0,1.4,N,0\n"


R1 = line(33.76, -7.43, "2025-07-01", "0251")
R2 = line(34.10, -6.20, "2025-07-02", "1249")
R3 = line(35.20, -5.30, "2025-07-03", "0130")
BAD = line(95.0, -7.43, "2025-07-01", "0300")  # latitude out of range


def run(base, files):
    raw = base / "raw"
    raw.mkdir(parents=True)
    for name, rows in files.items():
        (raw / name).write_text(HEADER + "".join(rows))
    return run_batch(raw, base / "out", base / "wh" / "w.duckdb")


def test_end_to_end_report_and_outputs(tmp_path):
    report = run(tmp_path, {"a.csv": [R1, R2, BAD], "b.csv": [R3]})
    assert report["rows_in"] == 4
    assert report["rows_rejected"] == 1
    assert report["rejected_by_reason"] == {"invalid_coordinates": 1}
    assert report["rows_out"] == 3
    assert report["tables"]["fact_fire_detection"] == 3
    out = tmp_path / "out"
    for name in ["fire_detections.parquet", "manifest.json", "run_report.json", "rejected_rows.csv"]:
        assert (out / name).exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert [m["file"] for m in manifest] == ["a.csv", "b.csv"]
    assert all(len(m["sha256"]) == 64 for m in manifest)


def test_duplicate_across_files_is_removed_and_reported(tmp_path):
    report = run(tmp_path, {"a.csv": [R1], "b.csv": [R1, R2]})
    assert report["cross_file_duplicates"] == 1
    assert report["rows_out"] == 2


def test_same_input_gives_same_output(tmp_path):
    files = {"a.csv": [R1, R2], "b.csv": [R3]}
    first = run(tmp_path / "run1", files)
    second = run(tmp_path / "run2", files)
    first.pop("duration_seconds")
    second.pop("duration_seconds")
    assert first == second
    ids1 = pd.read_parquet(tmp_path / "run1" / "out" / "fire_detections.parquet")["detection_id"].tolist()
    ids2 = pd.read_parquet(tmp_path / "run2" / "out" / "fire_detections.parquet")["detection_id"].tolist()
    assert ids1 == ids2


def test_missing_column_fails_and_names_the_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.csv").write_text("longitude,acq_date,acq_time\n-7.4,2025-07-01,0251\n")
    with pytest.raises(ValidationError, match="bad.csv"):
        run_batch(raw, tmp_path / "out", tmp_path / "w.duckdb")


def test_empty_raw_dir_fails(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(ExtractError):
        run_batch(tmp_path / "raw", tmp_path / "out", tmp_path / "w.duckdb")