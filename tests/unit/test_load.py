import duckdb
import pandas as pd

from wildfire_platform.batch.load import build_warehouse, write_parquet
from wildfire_platform.batch.transform import transform

TABLES = {"fact_fire_detection", "dim_date", "dim_geo_cell", "agg_fire_detections_daily"}


def make_df():
    base = {
        "satellite": "N", "instrument": "VIIRS", "daynight": "N",
        "bright_ti4": 300.0, "bright_ti5": 280.0, "version": 2,
    }
    rows = [
        # two detections, same cell, same day
        {**base, "latitude": 33.76, "longitude": -7.43, "acq_date": "2025-07-01",
         "acq_time": "0251", "confidence": "h", "frp": 2.0, "type": 0},
        {**base, "latitude": 33.80, "longitude": -7.45, "acq_date": "2025-07-01",
         "acq_time": "0252", "confidence": "n", "frp": 1.0, "type": 2},
        # two days later: 2025-07-02 has no detections (a gap day)
        {**base, "latitude": 33.76, "longitude": -7.43, "acq_date": "2025-07-03",
         "acq_time": "1249", "confidence": "n", "frp": 3.0, "type": 0},
    ]
    return transform(pd.DataFrame(rows))


def query(db, sql):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_all_tables_are_created(tmp_path):
    counts = build_warehouse(make_df(), tmp_path / "w.duckdb")
    assert set(counts) == TABLES
    assert counts["fact_fire_detection"] == 3


def test_daily_aggregate_counts(tmp_path):
    db = tmp_path / "w.duckdb"
    build_warehouse(make_df(), db)
    rows = query(db, """
        SELECT CAST(date AS VARCHAR), detections, vegetation_fire_detections,
               high_confidence_detections, total_frp
        FROM agg_fire_detections_daily ORDER BY date
    """)
    assert rows == [("2025-07-01", 2, 1, 1, 3.0), ("2025-07-03", 1, 1, 0, 3.0)]


def test_dim_date_has_no_gaps(tmp_path):
    db = tmp_path / "w.duckdb"
    build_warehouse(make_df(), db)
    assert query(db, "SELECT count(*) FROM dim_date")[0][0] == 3  # 1, 2 and 3 July


def test_geo_cell_bounds(tmp_path):
    db = tmp_path / "w.duckdb"
    build_warehouse(make_df(), db, cell_size=0.5)
    rows = query(db, "SELECT cell_id, lat_min, lat_max, lon_min, lon_max FROM dim_geo_cell")
    assert rows == [("33.50_-7.50", 33.5, 34.0, -7.5, -7.0)]


def test_rebuild_is_idempotent(tmp_path):
    db = tmp_path / "w.duckdb"
    df = make_df()
    assert build_warehouse(df, db) == build_warehouse(df, db)


def test_parquet_roundtrip(tmp_path):
    df = make_df()
    path = write_parquet(df, tmp_path / "out" / "fact.parquet")
    back = pd.read_parquet(path)
    assert len(back) == 3
    assert list(back["detection_id"]) == list(df["detection_id"])