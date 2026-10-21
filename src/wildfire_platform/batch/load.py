"""Load transformed detections into Parquet and a local DuckDB warehouse.

Tables:
- fact_fire_detection: one row per detection
- dim_date: one row per calendar day, including days with no detection
- dim_geo_cell: one row per grid cell, with its bounding box
- agg_fire_detections_daily: counts and FRP per day and cell

Every run rebuilds the tables from scratch (CREATE OR REPLACE), so the same
input always gives the same output.
"""
from pathlib import Path

import duckdb
import pandas as pd

TABLES = ["fact_fire_detection", "dim_date", "dim_geo_cell", "agg_fire_detections_daily"]


def write_parquet(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    days = pd.date_range(dates.min(), dates.max(), freq="D")
    return pd.DataFrame({
        "date": days.date,
        "year": days.year,
        "month": days.month,
        "day": days.day,
        "quarter": days.quarter,
        "day_of_week": days.dayofweek + 1,  # 1 = Monday
        "day_name": days.day_name(),
        "week_of_year": days.isocalendar()["week"].astype(int).to_numpy(),
    })


def build_warehouse(df: pd.DataFrame, db_path, cell_size: float = 0.5) -> dict:
    """Create the 4 tables in a DuckDB file. Returns the row count of each table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cell_size = float(cell_size)
    dim_date = _build_dim_date(df)

    con = duckdb.connect(str(db_path))
    try:
        con.register("fact_src", df)
        con.register("dim_date_src", dim_date)

        con.execute("""
            CREATE OR REPLACE TABLE fact_fire_detection AS
            SELECT * REPLACE (CAST(date AS DATE) AS date) FROM fact_src
        """)
        con.execute("""
            CREATE OR REPLACE TABLE dim_date AS
            SELECT * REPLACE (CAST(date AS DATE) AS date) FROM dim_date_src
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE dim_geo_cell AS
            SELECT
                cell_id,
                CAST(split_part(cell_id, '_', 1) AS DOUBLE) AS lat_min,
                CAST(split_part(cell_id, '_', 1) AS DOUBLE) + {cell_size} AS lat_max,
                CAST(split_part(cell_id, '_', 2) AS DOUBLE) AS lon_min,
                CAST(split_part(cell_id, '_', 2) AS DOUBLE) + {cell_size} AS lon_max,
                {cell_size} AS cell_size_deg
            FROM (SELECT DISTINCT cell_id FROM fact_fire_detection)
            ORDER BY cell_id
        """)
        con.execute("""
            CREATE OR REPLACE TABLE agg_fire_detections_daily AS
            SELECT
                date,
                cell_id,
                count(*) AS detections,
                count(*) FILTER (WHERE is_presumed_vegetation_fire) AS vegetation_fire_detections,
                count(*) FILTER (WHERE confidence_label = 'high') AS high_confidence_detections,
                round(sum(frp), 2) AS total_frp,
                max(frp) AS max_frp
            FROM fact_fire_detection
            GROUP BY date, cell_id
            ORDER BY date, cell_id
        """)

        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        con.close()