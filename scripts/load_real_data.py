import glob

import duckdb
import pandas as pd

from wildfire_platform.batch.load import build_warehouse, write_parquet
from wildfire_platform.batch.transform import transform
from wildfire_platform.batch.validate import validate

frames = []
for path in sorted(glob.glob("data/raw/*.csv")):
    raw = pd.read_csv(path, dtype={"acq_time": str})
    frames.append(validate(raw).clean)

df = transform(pd.concat(frames, ignore_index=True))
write_parquet(df, "data/processed/fire_detections.parquet")
counts = build_warehouse(df, "data/warehouse/wildfire.duckdb")
print("Tables:", counts)

con = duckdb.connect("data/warehouse/wildfire.duckdb", read_only=True)

print("\nDetections per year (all vs presumed vegetation fires):")
con.sql("""
    SELECT year, count(*) AS detections,
           count(*) FILTER (WHERE is_presumed_vegetation_fire) AS vegetation_fires
    FROM fact_fire_detection GROUP BY year ORDER BY year
""").show()

print("Seasonality: detections per month (all years, vegetation fires only):")
con.sql("""
    SELECT month, count(*) AS detections
    FROM fact_fire_detection WHERE is_presumed_vegetation_fire
    GROUP BY month ORDER BY month
""").show()

print("Top 5 cells (vegetation fires):")
con.sql("""
    SELECT g.cell_id, g.lat_min, g.lon_min, sum(a.vegetation_fire_detections) AS vegetation_fires
    FROM agg_fire_detections_daily a JOIN dim_geo_cell g USING (cell_id)
    GROUP BY ALL ORDER BY vegetation_fires DESC LIMIT 5
""").show()