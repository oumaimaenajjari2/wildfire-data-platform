import duckdb

# Reads every CSV in data/raw/ (DuckDB can query CSV files directly with SQL)
SRC = "data/raw/*.csv"
con = duckdb.connect()

print("\n1) COLUMNS AND TYPES")
con.sql(f"DESCRIBE SELECT * FROM '{SRC}'").show(max_rows=50)

print("\n2) FIRST 5 ROWS")
con.sql(f"SELECT * FROM '{SRC}' LIMIT 5").show(max_width=250)

print("\n3) ROW COUNT AND DATE RANGE")
con.sql(f"SELECT count(*) AS rows, min(acq_date) AS first_day, max(acq_date) AS last_day FROM '{SRC}'").show()

print("\n4) CONFIDENCE VALUES")
con.sql(f"SELECT confidence, count(*) AS n FROM '{SRC}' GROUP BY 1 ORDER BY n DESC").show()

print("\n5) LAT/LON RANGE")
con.sql(f"SELECT min(latitude), max(latitude), min(longitude), max(longitude) FROM '{SRC}'").show()

print("\n6) DUPLICATES (same position, day and time)")
con.sql(f"""
SELECT count(*) - count(DISTINCT (latitude, longitude, acq_date, acq_time)) AS duplicate_rows
FROM '{SRC}'
""").show()