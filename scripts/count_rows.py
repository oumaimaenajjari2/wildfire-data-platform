import duckdb

duckdb.sql("""
    SELECT filename, count(*) AS rows
    FROM read_csv('data/raw/*.csv', filename=true)
    GROUP BY filename
    ORDER BY filename
""").show(max_width=200)