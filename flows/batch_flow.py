"""Prefect flow: the same batch steps, run as tracked tasks.

Run from the project root:  python flows/batch_flow.py
"""
import time

from prefect import flow, task

from wildfire_platform.batch import pipeline

extract_t = task(pipeline.extract, name="extract")
validate_t = task(pipeline.validate_all, name="validate")
transform_t = task(pipeline.transform_all, name="transform")
load_t = task(pipeline.load_all, name="load")


@flow(name="wildfire-batch-pipeline", log_prints=True)
def batch_flow(raw_dir="data/raw", out_dir="data/processed",
               db_path="data/warehouse/wildfire.duckdb", cell_size=0.5):
    started = time.perf_counter()
    files, frames, manifest = extract_t(raw_dir)
    clean, rejected, reports = validate_t(files, frames)
    df, dupes = transform_t(clean, cell_size)
    tables = load_t(df, rejected, manifest, out_dir, db_path, cell_size)
    report = pipeline.build_report(manifest, reports, dupes, df, tables, started)
    pipeline.write_report(report, out_dir)
    print("Run report:", report)
    return report


if __name__ == "__main__":
    batch_flow()