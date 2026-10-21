import glob
import os

import pandas as pd

from wildfire_platform.batch.transform import transform
from wildfire_platform.batch.validate import validate

frames = []
for path in sorted(glob.glob("data/raw/*.csv")):
    raw = pd.read_csv(path, dtype={"acq_time": str})
    result = validate(raw)
    print(os.path.basename(path), result.report)
    frames.append(result.clean)

out = transform(pd.concat(frames, ignore_index=True))
print("\nShape:", out.shape)
print("\nRows per year:\n", out.groupby("year").size())
print("\nGrid cells used:", out["cell_id"].nunique())
print("\nVegetation-fire flag:\n", out["is_presumed_vegetation_fire"].value_counts(dropna=False))
print("\nDuplicate detection_ids:", out["detection_id"].duplicated().sum())
print(out.head(3).T)