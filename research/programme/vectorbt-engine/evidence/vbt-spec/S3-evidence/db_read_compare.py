import json
import sys

import numpy as np
import pandas as pd

E = sys.argv[1]
a = pd.read_parquet(f"{E}/db_old/wide.parquet")
b = pd.read_parquet(f"{E}/db_new/wide.parquet")
print("shape", a.shape, b.shape)
print(
    "index values equal (as UTC instants):",
    bool((a.index.tz_convert("UTC").as_unit("ns") == b.index.tz_convert("UTC").as_unit("ns")).all()),
)
print(
    "columns equal as strings:",
    [tuple(map(str, c)) for c in a.columns] == [tuple(map(str, c)) for c in b.columns],
)
print(
    "max abs value diff:",
    float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))),
    "bit-identical:",
    bool(np.array_equal(a.to_numpy(), b.to_numpy())),
)
print("old idx dtype", a.index.dtype, "| new", b.index.dtype)
print("old col dtype", a.columns.levels[0].dtype, "| new", b.columns.levels[0].dtype)
k = "write_capture_utc_midnight.json"
x = json.load(open(f"{E}/db_old/{k}"))
y = json.load(open(f"{E}/db_new/{k}"))
print(k, "identical:", x == y)
