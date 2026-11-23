"""Checks for the CRSP pilot command and projection."""

import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.kaggle_pilot import project, rows_by_year


def test_cli():
    for args, code in [(["--help"], 0), ([], 2)]:
        result = subprocess.run(
            [sys.executable, "scripts/kaggle_pilot.py", *args], capture_output=True, check=False
        )
        assert result.returncode == code


def test_year_counts(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"year": [1999, 2000, 1999, 2001], "permno": [1, 2, 3, 4]}), path, row_group_size=2
    )
    assert rows_by_year(path) == {1999: 2, 2000: 1, 2001: 1}


def test_projection():
    assert project({1999: 2, 2000: 3}, 1999, 100, 40, 2, 1000, 100) == {
        "full_uncompressed_bytes": 250,
        "full_compressed_bytes": 100,
        "reload_peak_bytes": 385,
        "free_after_full_load": 900,
        "free_after_reload_peak": 615,
        "margin_ok": True,
    }
