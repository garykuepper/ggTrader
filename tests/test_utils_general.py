"""Unit tests for general utility functions."""

import pytest
import pandas as pd
import numpy as np
from ggTrader.utils.utils import (
    make_end_anchored_tscv,
    periods_per_year_from_interval,
    convert_cols_to_numeric,
)


def test_make_end_anchored_tscv():
    n_samples = 100
    n_splits = 3
    test_ratio = 0.2

    tscv, test_size, max_train_size = make_end_anchored_tscv(n_samples, n_splits, test_ratio)

    assert n_splits == tscv.n_splits
    assert test_size > 0
    assert max_train_size > 0

    # Check splits
    splits = list(tscv.split(np.arange(n_samples)))
    assert len(splits) == 3


def test_periods_per_year_from_interval():
    assert periods_per_year_from_interval("1h") == 24 * 365
    assert periods_per_year_from_interval("4h") == 6 * 365
    assert periods_per_year_from_interval("1d") == 365


def test_convert_cols_to_numeric():
    # convert_cols_to_numeric requires "Start", "End", and "Period" columns
    df = pd.DataFrame(
        {
            "Start": ["2023-01-01", "2023-01-02"],
            "End": ["2023-01-02", "2023-01-03"],
            "Period": ["1h", "4h"],
            "Value": ["1.5", "2.5"],
            "Metric": [10, 20],
        }
    )

    df_clean = convert_cols_to_numeric(df)

    assert pd.api.types.is_datetime64_any_dtype(df_clean["Start"])
    assert pd.api.types.is_datetime64_any_dtype(df_clean["End"])
    assert pd.api.types.is_timedelta64_ns_dtype(df_clean["Period"])
    assert pd.api.types.is_numeric_dtype(df_clean["Value"])
    assert df_clean["Value"].iloc[0] == 1.5
