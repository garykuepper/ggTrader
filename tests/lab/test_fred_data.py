"""Tests for the FRED (Federal Reserve Economic Data) CSV loader --
free, no API key required (fredgraph.csv endpoint)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.fred_data import PUBLISH_LAG_DAYS, available_as_of, parse_fred_csv


def _csv(series_id: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"observation_date,{series_id}"]
    lines += [f"{d},{v}" for d, v in rows]
    return "\n".join(lines) + "\n"


class TestParseFredCsv:
    def test_parses_rows_into_date_value_columns(self):
        raw = _csv("CPIAUCSL", [("2020-01-01", "258.678"), ("2020-02-01", "259.250")])
        df = parse_fred_csv(raw, series_id="CPIAUCSL")
        assert list(df.columns) == ["date", "value"]
        assert len(df) == 2
        assert df.iloc[0]["value"] == pytest.approx(258.678)
        assert df["date"].iloc[0] == pd.Timestamp("2020-01-01")

    def test_drops_missing_value_marker_dot_rows(self):
        """FRED encodes missing observations as a literal '.' -- must not
        silently coerce to a bogus float or crash."""
        raw = _csv("IRSTCI01JPM156N", [("2020-01-01", "0.05"), ("2020-02-01", ".")])
        df = parse_fred_csv(raw, series_id="IRSTCI01JPM156N")
        assert len(df) == 1
        assert df.iloc[0]["date"] == pd.Timestamp("2020-01-01")

    def test_empty_csv_returns_empty_frame_with_expected_columns(self):
        df = parse_fred_csv("observation_date,CPIAUCSL\n", series_id="CPIAUCSL")
        assert df.empty
        assert list(df.columns) == ["date", "value"]


class TestAvailableAsOf:
    def test_excludes_observations_within_the_publish_lag(self):
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2026-05-01", "2026-06-01"]), "value": [1.0, 2.0]}
        )
        asof = pd.Timestamp("2026-06-20")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        # 05-01 + 45d lag = 06-15 (published by 06-20); 06-01 + 45d = 07-16 (not yet).
        assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-05-01"]

    def test_includes_observation_exactly_at_the_lag_boundary(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-05-01"]), "value": [1.0]})
        asof = pd.Timestamp("2026-05-01") + pd.Timedelta(days=PUBLISH_LAG_DAYS)
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert len(out) == 1

    def test_handles_tz_aware_asof_against_naive_dates(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-05-01"]), "value": [1.0]})
        asof = pd.Timestamp("2026-06-20", tz="UTC")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert len(out) == 1

    def test_returned_frame_has_tz_naive_date_even_if_input_was_tz_aware(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-05-01", tz="UTC")], "value": [1.0]})
        asof = pd.Timestamp("2026-06-20", tz="UTC")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert out["date"].dt.tz is None
        age = pd.Timestamp(asof).tz_localize(None) - out["date"].iloc[0]
        assert age.days == 50


class TestRowsForUpsert:
    def test_values_stay_floats_not_corrupted_to_nat(self):
        """Regression: DataFrame.iterrows() over a frame mixing a
        datetime64 date column and a float64 value column can coerce a
        row's value to NaT for some rows once enough rows are present --
        a genuine pandas dtype-unification gotcha, not user error. Must
        use a row-construction method immune to it (e.g. to_dict('records'))."""
        from ggTrader.lab.fred_data import _rows_for_upsert

        idx = pd.date_range("2020-01-01", periods=954, freq="MS")
        df = pd.DataFrame({"date": idx, "value": np.linspace(20.0, 330.0, 954)})
        rows = _rows_for_upsert("CPIAUCSL", df)
        assert all(isinstance(r["value"], float) for r in rows)
        assert rows[-1]["value"] == pytest.approx(330.0)


@pytest.mark.integration
def test_cache_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.fred_data import cache_series, ensure_schema, load_fred_series
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_FRED"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM fred_series WHERE series_id = :s"), {"s": marker})

    raw = _csv(marker, [("2020-01-01", "1.23"), ("2020-02-01", "1.45")])
    n = cache_series(marker, http_fetch=lambda sid: raw)
    assert n == 2

    df = load_fred_series(marker, "2019-01-01", "2021-01-01")
    assert len(df) == 2
    assert df.iloc[0]["value"] == pytest.approx(1.23)

    # Re-caching upserts, not duplicates.
    cache_series(marker, http_fetch=lambda sid: raw)
    df2 = load_fred_series(marker, "2019-01-01", "2021-01-01")
    assert len(df2) == 2

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM fred_series WHERE series_id = :s"), {"s": marker})
