"""Tests for early_roster and early_inputs: death roster, rank bands, permno, momentum parity."""

from __future__ import annotations

import early_inputs
import early_roster as er
import ef_dead_stock_first_test as fp
import ef_replay
import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.xs_momentum import momentum_scores

pytestmark = pytest.mark.study


def _make_panel(
    start_cal: str,
    end_cal: str,
    cols: list[str],
    *,
    seed: int = 7,
    tz: str | None = "UTC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic business-day closes and all-True eligibility for ``cols``."""
    days = pd.bdate_range(start_cal, end_cal, tz=tz)
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.015, size=(len(days), len(cols)))
    prices = 50.0 * np.exp(np.cumsum(steps, axis=0))
    close = pd.DataFrame(prices, index=days, columns=cols)
    eligible = pd.DataFrame(True, index=days, columns=cols)
    return close, eligible


def _make_ranked_panel(
    start_cal: str,
    end_cal: str,
    n_stocks: int,
    *,
    tz: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic panel where ``S000`` has rank 1 through ``S{n-1}`` at rank ``n``."""
    days = pd.bdate_range(start_cal, end_cal, tz=tz)
    t = np.arange(len(days), dtype=float)[:, None]
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 100.0 * np.exp(t * drifts)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    eligible = pd.DataFrame(True, index=days, columns=cols)
    return close, eligible


def _momentum_deaths_reference(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    *,
    start: str,
    end: str,
    crsp_end: str | None = None,
) -> dict[str, dict[str, int]]:
    """Independent copy of the momentum-deaths study's death-counting definition.

    Copied directly from ``momentum_deaths.compute_measures`` (never imported across studies)
    so the roster count and band counts can be verified against the exact reference definition.
    """
    spans, _ = fp.drop_file_end_rows(fp._normalise_spans(span_rows), end, crsp_end=crsp_end)
    ends = ef_replay.span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    death_dates = ends[~ends.index.isin(successors)]

    index = close.index
    days = ef_replay._decision_days(index, start, end)
    if not days:
        return {
            "A1_totals": {"deaths": 0},
            "A2_by_band": {"1-50": 0, "51-150": 0, "151-500": 0, "501+": 0},
        }
    scores = momentum_scores(close, fp.LOOKBACK, fp.SKIP).reindex(pd.DatetimeIndex(days)).to_numpy()
    allowed = ef_replay._aligned_eligibility(eligible, close, days)
    ended = ends.reindex(close.columns)
    last_exec = ef_replay._localize(pd.Timestamp(end), index)

    sets: list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray]] = []
    for i, decision in enumerate(days):
        pos = int(index.get_loc(decision)) + 1
        if pos >= len(index) or index[pos] > last_exec:
            continue
        already_dead = (ended < decision).to_numpy()
        usable = np.flatnonzero(allowed[i] & np.isfinite(scores[i]) & ~already_dead)
        order = np.argsort(-scores[i][usable], kind="stable")
        sets.append((decision, index[pos], usable[order]))

    window_end = ef_replay._localize(pd.Timestamp(end), index)
    symbols = np.array(close.columns)
    death_map = death_dates.to_dict()
    bands: tuple[tuple[str, int, int | None], ...] = (
        ("1-50", 1, 50),
        ("51-150", 51, 150),
        ("151-500", 151, 500),
        ("501+", 501, None),
    )
    band_deaths = {name: 0 for name, _, _ in bands}
    total_deaths = 0
    for i, (_dec, exec_date, ranked) in enumerate(sets):
        hold_end = sets[i + 1][1] - pd.Timedelta(days=1) if i + 1 < len(sets) else window_end
        month_deaths = {
            int(col)
            for col in ranked
            if (lb := death_map.get(str(symbols[col]))) is not None and exec_date <= lb <= hold_end
        }
        total_deaths += len(month_deaths)
        for name, lo_rank, hi_rank in bands:
            slice_cols = ranked[lo_rank - 1 : hi_rank]
            band_deaths[name] += sum(1 for c in slice_cols if int(c) in month_deaths)

    return {
        "A1_totals": {"deaths": total_deaths},
        "A2_by_band": band_deaths,
    }


def test_roster_counts_the_deaths_in_a_synthetic_panel() -> None:
    # 3 decision months:
    # Month 0: dec 2000-01-31 -> exec 2000-02-01, hold_end 2000-02-29
    # Month 1: dec 2000-02-29 -> exec 2000-03-01, hold_end 2000-04-02
    # Month 2: dec 2000-03-31 -> exec 2000-04-03, hold_end 2000-04-28
    cols = [f"S{i:03d}" for i in range(12)]
    close, eligible = _make_panel("1998-06-01", "2000-04-28", cols, seed=21, tz=None)
    start, end = "2000-01-03", "2000-04-28"

    # 3 real deaths (S001, S002, S003), 1 successor (S004), 1 file-end row (S005),
    # plus 1 stock ending on decision day 0 before exec 0 (S000, not a holding-month death).
    spec = [
        ("S000", "2000-01-31", False),
        ("S001", "2000-02-01", False),  # exec day of month 0 -> counted
        ("S002", "2000-02-29", False),  # last day of month 0 -> counted
        ("S003", "2000-03-15", False),  # inside month 1 -> counted
        ("S004", "2000-02-15", True),  # successor -> excluded
        ("S005", "2000-04-28", False),  # file-end row (equals crsp_end) -> excluded
    ]
    for sym, last_day, _ in spec:
        close.loc[close.index > pd.Timestamp(last_day), sym] = np.nan

    span_rows = pd.DataFrame(
        {
            "symbol": [s for s, _, _ in spec],
            "last_date": pd.to_datetime([d for _, d, _ in spec]),
            "has_successor": [succ for _, _, succ in spec],
            "total_loss": [False] * len(spec),
        }
    )

    roster = er.build_death_roster(
        close, eligible, span_rows, start=start, end=end, crsp_end="2000-04-28"
    )
    assert list(roster.columns) == [
        "column",
        "permno",
        "symbol",
        "last_bar",
        "decision_date",
        "rank_band",
        "held_by_rule",
    ]
    assert len(roster) == 3
    by_col = roster.set_index("column")
    assert set(by_col.index) == {"S001", "S002", "S003"}
    assert by_col.loc["S001", "last_bar"] == pd.Timestamp("2000-02-01")
    assert by_col.loc["S001", "decision_date"] == pd.Timestamp("2000-01-31")
    assert by_col.loc["S002", "last_bar"] == pd.Timestamp("2000-02-29")
    assert by_col.loc["S002", "decision_date"] == pd.Timestamp("2000-01-31")
    assert by_col.loc["S003", "last_bar"] == pd.Timestamp("2000-03-15")
    assert by_col.loc["S003", "decision_date"] == pd.Timestamp("2000-02-29")


def test_rank_band_and_held_flag_follow_the_decision_day_ranking() -> None:
    close, eligible = _make_ranked_panel("1998-10-01", "2000-03-31", 520, tz=None)

    # Plant deaths at every band edge in month 0 (decision 2000-01-31, holding Feb 2000):
    # rank 1 (S000), 50 (S049), 51 (S050), 150 (S149), 151 (S150), 500 (S499),
    # 501 (S500), 520 (S519).
    dying = ["S000", "S049", "S050", "S149", "S150", "S499", "S500", "S519"]
    close.loc["2000-02-15":, dying] = np.nan
    span_rows = pd.DataFrame(
        {
            "symbol": dying,
            "last_date": pd.to_datetime(["2000-02-14"] * len(dying)),
            "has_successor": [False] * len(dying),
            "total_loss": [False] * len(dying),
        }
    )

    roster = er.build_death_roster(close, eligible, span_rows, start="2000-01-03", end="2000-03-31")
    assert len(roster) == 8
    assert roster["held_by_rule"].dtype == bool
    by_col = roster.set_index("column")

    expected = {
        "S000": ("1-50", True),
        "S049": ("1-50", True),
        "S050": ("51-150", False),
        "S149": ("51-150", False),
        "S150": ("151-500", False),
        "S499": ("151-500", False),
        "S500": ("501+", False),
        "S519": ("501+", False),
    }
    for col, (band, held) in expected.items():
        assert by_col.loc[col, "rank_band"] == band
        assert bool(by_col.loc[col, "held_by_rule"]) is held
        assert by_col.loc[col, "decision_date"] == pd.Timestamp("2000-01-31")


def test_roster_equals_the_momentum_deaths_definition() -> None:
    # Mirrors the 520-stock synthetic panel from test_momentum_deaths.py, plus a second
    # month with a successor, an ineligible stock, a tie, and a file-end row.
    close, eligible = _make_ranked_panel("2018-10-01", "2020-03-31", 520, tz="UTC")
    # Tie between S000 and S001 to check stable tie-breaking parity.
    close["S001"] = close["S000"].copy()
    # Make S002 ineligible on decision day 1 (2020-02-28) so its March death is excluded.
    eligible.loc[pd.Timestamp("2020-02-28", tz="UTC"), "S002"] = False

    deaths_feb = ["S000", "S049", "S050", "S150", "S300", "S499", "S500"]
    deaths_mar = ["S002", "S010", "S060"]  # S002 ineligible on dec 1 -> excluded
    successor = ["S005"]
    file_end = ["S020"]

    close.loc[pd.Timestamp("2020-02-15", tz="UTC") :, deaths_feb] = np.nan
    close.loc[pd.Timestamp("2020-02-15", tz="UTC") :, successor] = np.nan
    close.loc[pd.Timestamp("2020-03-16", tz="UTC") :, deaths_mar] = np.nan

    all_syms = deaths_feb + deaths_mar + successor + file_end
    all_dates = (
        ["2020-02-14"] * len(deaths_feb)
        + ["2020-03-13"] * len(deaths_mar)
        + ["2020-02-14"] * len(successor)
        + ["2020-03-31"] * len(file_end)
    )
    all_succ = [False] * (len(deaths_feb) + len(deaths_mar)) + [True] + [False]
    span_rows = pd.DataFrame(
        {
            "symbol": all_syms,
            "last_date": pd.to_datetime(all_dates),
            "has_successor": all_succ,
            "total_loss": [False] * len(all_syms),
        }
    )

    ref = _momentum_deaths_reference(
        close,
        eligible,
        span_rows,
        start="2020-01-02",
        end="2020-03-31",
        crsp_end="2020-03-31",
    )
    roster = er.build_death_roster(
        close,
        eligible,
        span_rows,
        start="2020-01-02",
        end="2020-03-31",
        crsp_end="2020-03-31",
    )

    assert len(roster) == ref["A1_totals"]["deaths"]
    assert len(roster) == 9  # 7 in Feb + 2 in Mar (S002 ineligible, S005 successor, S020 file-end)
    assert int(roster["held_by_rule"].sum()) == ref["A2_by_band"]["1-50"]
    for band, want in ref["A2_by_band"].items():
        got = int((roster["rank_band"] == band).sum())
        assert got == want, f"band {band}: got {got}, expected {want}"
    assert "S002" not in set(roster["column"])
    assert "S005" not in set(roster["column"])
    assert "S020" not in set(roster["column"])


def test_tilde_suffix_keeps_base_symbol_and_reads_permno_from_release_spans() -> None:
    cols = ["PERMNO-10001~1", "PERMNO-10002", "ACME~2", "FOO", "BAR", "NOPRM", "ALIVE"]
    close, eligible = _make_panel("1998-06-01", "2000-03-31", cols, seed=9, tz=None)
    dying = ["PERMNO-10001~1", "PERMNO-10002", "ACME~2", "FOO", "BAR", "NOPRM"]
    close.loc["2000-02-15":, dying] = np.nan

    span_rows = pd.DataFrame(
        {
            "symbol": dying,
            "permno": [pd.NA, pd.NA, pd.NA, pd.NA, 77777, pd.NA],
            "last_date": pd.to_datetime(["2000-02-14"] * len(dying)),
            "has_successor": [False] * len(dying),
            "total_loss": [False] * len(dying),
        }
    )
    release_spans = pd.DataFrame(
        {
            "symbol": ["PERMNO-10001~1", "PERMNO-10002", "ACME~2", "FOO", "ALIVE"],
            "permno": [10001, 10002, 55555, 66666, 99999],
            "spell": [1, 1, 2, 1, 1],
            "first_date": pd.to_datetime(["1998-06-01"] * 5),
            "last_date": pd.to_datetime(["2000-02-14"] * 4 + ["2000-03-31"]),
            "kind": ["spell", "placeholder", "spell", "linked_yahoo", "linked_yahoo"],
            "possible_double": [False] * 5,
        }
    )

    roster = er.build_death_roster(
        close,
        eligible,
        span_rows,
        start="2000-01-03",
        end="2000-03-31",
        release_spans=release_spans,
    )
    assert str(roster["permno"].dtype) == "Int64"
    by_col = roster.set_index("column")
    assert by_col.loc["PERMNO-10001~1", "symbol"] == "PERMNO-10001"
    assert by_col.loc["PERMNO-10001~1", "permno"] == 10001
    assert by_col.loc["PERMNO-10002", "symbol"] == "PERMNO-10002"
    assert by_col.loc["PERMNO-10002", "permno"] == 10002
    assert by_col.loc["ACME~2", "symbol"] == "ACME"
    assert by_col.loc["ACME~2", "permno"] == 55555
    assert by_col.loc["FOO", "symbol"] == "FOO"
    assert by_col.loc["FOO", "permno"] == 66666
    assert by_col.loc["BAR", "symbol"] == "BAR"
    assert by_col.loc["BAR", "permno"] == 77777
    assert by_col.loc["NOPRM", "symbol"] == "NOPRM"
    assert pd.isna(by_col.loc["NOPRM", "permno"])

    # Without release_spans, PERMNO-<id>[~k] columns still resolve their permno from the name.
    no_rel = er.build_death_roster(
        close, eligible, span_rows, start="2000-01-03", end="2000-03-31"
    ).set_index("column")
    assert no_rel.loc["PERMNO-10001~1", "permno"] == 10001
    assert no_rel.loc["PERMNO-10002", "permno"] == 10002
    assert no_rel.loc["BAR", "permno"] == 77777
    assert pd.isna(no_rel.loc["FOO", "permno"])


def test_empty_roster_has_expected_columns_and_dtypes() -> None:
    close, eligible = _make_panel("1998-06-01", "2000-03-31", ["S000", "S001"], seed=5, tz=None)
    span_rows = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=object),
            "last_date": pd.to_datetime([]),
            "has_successor": pd.Series(dtype=bool),
            "total_loss": pd.Series(dtype=bool),
        }
    )
    roster = er.build_death_roster(close, eligible, span_rows, start="2000-01-03", end="2000-03-31")
    assert roster.empty
    assert list(roster.columns) == er.ROSTER_COLUMNS
    assert str(roster["permno"].dtype) == "Int64"
    assert roster["held_by_rule"].dtype == bool


def test_early_inputs_has_only_early_window_and_validates_window() -> None:
    assert early_inputs.WINDOWS == {"early": ("2000-01-03", "2004-12-31")}
    assert early_inputs.UNIVERSE == "A"
    with pytest.raises(ValueError, match="unknown window"):
        early_inputs.read_inputs("dev", 1, 1)
    assert early_inputs.drift_in({}) == []
    assert early_inputs.drift_in(
        {"data_provenance": {"candidates": {"drift": True, "drift_symbols": ["BBB", "AAA"]}}}
    ) == ["AAA", "BBB"]
