"""OhlcvFetchChain: the price-loader fallback, tested with fakes and no connections."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from ggTrader.data.live import fetch_chain
from ggTrader.data.live.fetch_chain import GapFiller, OhlcvFetchChain, fetch_stock_ohlcv
from tests.netguard import blocked_connections

A, B, C = "ZZTEST1", "ZZTEST2", "ZZTEST3"
START, END = "2024-01-01", "2024-01-03"
START_TS, END_TS = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC")

YF_INTRO = "  [data] fetching {n} symbols missing from cache..."
TIINGO_INTRO = "  [data] trying Tiingo for {n} symbols yfinance missed..."
SUMMARY = "  [data] {rows} rows x {n} symbols (1d)"


def frame(symbols: list[str], days: list[str]) -> pd.DataFrame:
    """Build a provider frame with a close column for each symbol."""
    return pd.DataFrame(
        1.0,
        index=pd.to_datetime(days, utc=True),
        columns=pd.MultiIndex.from_product([symbols, ["close"]]),
    )


def request(symbols: list[str], cache: bool = False) -> tuple:
    """Expected provider call; only the cache is asked with ``limit=None``."""
    kwargs = {"start_date": START_TS, "end_date": END_TS}
    if cache:
        kwargs["limit"] = None
    return (symbols, "1d"), kwargs


def symbols_of(df: pd.DataFrame) -> list[str]:
    """The distinct symbols of a (symbol, field) frame, in column order."""
    return list(df.columns.get_level_values(0).unique())


@pytest.fixture(autouse=True)
def no_connections() -> Iterator[None]:
    """Fail the test if anything tries to reach a network or database."""
    with blocked_connections():
        yield


@pytest.fixture
def rig():
    """Fakes for the cache, primary, yfinance-gap and Tiingo-gap providers plus a negative cache."""
    r = SimpleNamespace(
        cache=Mock(name="cache"),
        primary=Mock(name="primary"),
        tiingo=Mock(name="tiingo"),
        neg=Mock(name="negative_cache"),
    )
    r.cache.fetch_ohlcv.return_value = frame([A, B], [START, END])
    r.primary.fetch_ohlcv.return_value = pd.DataFrame()
    r.tiingo.fetch_ohlcv.return_value = pd.DataFrame()
    r.neg.load_skip.return_value = set()
    r.make_cache = Mock(return_value=r.cache)
    r.make_primary = Mock(return_value=r.primary)
    r.make_tiingo = Mock(return_value=r.tiingo)
    return r


def yf_gap(make) -> GapFiller:
    """The yfinance gap filler: announces before it constructs and lets errors propagate."""
    return GapFiller(
        make=make,
        intro=YF_INTRO,
        announce_before_make=True,
        persist_error="gap fetch",
        swallow_errors=None,
    )


def tiingo_gap(make) -> GapFiller:
    """The Tiingo gap filler: announces after it constructs and swallows every error."""
    return GapFiller(
        make=make,
        intro=TIINGO_INTRO,
        announce_before_make=False,
        persist_error="Tiingo data",
        swallow_errors="Tiingo fallback unavailable",
    )


def chain_of(rig, *, cache: bool = True, negative: bool = False) -> OhlcvFetchChain:
    """The production chain shape (cache, primary/yfinance gap, Tiingo gap) over the fakes."""
    return OhlcvFetchChain(
        primary=rig.make_primary,
        fillers=[yf_gap(rig.make_primary), tiingo_gap(rig.make_tiingo)],
        cache=rig.make_cache if cache else None,
        negative_cache=rig.neg if negative else None,
    )


def fetch(rig, symbols=(B, A), **kwargs) -> pd.DataFrame:
    """Fetch the standard window through the standard chain."""
    cache = kwargs.pop("cache", True)
    negative = kwargs.pop("negative", False)
    return chain_of(rig, cache=cache, negative=negative).fetch(list(symbols), START, END, **kwargs)


def lines(capsys) -> list[str]:
    """Everything printed so far, one entry per line."""
    return capsys.readouterr().out.splitlines()


# ---- primary fallback -------------------------------------------------------------------------


def test_cache_hit_runs_no_fallback_and_no_gap(rig, capsys):
    result = fetch(rig)
    assert symbols_of(result) == [A, B]
    assert rig.cache.fetch_ohlcv.call_args_list == [request([A, B], True)]
    rig.make_primary.assert_not_called()
    rig.make_tiingo.assert_not_called()
    rig.cache.persist.assert_not_called()
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]


def test_primary_runs_when_cache_left_df_empty_with_no_announcement(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.primary.fetch_ohlcv.return_value = frame([A, B], [START, END])
    result = fetch(rig)
    assert symbols_of(result) == [A, B]
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B])]
    rig.cache.persist.assert_not_called()
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]


def test_primary_runs_when_cache_disabled(rig, capsys):
    rig.primary.fetch_ohlcv.return_value = frame([A, B], [START, END])
    fetch(rig, cache=False)
    rig.make_cache.assert_not_called()
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B])]
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]


def test_primary_requests_all_tickers_including_skipped_names(rig):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.neg.load_skip.return_value = {B}
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    fetch(rig, symbols=(C, B, A), negative=True)
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B, C]), request([C])]


def test_primary_empty_result_raises_value_error_and_stops(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="yfinance returned no data for the requested universe"):
        fetch(rig, negative=True)
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B])]
    rig.make_tiingo.assert_not_called()
    rig.neg.record.assert_not_called()
    rig.cache.persist.assert_not_called()
    assert lines(capsys) == []


def test_primary_fetch_error_propagates(rig):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.primary.fetch_ohlcv.side_effect = RuntimeError("yf failure")
    with pytest.raises(RuntimeError, match="yf failure"):
        fetch(rig)


def test_primary_constructor_error_propagates(rig):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.make_primary.side_effect = RuntimeError("no yfinance")
    with pytest.raises(RuntimeError, match="no yfinance"):
        fetch(rig)


def test_primary_instance_is_reused_for_the_yfinance_gap(rig):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    fetch(rig)
    rig.make_primary.assert_called_once_with()
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B]), request([B])]


# ---- yfinance gap -----------------------------------------------------------------------------


def test_yfinance_gap_not_run_when_nothing_missing(rig):
    fetch(rig)
    rig.make_primary.assert_not_called()


def test_yfinance_gap_announces_before_constructing(capsys):
    cache = Mock()
    cache.fetch_ohlcv.return_value = frame([A], [START])
    yf = Mock()
    yf.fetch_ohlcv.return_value = frame([B], [END])

    def make_yf():
        print("MAKE yfinance")
        return yf

    chain = OhlcvFetchChain(primary=make_yf, fillers=[yf_gap(make_yf)], cache=lambda: cache)
    chain.fetch([B, A], START, END)
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "MAKE yfinance",
        SUMMARY.format(rows=2, n=2),
    ]


def test_yfinance_gap_constructs_when_primary_did_not_run(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.return_value = frame([B], [END])
    fetch(rig)
    rig.make_primary.assert_called_once_with()


def test_yfinance_gap_requests_missing_only_and_skips_skipped(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.neg.load_skip.return_value = {B}
    rig.primary.fetch_ohlcv.return_value = pd.DataFrame()
    fetch(rig, symbols=(C, B, A), negative=True)
    assert rig.primary.fetch_ohlcv.call_args_list == [request([C])]


def test_yfinance_gap_errors_propagate(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.side_effect = RuntimeError("yf failure")
    with pytest.raises(RuntimeError, match="yf failure"):
        fetch(rig)
    rig.make_tiingo.assert_not_called()


def test_yfinance_gap_constructor_error_propagates(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.make_primary.side_effect = RuntimeError("no yfinance")
    with pytest.raises(RuntimeError, match="no yfinance"):
        fetch(rig)


def test_yfinance_gap_persists_through_cache_then_merges(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    extra = frame([B], [END])
    rig.primary.fetch_ohlcv.return_value = extra
    result = fetch(rig)
    assert symbols_of(result) == [A, B]
    assert [c.args[1] for c in rig.cache.persist.call_args_list] == ["1d"]
    pd.testing.assert_frame_equal(rig.cache.persist.call_args.args[0], extra)
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        SUMMARY.format(rows=2, n=2),
    ]


def test_yfinance_gap_empty_result_is_not_persisted(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    fetch(rig)
    rig.cache.persist.assert_not_called()


def test_yfinance_gap_persist_failure_is_reported_and_merge_continues(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.return_value = frame([B], [END])
    rig.cache.persist.side_effect = RuntimeError("write")
    assert symbols_of(fetch(rig)) == [A, B]
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] failed to persist gap fetch: RuntimeError('write')",
        SUMMARY.format(rows=2, n=2),
    ]


def test_yfinance_gap_without_writer_is_not_persisted(rig):
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    result = fetch(rig, cache=False)
    assert symbols_of(result) == [A, B]
    rig.cache.persist.assert_not_called()


def test_gap_merge_concats_then_sorts_columns_in_place(rig):
    cache = frame([B], [START, END])
    gap = frame([A], [START, END])
    rig.cache.fetch_ohlcv.return_value = cache
    rig.primary.fetch_ohlcv.return_value = gap
    result = fetch(rig)
    expected = pd.concat([cache, gap], axis=1).sort_index(axis=1)
    expected.columns.names = ["symbol", "field"]
    pd.testing.assert_frame_equal(result, expected)


# ---- Tiingo gap -------------------------------------------------------------------------------


def test_tiingo_gap_not_run_when_nothing_still_missing(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.return_value = frame([B], [END])
    fetch(rig)
    rig.make_tiingo.assert_not_called()


def test_tiingo_gap_announces_after_constructing(capsys):
    cache = Mock()
    cache.fetch_ohlcv.return_value = frame([A], [START])
    yf = Mock()
    yf.fetch_ohlcv.return_value = pd.DataFrame()
    tiingo = Mock()
    tiingo.fetch_ohlcv.return_value = frame([B], [END])

    def make_tiingo():
        print("MAKE tiingo")
        return tiingo

    chain = OhlcvFetchChain(
        primary=lambda: yf,
        fillers=[yf_gap(lambda: yf), tiingo_gap(make_tiingo)],
        cache=lambda: cache,
    )
    chain.fetch([B, A], START, END)
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "MAKE tiingo",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        SUMMARY.format(rows=2, n=2),
    ]


def test_tiingo_gap_requests_still_missing_only(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.neg.load_skip.return_value = {B}
    rig.primary.fetch_ohlcv.return_value = pd.DataFrame()
    fetch(rig, symbols=(C, B, A), negative=True)
    assert rig.primary.fetch_ohlcv.call_args_list == [request([C])]
    assert rig.tiingo.fetch_ohlcv.call_args_list == [request([C])]


def test_tiingo_gap_persists_then_merges(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    extra = frame([B], [END])
    rig.tiingo.fetch_ohlcv.return_value = extra
    result = fetch(rig)
    assert symbols_of(result) == [A, B]
    assert [c.args[1] for c in rig.cache.persist.call_args_list] == ["1d"]
    pd.testing.assert_frame_equal(rig.cache.persist.call_args.args[0], extra)
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        SUMMARY.format(rows=2, n=2),
    ]


def test_tiingo_gap_persist_failure_is_reported_and_merge_continues(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    rig.cache.persist.side_effect = RuntimeError("write")
    assert symbols_of(fetch(rig)) == [A, B]
    assert "  [data] failed to persist Tiingo data: RuntimeError('write')" in lines(capsys)


def test_tiingo_gap_without_writer_is_not_persisted(rig):
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    rig.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    assert symbols_of(fetch(rig, cache=False)) == [A, B]
    rig.cache.persist.assert_not_called()


def test_tiingo_gap_fetch_error_is_reported_not_raised(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.tiingo.fetch_ohlcv.side_effect = RuntimeError("tiingo failure")
    assert symbols_of(fetch(rig)) == [A]
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        "  [data] Tiingo fallback unavailable: RuntimeError('tiingo failure')",
        SUMMARY.format(rows=1, n=1),
    ]


def test_tiingo_gap_constructor_error_is_reported_before_the_announcement(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.make_tiingo.side_effect = RuntimeError("offline")
    assert symbols_of(fetch(rig)) == [A]
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] Tiingo fallback unavailable: RuntimeError('offline')",
        SUMMARY.format(rows=1, n=1),
    ]


def test_tiingo_gap_concat_error_is_inside_the_guard(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.tiingo.fetch_ohlcv.return_value = SimpleNamespace(empty=False)
    assert symbols_of(fetch(rig)) == [A]
    out = lines(capsys)
    assert out[2].startswith("  [data] Tiingo fallback unavailable: TypeError(")
    rig.cache.persist.assert_called_once()


# ---- the cache route --------------------------------------------------------------------------


def test_cache_empty_keeps_writer(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    extra = frame([B], [END])
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), extra]
    fetch(rig)
    assert rig.cache.persist.call_count == 1
    pd.testing.assert_frame_equal(rig.cache.persist.call_args.args[0], extra)
    assert "DB cache unavailable" not in "".join(lines(capsys))


def test_cache_constructor_error_drops_writer(rig, capsys):
    rig.make_cache.side_effect = RuntimeError("offline")
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    result = fetch(rig)
    assert symbols_of(result) == [A, B]
    assert rig.primary.fetch_ohlcv.call_args_list == [request([A, B]), request([B])]
    rig.cache.persist.assert_not_called()
    assert lines(capsys) == [
        "  [data] DB cache unavailable (RuntimeError('offline')); falling back to yfinance only",
        "  [data] fetching 1 symbols missing from cache...",
        SUMMARY.format(rows=2, n=2),
    ]


def test_cache_fetch_error_drops_writer(rig, capsys):
    rig.cache.fetch_ohlcv.side_effect = RuntimeError("down")
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    fetch(rig)
    rig.cache.persist.assert_not_called()
    assert lines(capsys)[0] == (
        "  [data] DB cache unavailable (RuntimeError('down')); falling back to yfinance only"
    )


def test_factory_raising_import_error_is_cache_unavailable(rig, capsys):
    rig.make_cache.side_effect = ImportError("no psycopg2")
    rig.primary.fetch_ohlcv.return_value = frame([A, B], [START, END])
    assert symbols_of(fetch(rig)) == [A, B]
    assert lines(capsys)[0] == (
        "  [data] DB cache unavailable (ImportError('no psycopg2')); falling back to yfinance only"
    )


# ---- the negative cache -----------------------------------------------------------------------


def test_negative_cache_is_loaded_before_anything_is_fetched(rig, capsys):
    order: list[str] = []
    rig.neg.load_skip.side_effect = lambda interval: order.append("load") or {B}
    rig.make_cache.side_effect = lambda: order.append("cache") or rig.cache
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    fetch(rig, negative=True)
    assert order == ["load", "cache"]
    rig.neg.load_skip.assert_called_once_with("1d")
    assert lines(capsys)[0] == "  [data] negative cache: skipping 1 known no-data symbols"


def test_negative_cache_empty_skip_set_is_silent(rig, capsys):
    fetch(rig, negative=True)
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]


def test_negative_cache_load_failure_fetches_all(rig, capsys):
    rig.neg.load_skip.side_effect = RuntimeError("read")
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    fetch(rig, negative=True)
    assert (
        lines(capsys)[0]
        == "  [data] negative cache unavailable (RuntimeError('read')); fetching all"
    )
    assert rig.primary.fetch_ohlcv.call_args == request([B])
    assert rig.tiingo.fetch_ohlcv.call_args == request([B])


def test_negative_cache_records_symbols_no_provider_supplied(rig, capsys):
    rig.neg.load_skip.return_value = {B}
    rig.cache.fetch_ohlcv.return_value = pd.DataFrame()
    rig.primary.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    result = fetch(rig, symbols=(C, B, A), negative=True)
    assert symbols_of(result) == [A]
    rig.neg.record.assert_called_once_with([C], "1d")
    assert lines(capsys) == [
        "  [data] negative cache: skipping 1 known no-data symbols",
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        "  [data] negative cache: recorded 1 no-data symbols",
        SUMMARY.format(rows=1, n=1),
    ]


def test_negative_cache_records_after_the_gaps_run(rig):
    order: list[str] = []
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.tiingo.fetch_ohlcv.side_effect = lambda *a, **k: order.append("tiingo") or pd.DataFrame()
    rig.neg.record.side_effect = lambda *a: order.append("record")
    fetch(rig, negative=True)
    assert order == ["tiingo", "record"]


def test_negative_cache_nothing_to_record_when_all_supplied(rig):
    fetch(rig, negative=True)
    rig.neg.record.assert_not_called()


def test_negative_cache_write_failure_continues(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.neg.record.side_effect = RuntimeError("write")
    result = fetch(rig, negative=True)
    assert symbols_of(result) == [A]
    assert "  [data] negative cache write failed (RuntimeError('write'))" in lines(capsys)


def test_negative_cache_is_optional_and_never_touched_when_absent(rig, monkeypatch):
    exploding = Mock(side_effect=AssertionError("negative cache touched"))
    monkeypatch.setitem(sys.modules, "ggTrader.lab.negative_cache", exploding)
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    assert symbols_of(fetch(rig, negative=False)) == [A]
    rig.neg.load_skip.assert_not_called()
    rig.neg.record.assert_not_called()


# ---- the closing steps ------------------------------------------------------------------------


def test_window_is_trimmed_to_start_and_end(rig, capsys):
    rig.cache.fetch_ohlcv.return_value = frame([A, B], ["2023-12-31", START, END, "2024-01-04"])
    result = fetch(rig)
    assert list(result.index) == list(pd.to_datetime([START, END], utc=True))
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]


def test_no_end_passes_none_and_keeps_later_rows(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START, "2025-01-01"])
    rig.primary.fetch_ohlcv.return_value = frame([B], ["2025-01-01"])
    result = chain_of(rig).fetch([A, B], START)
    assert list(result.index) == list(pd.to_datetime([START, "2025-01-01"], utc=True))
    assert rig.cache.fetch_ohlcv.call_args.kwargs["end_date"] is None
    assert rig.primary.fetch_ohlcv.call_args.kwargs["end_date"] is None


def test_min_coverage_drops_sparse_symbols(rig, capsys):
    data = frame([A, B], [START, END])
    data.loc[END_TS, (B, "close")] = float("nan")
    rig.cache.fetch_ohlcv.return_value = data
    result = fetch(rig, min_coverage=0.75)
    assert symbols_of(result) == [A]
    assert lines(capsys) == [SUMMARY.format(rows=2, n=1)]


def test_result_columns_are_named_symbol_and_field(rig):
    assert list(fetch(rig).columns.names) == ["symbol", "field"]


def test_tickers_are_normalised_and_deduplicated_for_every_provider(rig):
    rig.cache.fetch_ohlcv.return_value = frame(["BRK-B"], [START])
    chain_of(rig).fetch([" brk.b ", "BRK.B", " aapl ", "AAPL"], START, END)
    assert rig.cache.fetch_ohlcv.call_args == request(["AAPL", "BRK-B"], True)
    assert rig.primary.fetch_ohlcv.call_args == request(["AAPL"])
    assert rig.tiingo.fetch_ohlcv.call_args == request(["AAPL"])


def test_persisted_frames_keep_their_utc_index(rig):
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.return_value = frame([B], [END])
    fetch(rig)
    persisted = rig.cache.persist.call_args.args[0]
    assert str(persisted.index.tz) == "UTC"
    assert list(persisted.index) == [pd.Timestamp(END, tz="UTC")]


def test_chain_is_reusable_between_fetches(rig):
    chain = chain_of(rig)
    rig.cache.fetch_ohlcv.return_value = frame([A], [START])
    rig.primary.fetch_ohlcv.return_value = frame([B], [END])
    first = chain.fetch([A, B], START, END)
    second = chain.fetch([A, B], START, END)
    pd.testing.assert_frame_equal(first, second)


# ---- the module's own rules -------------------------------------------------------------------


def test_chain_has_no_lab_or_paper_import():
    tree = ast.parse(Path(fetch_chain.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported += [f"{module}.{alias.name}" for alias in node.names]
    banned = [
        name
        for name in imported
        if any(part in {"lab", "paper", "cli"} for part in name.split("."))
    ]
    assert banned == []


# ---- the default chain ------------------------------------------------------------------------


@pytest.fixture
def default_seams(monkeypatch):
    """Patch the three loader classes on their modules, as the characterization does."""
    from ggTrader.data.live import cached_yfinance_loader, tiingo_loader, yfinance_loader

    s = SimpleNamespace(cache=Mock(), yf=Mock(), tiingo=Mock())
    s.cache.fetch_ohlcv.return_value = frame([A], [START])
    s.yf.fetch_ohlcv.return_value = pd.DataFrame()
    s.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    s.cache_cls = Mock(return_value=s.cache)
    s.yf_cls = Mock(return_value=s.yf)
    s.tiingo_cls = Mock(return_value=s.tiingo)
    monkeypatch.setattr(cached_yfinance_loader, "CachedYFinanceLoader", s.cache_cls)
    monkeypatch.setattr(yfinance_loader, "YFinanceDataLoader", s.yf_cls)
    monkeypatch.setattr(tiingo_loader, "TiingoDataLoader", s.tiingo_cls)
    return s


def test_default_factories_resolve_classes_at_call_time(monkeypatch):
    from ggTrader.data.live import cached_yfinance_loader, tiingo_loader, yfinance_loader

    chain = fetch_chain.default_chain()
    s = SimpleNamespace(cache=Mock(), yf=Mock(), tiingo=Mock())
    s.cache.fetch_ohlcv.return_value = frame([A], [START])
    s.yf.fetch_ohlcv.return_value = pd.DataFrame()
    s.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    cache_cls, yf_cls, tiingo_cls = (Mock(return_value=x) for x in (s.cache, s.yf, s.tiingo))
    monkeypatch.setattr(cached_yfinance_loader, "CachedYFinanceLoader", cache_cls)
    monkeypatch.setattr(yfinance_loader, "YFinanceDataLoader", yf_cls)
    monkeypatch.setattr(tiingo_loader, "TiingoDataLoader", tiingo_cls)
    result = chain.fetch([B, A], START, END)
    assert symbols_of(result) == [A, B]
    cache_cls.assert_called_once_with()
    yf_cls.assert_called_once_with()
    tiingo_cls.assert_called_once_with()


def test_fetch_stock_ohlcv_runs_the_default_chain_in_order(default_seams, capsys):
    s = default_seams
    result = fetch_stock_ohlcv([B, A], START, END)
    assert symbols_of(result) == [A, B]
    assert s.cache.fetch_ohlcv.call_args_list == [request([A, B], True)]
    assert s.yf.fetch_ohlcv.call_args_list == [request([B])]
    assert s.tiingo.fetch_ohlcv.call_args_list == [request([B])]
    assert s.cache.persist.call_count == 1
    s.yf_cls.assert_called_once_with()
    assert lines(capsys) == [
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        SUMMARY.format(rows=2, n=2),
    ]


def test_fetch_stock_ohlcv_without_db_cache_never_builds_the_cache(default_seams):
    s = default_seams
    s.yf.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    fetch_stock_ohlcv([B, A], START, END, use_db_cache=False)
    s.cache_cls.assert_not_called()
    s.cache.persist.assert_not_called()
    assert s.yf.fetch_ohlcv.call_args_list == [request([A, B]), request([B])]


def test_fetch_stock_ohlcv_passes_the_negative_cache_through(default_seams):
    neg = Mock()
    neg.load_skip.return_value = {B}
    fetch_stock_ohlcv([B, A], START, END, negative_cache=neg)
    neg.load_skip.assert_called_once_with("1d")
    default_seams.yf.fetch_ohlcv.assert_not_called()
    default_seams.tiingo.fetch_ohlcv.assert_not_called()


def test_fetch_stock_ohlcv_passes_min_coverage_and_interval(default_seams, capsys):
    default_seams.cache.fetch_ohlcv.return_value = frame([A, B], [START, END])
    result = fetch_stock_ohlcv([B, A], START, END, interval="1d", min_coverage=0.5)
    assert symbols_of(result) == [A, B]
    assert lines(capsys) == [SUMMARY.format(rows=2, n=2)]
