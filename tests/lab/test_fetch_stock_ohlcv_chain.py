"""Characterize the stock price provider fallback chain without connections."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from ggTrader.lab.data import fetch_stock_ohlcv
from tests.netguard import blocked_connections

A, B, C = "ZZTEST1", "ZZTEST2", "ZZTEST3"
START, END = "2024-01-01", "2024-01-03"


def frame(symbols: list[str], days: list[str]) -> pd.DataFrame:
    """Build a provider frame with a close column for each symbol."""
    return pd.DataFrame(
        1.0,
        index=pd.to_datetime(days, utc=True),
        columns=pd.MultiIndex.from_product([symbols, ["close"]]),
    )


@pytest.fixture(autouse=True)
def seams(monkeypatch: pytest.MonkeyPatch):
    """Replace every external seam and fail on connection attempts."""
    from ggTrader.data.live import cached_yfinance_loader, tiingo_loader, yfinance_loader
    from ggTrader.lab import negative_cache

    s = SimpleNamespace(
        cache=Mock(), yf=Mock(), tiingo=Mock(), skip=Mock(return_value=set()), record=Mock()
    )
    s.cache.fetch_ohlcv.return_value = frame([A, B], [START, END])
    s.yf.fetch_ohlcv.return_value = pd.DataFrame()
    s.tiingo.fetch_ohlcv.return_value = pd.DataFrame()
    s.cache_cls = Mock(return_value=s.cache)
    s.yf_cls = Mock(return_value=s.yf)
    s.tiingo_cls = Mock(return_value=s.tiingo)
    monkeypatch.setattr(cached_yfinance_loader, "CachedYFinanceLoader", s.cache_cls)
    monkeypatch.setattr(yfinance_loader, "YFinanceDataLoader", s.yf_cls)
    monkeypatch.setattr(tiingo_loader, "TiingoDataLoader", s.tiingo_cls)
    monkeypatch.setattr(negative_cache, "load_skip_symbols", s.skip)
    monkeypatch.setattr(negative_cache, "record_no_data", s.record)
    with blocked_connections():
        yield s


def request(symbols: list[str], cache: bool = False) -> tuple:
    """Expected provider call, including the cache-only limit."""
    kwargs = {"start_date": pd.Timestamp(START, tz="UTC"), "end_date": pd.Timestamp(END, tz="UTC")}
    if cache:
        kwargs["limit"] = None
    return (symbols, "1d"), kwargs


def check(s, capsys, result, symbols, days, yf, tiingo, persisted, lines, cache=True):
    """Check data, every provider request, persistence and output."""
    assert list(result.columns.get_level_values(0).unique()) == symbols
    assert list(result.index) == list(pd.to_datetime(days, utc=True))
    assert s.cache.fetch_ohlcv.call_args_list == ([request([A, B], True)] if cache else [])
    assert s.yf.fetch_ohlcv.call_args_list == [request(x) for x in yf]
    assert s.tiingo.fetch_ohlcv.call_args_list == [request(x) for x in tiingo]
    assert s.cache.persist.call_count == len(persisted)
    for call, expected in zip(s.cache.persist.call_args_list, persisted):
        pd.testing.assert_frame_equal(call.args[0], expected)
        assert call.args[1:] == ("1d",)
    assert capsys.readouterr().out.splitlines() == lines


def run(**kwargs):
    """Fetch the standard two-symbol universe."""
    return fetch_stock_ohlcv([B, A], START, END, **kwargs)


def test_cache_hit_returns_cache_only(seams, capsys):
    result = run()
    check(
        seams,
        capsys,
        result,
        [A, B],
        [START, END],
        [],
        [],
        [],
        ["  [data] 2 rows x 2 symbols (1d)"],
    )


def test_cache_constructor_fails_falls_back_to_yfinance_without_writer(seams, capsys):
    seams.cache_cls.side_effect = RuntimeError("offline")
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    result = run()
    check(
        seams,
        capsys,
        result,
        [A, B],
        [START, END],
        [[A, B], [B]],
        [],
        [],
        [
            "  [data] DB cache unavailable (RuntimeError('offline')); "
            "falling back to yfinance only",
            "  [data] fetching 1 symbols missing from cache...",
            "  [data] 2 rows x 2 symbols (1d)",
        ],
        cache=False,
    )


def test_cache_fetch_raises_falls_back_and_clears_writer(seams, capsys):
    """The cache constructs but its fetch raises: the writer is dropped, so no gap is saved.

    Unlike the constructor-fails case, the fake cache here is live, so a gap fill that wrongly
    reached `persist` would be recorded.
    """
    seams.cache.fetch_ohlcv.side_effect = RuntimeError("down")
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    result = run()
    check(
        seams,
        capsys,
        result,
        [A, B],
        [START, END],
        [[A, B], [B]],
        [],
        [],
        [
            "  [data] DB cache unavailable (RuntimeError('down')); falling back to yfinance only",
            "  [data] fetching 1 symbols missing from cache...",
            "  [data] 2 rows x 2 symbols (1d)",
        ],
    )


def test_cache_empty_falls_back_to_yfinance_and_keeps_writer(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = pd.DataFrame()
    extra = frame([B], [END])
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), extra]
    result = run()
    check(
        seams,
        capsys,
        result,
        [A, B],
        [START, END],
        [[A, B], [B]],
        [],
        [extra],
        ["  [data] fetching 1 symbols missing from cache...", "  [data] 2 rows x 2 symbols (1d)"],
    )


def test_yfinance_empty_raises_value_error(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="yfinance returned no data"):
        run()
    assert seams.cache.fetch_ohlcv.call_args_list == [request([A, B], True)]
    assert seams.yf.fetch_ohlcv.call_args_list == [request([A, B])]
    seams.tiingo.fetch_ohlcv.assert_not_called()
    seams.cache.persist.assert_not_called()
    assert capsys.readouterr().out == ""


def test_gap_filled_by_yfinance_is_persisted(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    extra = frame([B], [END])
    seams.yf.fetch_ohlcv.return_value = extra
    check(
        seams,
        capsys,
        run(),
        [A, B],
        [START, END],
        [[B]],
        [],
        [extra],
        ["  [data] fetching 1 symbols missing from cache...", "  [data] 2 rows x 2 symbols (1d)"],
    )


def test_gap_persist_failure_is_reported_and_ignored(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    extra = frame([B], [END])
    seams.yf.fetch_ohlcv.return_value = extra
    seams.cache.persist.side_effect = RuntimeError("write")
    check(
        seams,
        capsys,
        run(),
        [A, B],
        [START, END],
        [[B]],
        [],
        [extra],
        [
            "  [data] fetching 1 symbols missing from cache...",
            "  [data] failed to persist gap fetch: RuntimeError('write')",
            "  [data] 2 rows x 2 symbols (1d)",
        ],
    )


def test_gap_filled_by_tiingo_is_persisted(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    extra = frame([B], [END])
    seams.tiingo.fetch_ohlcv.return_value = extra
    check(
        seams,
        capsys,
        run(),
        [A, B],
        [START, END],
        [[B]],
        [[B]],
        [extra],
        [
            "  [data] fetching 1 symbols missing from cache...",
            "  [data] trying Tiingo for 1 symbols yfinance missed...",
            "  [data] 2 rows x 2 symbols (1d)",
        ],
    )


def test_tiingo_unavailable_is_reported(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    seams.tiingo_cls.side_effect = RuntimeError("offline")
    check(
        seams,
        capsys,
        run(),
        [A],
        [START],
        [[B]],
        [],
        [],
        [
            "  [data] fetching 1 symbols missing from cache...",
            "  [data] Tiingo fallback unavailable: RuntimeError('offline')",
            "  [data] 1 rows x 1 symbols (1d)",
        ],
    )


def test_negative_cache_skips_and_records(seams, capsys):
    seams.skip.return_value = {B}
    seams.cache.fetch_ohlcv.return_value = pd.DataFrame()
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    result = fetch_stock_ohlcv([C, B, A], START, END, use_negative_cache=True)
    assert seams.cache.fetch_ohlcv.call_args_list == [request([A, B, C], True)]
    assert seams.yf.fetch_ohlcv.call_args_list == [request([A, B, C]), request([C])]
    assert seams.tiingo.fetch_ohlcv.call_args_list == [request([C])]
    seams.skip.assert_called_once_with("1d")
    seams.record.assert_called_once_with([C], "1d")
    seams.cache.persist.assert_not_called()
    assert list(result.columns.get_level_values(0).unique()) == [A]
    assert list(result.index) == list(pd.to_datetime([START], utc=True))
    assert capsys.readouterr().out.splitlines() == [
        "  [data] negative cache: skipping 1 known no-data symbols",
        "  [data] fetching 1 symbols missing from cache...",
        "  [data] trying Tiingo for 1 symbols yfinance missed...",
        "  [data] negative cache: recorded 1 no-data symbols",
        "  [data] 1 rows x 1 symbols (1d)",
    ]


def test_min_coverage_drops_sparse_symbols(seams, capsys):
    data = frame([A, B], [START, END])
    data.loc[pd.Timestamp(END, tz="UTC"), (B, "close")] = float("nan")
    seams.cache.fetch_ohlcv.return_value = data
    check(
        seams,
        capsys,
        run(min_coverage=0.75),
        [A],
        [START, END],
        [],
        [],
        [],
        ["  [data] 2 rows x 1 symbols (1d)"],
    )


def test_window_trimmed_to_start_and_end(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A, B], ["2023-12-31", START, END, "2024-01-04"])
    check(
        seams, capsys, run(), [A, B], [START, END], [], [], [], ["  [data] 2 rows x 2 symbols (1d)"]
    )


def rich_frame(symbol, days, base):
    """Build distinguishable OHLCV values for every field and row."""
    fields = ["open", "high", "low", "close", "volume"]
    return pd.DataFrame(
        [[base + row * 10 + col for col in range(5)] for row in range(len(days))],
        index=pd.to_datetime(days, utc=True),
        columns=pd.MultiIndex.from_product([[symbol], fields]),
        dtype=float,
    )


@pytest.mark.parametrize("source", ["yf", "tiingo"])
def test_full_frame_cache_and_gap_values(seams, source):
    cache = rich_frame(A, [START, END], 10)
    gap = rich_frame(B, [START, END], 100 if source == "yf" else 200)
    seams.cache.fetch_ohlcv.return_value = cache
    getattr(seams, source).fetch_ohlcv.return_value = gap
    result = run()
    expected = pd.concat([cache, gap], axis=1).sort_index(axis=1)
    expected.columns.names = ["symbol", "field"]
    pd.testing.assert_frame_equal(result, expected)
    assert result.columns.names == ["symbol", "field"]


def test_no_end_passes_none_and_keeps_later_rows(seams):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START, "2025-01-01"])
    seams.yf.fetch_ohlcv.return_value = frame([B], ["2025-01-01"])
    result = fetch_stock_ohlcv([A, B], START)
    assert list(result.index) == list(pd.to_datetime([START, "2025-01-01"], utc=True))
    assert seams.cache.fetch_ohlcv.call_args.kwargs["end_date"] is None
    assert seams.yf.fetch_ohlcv.call_args.kwargs["end_date"] is None


def test_cache_disabled_uses_plain_without_persisting(seams):
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), frame([B], [END])]
    run(use_db_cache=False)
    seams.cache_cls.assert_not_called()
    assert seams.yf.fetch_ohlcv.call_args_list == [request([A, B]), request([B])]
    seams.cache.persist.assert_not_called()


def test_normalizes_and_deduplicates_for_all_providers(seams):
    seams.cache.fetch_ohlcv.return_value = frame(["BRK-B"], [START])
    fetch_stock_ohlcv([" brk.b ", "BRK.B", " aapl ", "AAPL"], START, END)
    assert seams.cache.fetch_ohlcv.call_args == request(["AAPL", "BRK-B"], True)
    assert seams.yf.fetch_ohlcv.call_args == request(["AAPL"])
    assert seams.tiingo.fetch_ohlcv.call_args == request(["AAPL"])


def test_negative_cache_default_never_reads_or_writes(seams):
    seams.skip.return_value = {B}
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    run()
    seams.skip.assert_not_called()
    seams.record.assert_not_called()
    seams.yf.fetch_ohlcv.assert_called_once_with(*request([B])[0], **request([B])[1])


def test_negative_cache_read_failure_fetches_all(seams, capsys):
    seams.skip.side_effect = RuntimeError("read")
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    run(use_negative_cache=True)
    assert "negative cache unavailable (RuntimeError('read')); fetching all" in (
        capsys.readouterr().out
    )
    assert seams.yf.fetch_ohlcv.call_args == request([B])
    assert seams.tiingo.fetch_ohlcv.call_args == request([B])


def test_negative_cache_write_failure_continues(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    seams.record.side_effect = RuntimeError("write")
    result = run(use_negative_cache=True)
    assert list(result.columns.get_level_values(0).unique()) == [A]
    assert "negative cache write failed (RuntimeError('write'))" in capsys.readouterr().out


@pytest.mark.parametrize("empty_cache", [True, False])
def test_yfinance_fetch_errors_propagate(seams, empty_cache):
    seams.cache.fetch_ohlcv.return_value = pd.DataFrame() if empty_cache else frame([A], [START])
    seams.yf.fetch_ohlcv.side_effect = RuntimeError("yf failure")
    with pytest.raises(RuntimeError, match="yf failure"):
        run()


def test_tiingo_fetch_error_is_caught(seams, capsys):
    """Observed behavior: Tiingo fetch errors are caught, unlike yfinance errors."""
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    seams.tiingo.fetch_ohlcv.side_effect = RuntimeError("tiingo failure")
    assert list(run().columns.get_level_values(0).unique()) == [A]
    assert "Tiingo fallback unavailable: RuntimeError('tiingo failure')" in capsys.readouterr().out


def test_tiingo_persist_error_continues(seams, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A], [START])
    seams.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    seams.cache.persist.side_effect = RuntimeError("write")
    assert list(run().columns.get_level_values(0).unique()) == [A, B]
    assert "failed to persist Tiingo data: RuntimeError('write')" in capsys.readouterr().out


def test_tiingo_without_writer_is_not_persisted(seams):
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    seams.tiingo.fetch_ohlcv.return_value = frame([B], [END])
    assert list(run(use_db_cache=False).columns.get_level_values(0).unique()) == [A, B]
    seams.cache_cls.assert_not_called()
    seams.cache.persist.assert_not_called()


def test_provider_constructor_counts_and_arguments(seams):
    seams.cache.fetch_ohlcv.return_value = pd.DataFrame()
    seams.yf.fetch_ohlcv.side_effect = [frame([A], [START]), pd.DataFrame()]
    run()
    seams.cache_cls.assert_called_once_with()
    seams.yf_cls.assert_called_once_with()
    seams.tiingo_cls.assert_called_once_with()


def test_coverage_threshold_applies_after_window_trim(seams):
    data = frame([A, B], ["2023-12-31", START, END, "2024-01-04"])
    data.loc[pd.Timestamp(END, tz="UTC"), (B, "close")] = float("nan")
    data.loc[pd.Timestamp("2023-12-31", tz="UTC"), (B, "close")] = float("nan")
    data.loc[pd.Timestamp("2024-01-04", tz="UTC"), (B, "close")] = float("nan")
    seams.cache.fetch_ohlcv.return_value = data
    assert list(run(min_coverage=0.5).columns.get_level_values(0).unique()) == [A, B]


def test_gap_earlier_dates_are_sorted_in_result(seams):
    seams.cache.fetch_ohlcv.return_value = frame([A], [END])
    seams.yf.fetch_ohlcv.return_value = frame([B], [START])
    assert list(run().index) == list(pd.to_datetime([START, END], utc=True))
