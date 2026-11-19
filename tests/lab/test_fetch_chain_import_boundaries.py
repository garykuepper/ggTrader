"""The fetch chain fails on a missing loader module, class or function as the originals did."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from ggTrader.data.live import cached_yfinance_loader, tiingo_loader, yfinance_loader
from ggTrader.data.live.fetch_chain import fetch_stock_ohlcv as chain_fetch
from ggTrader.lab import negative_cache
from ggTrader.lab.data import fetch_stock_ohlcv as lab_fetch
from tests.netguard import blocked_connections

A, B = "ZZTEST1", "ZZTEST2"
START, END = "2024-01-01", "2024-01-03"
CACHE_MODULE = cached_yfinance_loader.__name__
YF_MODULE = yfinance_loader.__name__
TIINGO_MODULE = tiingo_loader.__name__
NEGATIVE_MODULE = negative_cache.__name__


def frame(symbols: list[str]) -> pd.DataFrame:
    """Build a provider frame with a close column for each symbol."""
    return pd.DataFrame(
        1.0,
        index=pd.to_datetime([START, END], utc=True),
        columns=pd.MultiIndex.from_product([symbols, ["close"]]),
    )


def original_failure(module: str, name: str) -> str:
    """The repr of the error the original ``from module import name`` raises in this state."""
    try:
        exec(f"from {module} import {name}")
    except Exception as exc:
        return repr(exc)
    raise AssertionError(f"from {module} import {name} did not fail")


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch):
    """Fake loaders and negative cache; the cache holds A and B, so a complete run needs nothing."""
    s = SimpleNamespace(
        cache=Mock(), yf=Mock(), tiingo=Mock(), skip=Mock(return_value=set()), record=Mock()
    )
    s.cache.fetch_ohlcv.return_value = frame([A, B])
    s.yf.fetch_ohlcv.return_value = frame([A, B])
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


def break_class(monkeypatch: pytest.MonkeyPatch, module, name: str) -> None:
    """Remove a class from its module, so ``from module import name`` raises ImportError."""
    monkeypatch.delattr(module, name)


def break_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``from <name> import x`` raise ModuleNotFoundError even though the package has it."""
    monkeypatch.setitem(sys.modules, name, None)


def test_cache_class_missing_reports_the_import_error(seams, monkeypatch, capsys):
    break_class(monkeypatch, cached_yfinance_loader, "CachedYFinanceLoader")
    expected = original_failure(CACHE_MODULE, "CachedYFinanceLoader")
    assert expected.startswith("ImportError(")

    result = chain_fetch([A, B], START, END)

    assert list(result.columns.get_level_values(0).unique()) == [A, B]
    assert capsys.readouterr().out.splitlines()[0] == (
        f"  [data] DB cache unavailable ({expected}); falling back to yfinance only"
    )
    assert seams.yf.fetch_ohlcv.call_count == 1


def test_cache_module_missing_reports_the_module_error(seams, monkeypatch, capsys):
    break_module(monkeypatch, CACHE_MODULE)
    expected = original_failure(CACHE_MODULE, "CachedYFinanceLoader")
    assert expected.startswith("ModuleNotFoundError(")

    chain_fetch([A, B], START, END)

    assert capsys.readouterr().out.splitlines()[0] == (
        f"  [data] DB cache unavailable ({expected}); falling back to yfinance only"
    )
    assert seams.cache_cls.call_count == 0


def test_tiingo_class_missing_reports_the_import_error(seams, monkeypatch, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A])
    seams.yf.fetch_ohlcv.return_value = pd.DataFrame()
    break_class(monkeypatch, tiingo_loader, "TiingoDataLoader")
    expected = original_failure(TIINGO_MODULE, "TiingoDataLoader")
    assert expected.startswith("ImportError(")

    result = chain_fetch([A, B], START, END)

    assert list(result.columns.get_level_values(0).unique()) == [A]
    lines = capsys.readouterr().out.splitlines()
    assert f"  [data] Tiingo fallback unavailable: {expected}" in lines
    assert not any("trying Tiingo" in line for line in lines)


def test_tiingo_module_missing_reports_the_module_error(seams, monkeypatch, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A])
    seams.yf.fetch_ohlcv.return_value = pd.DataFrame()
    break_module(monkeypatch, TIINGO_MODULE)
    expected = original_failure(TIINGO_MODULE, "TiingoDataLoader")
    assert expected.startswith("ModuleNotFoundError(")

    chain_fetch([A, B], START, END)

    assert (
        f"  [data] Tiingo fallback unavailable: {expected}" in capsys.readouterr().out.splitlines()
    )


def test_negative_cache_read_function_missing_reports_the_import_error(seams, monkeypatch, capsys):
    break_class(monkeypatch, negative_cache, "load_skip_symbols")
    expected = original_failure(NEGATIVE_MODULE, "load_skip_symbols")
    assert expected.startswith("ImportError(")

    lab_fetch([A, B], START, END, use_negative_cache=True)

    assert capsys.readouterr().out.splitlines()[0] == (
        f"  [data] negative cache unavailable ({expected}); fetching all"
    )


def test_negative_cache_write_function_missing_reports_the_import_error(seams, monkeypatch, capsys):
    seams.cache.fetch_ohlcv.return_value = frame([A])
    seams.yf.fetch_ohlcv.return_value = pd.DataFrame()
    break_class(monkeypatch, negative_cache, "record_no_data")
    expected = original_failure(NEGATIVE_MODULE, "record_no_data")
    assert expected.startswith("ImportError(")

    lab_fetch([A, B], START, END, use_negative_cache=True)

    assert (
        f"  [data] negative cache write failed ({expected})" in capsys.readouterr().out.splitlines()
    )


def test_negative_cache_module_missing_reports_the_module_error(seams, monkeypatch, capsys):
    break_module(monkeypatch, NEGATIVE_MODULE)
    expected = original_failure(NEGATIVE_MODULE, "load_skip_symbols")
    assert expected.startswith("ModuleNotFoundError(")

    lab_fetch([A, B], START, END, use_negative_cache=True)

    assert capsys.readouterr().out.splitlines()[0] == (
        f"  [data] negative cache unavailable ({expected}); fetching all"
    )


def test_yfinance_class_missing_stops_before_the_cache_is_tried(seams, monkeypatch):
    break_class(monkeypatch, yfinance_loader, "YFinanceDataLoader")
    expected = original_failure(YF_MODULE, "YFinanceDataLoader")

    with pytest.raises(ImportError) as raised:
        chain_fetch([A, B], START, END)

    assert repr(raised.value) == expected
    assert seams.cache_cls.call_count == 0
    assert seams.cache.fetch_ohlcv.call_count == 0


def test_yfinance_module_missing_stops_before_the_cache_is_tried(seams, monkeypatch):
    break_module(monkeypatch, YF_MODULE)
    expected = original_failure(YF_MODULE, "YFinanceDataLoader")

    with pytest.raises(ModuleNotFoundError) as raised:
        chain_fetch([A, B], START, END)

    assert repr(raised.value) == expected
    assert seams.cache_cls.call_count == 0


def test_yfinance_class_is_resolved_without_constructing_it(seams):
    chain_fetch([A, B], START, END)

    assert seams.yf_cls.call_count == 0
    assert seams.cache_cls.call_count == 1


def test_loaders_patched_after_import_are_still_used(seams, monkeypatch):
    late = Mock()
    late.fetch_ohlcv.return_value = frame([A, B])
    monkeypatch.setattr(cached_yfinance_loader, "CachedYFinanceLoader", Mock(return_value=late))

    chain_fetch([A, B], START, END)

    assert late.fetch_ohlcv.call_count == 1
    assert seams.cache.fetch_ohlcv.call_count == 0
