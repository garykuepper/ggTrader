"""Tests for the combined release bars interface."""

import pytest

from ggTrader.data.bars import load_bars
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture


def test_release_id_with_db_source_is_rejected():
    with pytest.raises(ValueError, match="release_id"):
        load_bars(["AAPL"], source="db", release_id=1)


def test_combined_rejects_snapshot_path():
    with pytest.raises(ValueError, match="snapshot_path"):
        load_bars(["AAPL"], source="combined", snapshot_path="unused")


def test_permno_symbols_not_normalized(monkeypatch):
    import pandas as pd

    import ggTrader.data.bars as module

    def read(release_id, symbols, start, end):
        assert symbols == ("PERMNO-1~7", "PERMNO-84761")
        flat = pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )
        flat["timestamp"] = pd.to_datetime(flat.timestamp)
        flags = pd.DataFrame(columns=["source", "abs_prc", "midpoint_flag", "possible_double"])
        flags.attrs["data_release"] = {"release_id": 1, "created_at": "", "yahoo_fingerprint": {}}
        flags.attrs["drift_symbols"] = []
        return flat, flags, pd.DataFrame(), 1

    monkeypatch.setattr(module, "_read_combined_long", read, raising=False)
    bars = load_bars(["PERMNO-84761", "PERMNO-1~7"], source="combined")
    assert bars.missing_symbols == ("PERMNO-1~7", "PERMNO-84761")
    assert bars.provenance["data_release"]["release_id"] == 1


conn = db_fixture


@pytest.fixture
def published(conn, monkeypatch):
    import ggTrader.data.bars as module
    from ggTrader.data.kaggle_import.release import build_release

    db, load = conn
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    import os
    from urllib.parse import urlparse, urlunparse

    original = module.connect_readonly
    url = urlparse(os.environ["GGT_TEST_DB_URL"])
    scratch = urlunparse(url._replace(path="/" + db.info.dbname))
    monkeypatch.setattr(module, "connect_readonly", lambda: original(scratch))
    return db, load, rid


@pytest.mark.integration
@pytest.mark.db
def test_combined_returns_five_fields_and_flags_from_one_call(published, monkeypatch):
    import pandas as pd

    import ggTrader.data.kaggle_import.reader as reader
    from ggTrader.data.bars import _read_combined_long

    calls = []
    original = reader.read_combined

    def observe(conn, *args):
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_isolation")
            assert cur.fetchone()[0] == "repeatable read"
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone()[0] == "on"
        calls.append(conn)
        return original(conn, *args)

    monkeypatch.setattr(reader, "read_combined", observe)

    _, _, rid = published
    flat, flags, _, resolved = _read_combined_long(
        rid,
        ["PERMNO-2", "TTWO"],
        pd.Timestamp("1990-01-01"),
        pd.Timestamp("2010-01-01"),
        batch_size=1,
    )
    assert resolved == rid
    assert len(calls) == 2 and calls[0] is calls[1]
    assert list(flat.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert set(flags.source) == {"crsp", "yfinance"}
    assert str(flags.index.levels[0].tz) == "UTC"


@pytest.mark.integration
@pytest.mark.db
def test_default_release_is_newest_and_id_is_recorded(published):
    from ggTrader.data.kaggle_import.release import build_release, withdraw_release

    db, load, _ = published
    links, settings = _inputs()
    latest = build_release(
        db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings
    )
    withdrawn = build_release(
        db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings
    )
    withdraw_release(db, withdrawn)
    result = load_bars(["TTWO"], start="2000-01-03", source="combined")
    assert result.provenance["data_release"]["release_id"] == latest
    assert result.provenance["bars_sha256"]


@pytest.mark.integration
@pytest.mark.db
def test_withdrawn_or_unknown_release_raises_release_unavailable(published):
    from ggTrader.data.kaggle_import.release import ReleaseUnavailable, withdraw_release

    db, _, rid = published
    withdraw_release(db, rid)
    for value in (rid, -1, None):
        with pytest.raises(ReleaseUnavailable):
            load_bars(["TTWO"], source="combined", release_id=value)


@pytest.mark.integration
@pytest.mark.db
def test_spans_exclude_alive_and_file_end_series_and_mark_successors(published):
    import pandas as pd

    from ggTrader.data.bars import _read_combined_long

    db, load, rid = published
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) VALUES "
            "(%s,20,1,'1999-01-01',1),(%s,20,2,'2000-01-01',1)",
            (load, load),
        )
        cur.execute(
            "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,last_date) VALUES "
            "(%s,'OLD',10,1,'yahoo','2000-01-01'),"
            "(%s,'ALIVE',11,1,'yahoo','2000-02-01'),"
            "(%s,'PERMNO-20~1',20,1,'spell','1999-01-01'),"
            "(%s,'PERMNO-20',20,2,'placeholder','2000-01-01')",
            (rid,) * 4,
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) VALUES "
            "('2000-01-03','OLD','1d','yfinance',1),"
            "('2000-02-01','ALIVE','1d','yfinance',1)"
        )
    db.commit()
    _, _, spans, _ = _read_combined_long(
        rid,
        ["OLD", "ALIVE", "PERMNO-20~1", "PERMNO-20"],
        pd.Timestamp("1900-01-01"),
        pd.Timestamp("2200-01-01"),
    )
    by_symbol = spans.set_index("symbol")
    assert set(by_symbol.index) == {"OLD", "PERMNO-20~1"}
    assert bool(by_symbol.loc["PERMNO-20~1", "has_successor"])
    assert not by_symbol.total_loss.any()


@pytest.mark.integration
@pytest.mark.db
def test_drift_flag_reaches_provenance(published):
    db, _, rid = published
    with db.cursor() as cur:
        cur.execute(
            "UPDATE public.ohlcv SET close=99 WHERE symbol='TTWO' AND timestamp='2000-01-03'"
        )
    db.commit()
    bars = load_bars(["TTWO"], start="2000-01-03", source="combined", release_id=rid)
    assert bars.provenance["drift"]
    assert bars.provenance["drift_symbols"] == ["TTWO"]


def test_validate_long_still_enforces_weekday_and_duplicates_on_combined(monkeypatch):
    import pandas as pd

    import ggTrader.data.bars as module

    def read(*args):
        flat = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2000-01-01", "2000-01-01"]), "symbol": ["X", "X"]}
        )
        flags = pd.DataFrame()
        flags.attrs = {"data_release": {}, "drift_symbols": []}
        return flat, flags, pd.DataFrame(), 1

    monkeypatch.setattr(module, "_read_combined_long", read)
    with pytest.raises(module.BarsValidationError, match="duplicate.*weekend"):
        load_bars(["X"], source="combined")


@pytest.mark.slow
def test_wide_frame_memory_budget(monkeypatch):
    import resource

    import numpy as np
    import pandas as pd

    import ggTrader.data.bars as module

    dates = pd.bdate_range("1950-01-02", periods=14_000)
    names = [f"S{i:04d}" for i in range(1000)]
    count = len(dates) * len(names)
    flat = pd.DataFrame(
        {
            "timestamp": np.tile(dates.to_numpy(), len(names)),
            "symbol": pd.Categorical.from_codes(
                np.repeat(np.arange(len(names)), len(dates)), names
            ),
        }
    )
    for field in ("open", "high", "low", "close", "volume"):
        flat[field] = np.ones(count, dtype="float64")
    monkeypatch.setattr(module, "_read_db_long", lambda *args: flat)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = load_bars(names, source="db")
    peak_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert result.close.shape == (14_000, 1000)
    assert peak_delta < 2.5 * 1024 * 1024


@pytest.mark.integration
@pytest.mark.db
def test_missing_symbols_lists_unknown_permno_and_k(published):
    _, _, rid = published
    bars = load_bars(["PERMNO-84761", "PERMNO-1~7"], source="combined", release_id=rid)
    assert bars.missing_symbols == ("PERMNO-1~7", "PERMNO-84761")
    assert bars.flags.empty
    assert len(bars.provenance["bars_sha256"]) == 64
