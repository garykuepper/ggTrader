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
        return flat, flags, pd.DataFrame(), 1, None

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
    flat, flags, _, resolved, _ = _read_combined_long(
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
    _, _, spans, _, _ = _read_combined_long(
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
        return flat, flags, pd.DataFrame(), 1, None

    monkeypatch.setattr(module, "_read_combined_long", read)
    with pytest.raises(module.BarsValidationError, match="duplicate.*weekend"):
        load_bars(["X"], source="combined")


def test_wide_frame_memory_budget():
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import resource
        from datetime import datetime, timezone
        import numpy as np
        import pandas as pd
        import ggTrader.data.bars as bars
        import ggTrader.data.kaggle_import.reader as reader

        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, *args): self.sql = sql
            def fetchone(self):
                if "max(last_date)" in self.sql: return (None,)
                return (1, datetime.now(timezone.utc), {})
            def fetchall(self): return []
        class Connection:
            def set_session(self, **kw): pass
            def cursor(self): return Cursor()
            def close(self): pass
        bars.connect_readonly = lambda: Connection()
        dates = pd.bdate_range("2000-01-03", periods=5000)
        def fake(conn, release, symbols, start, end):
            n = len(dates) * len(symbols)
            data = {"ts": np.tile(dates.to_numpy(), len(symbols)),
                    "symbol": np.repeat(symbols, len(dates)),
                    "source": np.repeat("crsp", n), "drift": np.zeros(n, dtype=bool),
                    "midpoint_flag": np.zeros(n, dtype=bool),
                    "possible_double": np.zeros(n, dtype=bool)}
            for col in ("open", "high", "low", "close", "volume", "abs_prc"):
                data[col] = np.ones(n, dtype="float64")
            return pd.DataFrame(data)
        reader.read_combined = fake
        result = bars.load_bars([f"S{i:03}" for i in range(300)], source="combined")
        assert result.close.shape == (5000, 300)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(f"peak_bytes={peak} rows=1500000 bytes_per_row={peak / 1500000:.1f}")
        assert peak < 552000000  # 424267776 measured, plus 30% slack.
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    print(result.stdout, result.stderr)
    assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.db
def test_missing_symbols_lists_unknown_permno_and_k(published):
    _, _, rid = published
    bars = load_bars(["PERMNO-84761", "PERMNO-1~7"], source="combined", release_id=rid)
    assert bars.missing_symbols == ("PERMNO-1~7", "PERMNO-84761")
    assert bars.flags.empty
    assert len(bars.provenance["bars_sha256"]) == 64


def test_combined_sorted_plain_columns_and_flags_match_filtered_bars(monkeypatch):
    import pandas as pd

    import ggTrader.data.bars as module

    flat = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": pd.Categorical(["BBB", "BBB", "AAA"], categories=["AAA", "BBB", "CCC"]),
            **{name: [1.0, 2.0, 3.0] for name in ("open", "high", "low", "close", "volume")},
        }
    )
    flags = flat.set_index(["timestamp", "symbol"])[[]]
    flags.index = flags.index.set_levels(flags.index.levels[0].tz_localize("UTC"), level=0)
    flags.attrs = {"data_release": {}, "drift_symbols": []}

    def read(_release, _symbols, start, end):
        keep = (flat.timestamp >= start) & (flat.timestamp <= end)
        return flat.loc[keep].copy(), flags.loc[keep.to_numpy()], pd.DataFrame(), 1, None

    monkeypatch.setattr(module, "_read_combined_long", read)
    monkeypatch.setattr(
        module, "_read_db_long", lambda *a: flat.assign(symbol=flat.symbol.astype(str))
    )
    combined = load_bars(["AAA", "BBB", "CCC"], source="combined", start="2020-01-01 12:00")
    db = load_bars(["AAA", "BBB", "CCC"], source="db", start="2020-01-01 12:00")
    assert combined.symbols == ["AAA", "BBB"]
    assert not isinstance(combined.close.columns, pd.CategoricalIndex)
    pd.testing.assert_frame_equal(combined.close, db.close)
    assert combined.flags.index.get_level_values("timestamp").min() == pd.Timestamp(
        "2020-01-02", tz="UTC"
    )


@pytest.mark.integration
@pytest.mark.db
def test_combined_non_midnight_start_filters_flags(published):
    import pandas as pd

    _, _, rid = published
    bars = load_bars(["TTWO"], start="2000-01-03 12:00", source="combined", release_id=rid)
    assert bars.close.index.tolist() == [pd.Timestamp("2000-01-04", tz="UTC")]
    assert bars.flags.index.get_level_values("timestamp").tolist() == bars.close.index.tolist()


@pytest.mark.integration
@pytest.mark.db
def test_combined_records_crsp_end_in_provenance(published):
    _, _, rid = published
    bars = load_bars(["TTWO"], start="2000-01-03", source="combined", release_id=rid)
    assert bars.provenance["crsp_end"] == "1999-01-01"
