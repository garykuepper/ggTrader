"""Release reader integration tests against synthetic data."""

from datetime import date

import pandas as pd
import pytest

from ggTrader.data.kaggle_import.reader import read_combined
from ggTrader.data.kaggle_import.release import ReleaseUnavailable, build_release, withdraw_release
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture

conn = db_fixture

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def release(request):
    db, load = request.getfixturevalue("conn")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close,abs_prc,midpoint_flag) VALUES (%s,10128,1,'1990-01-01',1,1,true),(%s,10128,2,'1991-01-01',2,2,false),(%s,10128,3,'1992-01-01',3,3,false)",  # noqa: E501
            (load,) * 3,
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) VALUES ('2000-01-01','VC','1d','yfinance',7),('2000-01-02','PLAIN','1d','yfinance',8),('2000-01-03','PLAIN','1d','yfinance',9)"  # noqa: E501
        )
    db.commit()
    links, settings = _inputs()
    links.loc[0, "status"] = "unlinked"
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    # Represent the Visteon split: the later spell is deliberately inaccessible.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) VALUES (%s,88319,1,'1990-01-01',4),(%s,88319,2,'1991-01-01',5)",  # noqa: E501
            (load, load),
        )
        cur.execute("SET LOCAL ggt.release_build = 'on'")
        cur.execute(
            "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,possible_double) VALUES (%s,'VC',88319,2,'yahoo',true),(%s,'PERMNO-88319~1',88319,1,'spell',true)",  # noqa: E501
            (rid, rid),
        )
    db.commit()
    return db, rid


def read(release, symbols, start=None, end=None):
    db, rid = release
    return read_combined(db, rid, symbols, start, end)


def test_plain_yahoo_symbol_not_in_release(release):
    result = read(release, ["PLAIN"])
    assert result.source.tolist() == ["yfinance", "yfinance"]
    assert result.close.tolist() == [8, 9]
    assert not result.possible_double.any()


def test_placeholder_and_spell_columns(release):
    result = read(release, ["PERMNO-10128~1", "PERMNO-10128~2", "PERMNO-10128"])
    assert result.close.tolist() == [1, 2, 3]
    assert result.source.tolist() == ["crsp"] * 3
    assert result[["open", "high", "low", "volume"]].isna().all().all()
    assert result.abs_prc.tolist() == [1, 2, 3]
    assert result.midpoint_flag.tolist() == [True, False, False]


def test_visteon_yahoo_only_and_old_spell(release):
    assert read(release, ["VC"]).close.tolist() == [7]
    assert read(release, ["PERMNO-88319~1"]).close.tolist() == [4]
    assert read(release, ["PERMNO-88319"]).empty


def test_possible_double_flag_passes_through(release):
    assert read(release, ["VC", "PERMNO-88319~1"]).possible_double.all()


def test_date_range_inclusive_and_open_ended(release):
    assert read(release, ["PLAIN"], date(2000, 1, 2), date(2000, 1, 2)).close.tolist() == [8]
    assert read(release, ["PLAIN"], None, date(2000, 1, 2)).close.tolist() == [8]
    assert read(release, ["PLAIN"], date(2000, 1, 3), None).close.tolist() == [9]


def test_withdrawn_release_raises_unavailable_but_empty_symbols_is_empty(release):
    db, rid = release
    assert read(release, []).empty
    withdraw_release(db, rid)
    with pytest.raises(ReleaseUnavailable):
        read(release, ["PLAIN"])


def test_unknown_release_raises_unavailable(release):
    with pytest.raises(ReleaseUnavailable):
        read_combined(release[0], -1, ["PLAIN"], None, None)


def test_requested_symbol_missing_everywhere_returns_nothing(release):
    result = read(release, ["MISSING"])
    assert result.empty and str(result.ts.dtype) == "datetime64[ns]"


def test_no_duplicate_dates_per_symbol(release):
    result = read(release, ["PLAIN", "PLAIN"])
    assert not result.duplicated(["ts", "symbol"]).any()


def test_timezone_invariance(release):
    db, _ = release
    with db.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC'")
    utc = read(release, ["PLAIN", "PERMNO-10128"])
    with db.cursor() as cur:
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
    pd.testing.assert_frame_equal(utc, read(release, ["PLAIN", "PERMNO-10128"]))


@pytest.fixture
def linked(conn):
    db, load = conn
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    return db, rid


def test_linked_trace_and_join(linked):
    result = read(linked, ["TTWO"])
    assert result.close.tolist() == [5, 10, 11, 12]
    assert result.source.tolist() == ["crsp", "yfinance", "yfinance", "yfinance"]
    assert result.close.iloc[1] / result.close.iloc[0] == 2
    assert not result.drift.any()
    assert read(linked, ["PERMNO-1"]).empty


def test_linked_rescale_and_drift(linked):
    db, _ = linked
    before = read(linked, ["TTWO"])
    with db.cursor() as cur:
        cur.execute("UPDATE public.ohlcv SET close=close*0.9 WHERE symbol='TTWO'")
    after = read(linked, ["TTWO"])
    assert after.close.tolist() == pytest.approx((before.close * 0.9).tolist())
    assert not after.drift.any()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE public.ohlcv SET close=99 WHERE symbol='TTWO' AND timestamp='2000-01-03'"
        )
    assert read(linked, ["TTWO"]).drift.all()


def test_linked_backfill_and_missing(linked):
    from ggTrader.data.kaggle_import.reader import ReleaseInputMissing

    db, _ = linked
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) "
            "VALUES ('2000-01-01','TTWO','1d','yfinance',1)"
        )
    result = read(linked, ["TTWO"])
    assert result.drift.all() and result.source.tolist().count("yfinance") == 3
    with db.cursor() as cur:
        cur.execute("DELETE FROM public.ohlcv WHERE symbol='TTWO' AND timestamp='2000-01-02'")
    with pytest.raises(ReleaseInputMissing):
        read(linked, ["TTWO"])


@pytest.fixture
def two_linked(conn):
    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "VALUES (%s,3,1,'2000-01-02',1)",
            (load,),
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) VALUES "
            "('2000-01-01','AAPL','1d','yfinance',9),"
            "('2000-01-02','AAPL','1d','yfinance',10),"
            "('2000-01-03','AAPL','1d','yfinance',12),"
            "('2000-01-04','AAPL','1d','yfinance',13)"
        )
    db.commit()
    links, settings = _inputs()
    links.loc[1] = ["AAPL", 3, "linked", 1.0, 1.0, 1.0]
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    return db, rid


def test_smci_first_day_split_crsp_part_empty(two_linked):
    result = read(two_linked, ["AAPL"])
    assert result.source.tolist() == ["yfinance"] * 4
    assert not result.drift.any()


def test_two_linked_symbols_in_one_call(two_linked):
    result = read(two_linked, ["TTWO", "AAPL"])
    assert set(result.symbol) == {"TTWO", "AAPL"}
    assert len(result) == 8


def test_missing_crsp_join_row_raises(linked):
    from ggTrader.data.kaggle_import.reader import ReleaseInputMissing

    db, _ = linked
    with db.cursor() as cur:
        cur.execute("DELETE FROM market.daily_bars WHERE permno=1 AND date='2000-01-02'")
    with pytest.raises(ReleaseInputMissing):
        read(linked, ["TTWO"])


def test_drift_isolated_to_the_changed_symbol(two_linked):
    db, _ = two_linked
    with db.cursor() as cur:
        cur.execute(
            "UPDATE public.ohlcv SET close=99 WHERE symbol='TTWO' AND timestamp='2000-01-03'"
        )
    result = read(two_linked, ["TTWO", "AAPL"])
    assert result[result.symbol == "TTWO"].drift.all()
    assert not result[result.symbol == "AAPL"].drift.any()


def test_function_is_fast_enough_on_a_small_release(linked):
    from time import perf_counter

    start = perf_counter()
    read(linked, ["TTWO"])
    assert perf_counter() - start < 2


def test_pushdown_narrow_range(linked):
    db, rid = linked
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "SELECT load_id,1,1,make_date(y,1,1),1 FROM ref.release_loads, "
            "generate_series(1990,1998) y WHERE release_id=%s AND role='crsp-bars'",
            (rid,),
        )
        cur.execute(
            "SELECT chunk_name FROM timescaledb_information.chunks "
            "WHERE hypertable_schema='market' AND hypertable_name='daily_bars'"
        )
        chunks = {row[0] for row in cur.fetchall()}
        cur.execute(
            "EXPLAIN (ANALYZE, FORMAT JSON) SELECT b.date FROM market.daily_bars b "
            "WHERE b.load_id=(SELECT load_id FROM ref.release_loads "
            "WHERE release_id=%s AND role='crsp-bars') AND b.permno=1 AND b.spell=1 "
            "AND b.date < DATE '2000-01-02' AND b.date >= DATE '1999-01-01' "
            "AND b.date <= DATE '2000-01-01'",
            (rid,),
        )
        plan = cur.fetchone()[0]
    import json

    text = json.dumps(plan)
    assert len(chunks) > 2
    assert 0 < sum(name in text for name in chunks) < len(chunks)


def test_reader_does_not_scan_all_chunks(linked):
    db, rid = linked
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "SELECT load_id,1,1,make_date(y,1,1),1 FROM ref.release_loads, "
            "generate_series(1990,1998) y WHERE release_id=%s AND role='crsp-bars'",
            (rid,),
        )
        cur.execute(
            "SELECT chunk_name FROM timescaledb_information.chunks "
            "WHERE hypertable_schema='market' AND hypertable_name='daily_bars'"
        )
        chunks = {row[0] for row in cur.fetchall()}
        assert len(chunks) >= 4
        queries = [
            "SELECT (SELECT b.close FROM market.daily_bars b "
            "WHERE b.load_id=l.load_id AND b.permno=s.permno AND b.date=sp.common_day) "
            "FROM market.stock_spans s JOIN ref.release_loads l "
            "ON l.release_id=s.release_id AND l.role='crsp-bars' "
            "JOIN ref.splice_links sp ON sp.release_id=s.release_id AND sp.permno=s.permno "
            "WHERE s.release_id=%s AND s.symbol='TTWO'",
            "WITH linked AS MATERIALIZED (SELECT * FROM "
            "market._combined_linked(%s, ARRAY['TTWO'])) "
            "SELECT b.date FROM linked x JOIN LATERAL (SELECT b.* FROM market.daily_bars b "
            "WHERE b.load_id=x.load_id AND b.permno=x.permno AND b.spell=x.spell "
            "AND b.date<x.crsp_until AND b.date>=DATE '1999-01-01' "
            "AND b.date<=DATE '2000-01-01' OFFSET 0) b ON true",
        ]
        for query in queries:
            cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + query, (rid,))
            plan = cur.fetchone()[0][0]["Plan"]
            nodes = [plan]
            scanned = set()
            while nodes:
                node = nodes.pop()
                if node.get("Relation Name") in chunks and node.get("Actual Loops", 0) > 0:
                    scanned.add(node["Relation Name"])
                nodes.extend(node.get("Plans", []))
            assert 0 < len(scanned) < len(chunks), (scanned, chunks)
