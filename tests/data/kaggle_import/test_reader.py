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
