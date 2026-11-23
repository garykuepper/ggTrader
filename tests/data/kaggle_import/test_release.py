"""Release publication against an isolated scratch database."""

from datetime import date

import pandas as pd
import psycopg2
import pytest

from ggTrader.data.kaggle_import.ddl import apply_ddl
from ggTrader.data.kaggle_import.links import LinkSettings
from ggTrader.data.kaggle_import.release import (
    ReleaseUnavailable,
    build_release,
    stocks_alive_on,
    withdraw_release,
)
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def conn():
    with scratch_database() as url, psycopg2.connect(url) as db:
        apply_ddl(db)
        with db.cursor() as cur:
            cur.execute(
                "CREATE TABLE public.ohlcv (timestamp timestamp, symbol text, interval text, "
                "venue text, open float, high float, low float, close float, volume float, "
                "trades integer)"
            )
            cur.execute(
                "INSERT INTO ref.loads(dataset,status,parser_version) VALUES "
                "('crsp-derived-bars','running','derive-1') RETURNING load_id"
            )
            load = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) VALUES "
                "(%s,1,1,'2000-01-01',1),(%s,1,1,'2000-01-02',2),(%s,1,2,'2002-01-01',3),(%s,2,1,'1999-01-01',1)",
                (load,) * 4,
            )
            cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (load,))
            cur.execute(
                "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) VALUES "
                "('2000-01-02','TTWO','1d','yfinance',10),('2000-01-03','TTWO','1d','yfinance',11),('2000-01-04','TTWO','1d','yfinance',12)"
            )
        db.commit()
        yield db, load


def test_publish_and_withdraw(conn):
    db, load = conn
    links = pd.DataFrame(
        [
            dict(
                symbol="TTWO",
                permno=1,
                status="linked",
                corr=1.0,
                corr_first_half=1.0,
                corr_second_half=1.0,
            )
        ]
    )
    settings = LinkSettings(2, 1, 0.99, 0.001, date(2000, 1, 1), date(2024, 12, 31), 2)
    release = build_release(
        db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT symbol,kind FROM market.stock_spans WHERE release_id=%s ORDER BY symbol",
            (release,),
        )
        assert cur.fetchall() == [("PERMNO-2", "placeholder"), ("TTWO", "linked_yahoo")]
        cur.execute(
            "SELECT common_day, array_length(join_returns,1) FROM ref.splice_links WHERE "
            "release_id=%s",
            (release,),
        )
        assert cur.fetchone() == (date(2000, 1, 2), 2)
    assert "PERMNO-2" in stocks_alive_on(db, release, date(1999, 1, 1)).symbol.tolist()
    withdraw_release(db, release)
    with pytest.raises(ReleaseUnavailable):
        stocks_alive_on(db, release, date(1999, 1, 1))


def _inputs():
    links = pd.DataFrame(
        [
            dict(
                symbol="TTWO",
                permno=1,
                status="linked",
                corr=1.0,
                corr_first_half=1.0,
                corr_second_half=1.0,
            )
        ]
    )
    settings = LinkSettings(2, 1, 0.99, 0.001, date(2000, 1, 1), date(2024, 12, 31), 2)
    return links, settings


def test_fingerprint_and_two_releases(conn):
    db, load = conn
    links, settings = _inputs()
    one = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute("UPDATE public.ohlcv SET close=13 WHERE timestamp='2000-01-03'")
    db.commit()
    two = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute(
            "SELECT yahoo_fingerprint->>'sha256' FROM ref.releases WHERE release_id IN (%s,%s)",
            (one, two),
        )
        assert len({r[0] for r in cur.fetchall()}) == 2


def test_atomic_and_incomplete(conn, monkeypatch):
    import ggTrader.data.kaggle_import.release as release_module
    from ggTrader.data.kaggle_import.loadlog import LoadFailed

    db, load = conn
    links, settings = _inputs()
    with pytest.raises(LoadFailed):
        build_release(
            db, crsp_bars_load=load + 999, mcgill_load=None, links=links, settings=settings
        )

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(release_module, "execute_values", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        for table in (
            "ref.releases",
            "ref.release_loads",
            "ref.symbol_map",
            "ref.splice_links",
            "market.stock_spans",
        ):
            cur.execute(f"SELECT count(*) FROM {table}")
            assert cur.fetchone()[0] == 0


def test_unlinked_has_placeholder(conn):
    db, load = conn
    links, settings = _inputs()
    links.loc[0, "status"] = "unlinked"
    release = build_release(
        db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT symbol, kind, possible_double FROM market.stock_spans "
            "WHERE release_id=%s ORDER BY symbol",
            (release,),
        )
        assert cur.fetchall() == [
            ("PERMNO-1", "placeholder", True),
            ("PERMNO-1~1", "spell", True),
            ("PERMNO-2", "placeholder", False),
            ("TTWO", "yahoo", True),
        ]


def test_mcgill_and_link_dates_from_csv(conn):
    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.loads(dataset,status) VALUES ('mcgill-fiam','running') "
            "RETURNING load_id"
        )
        mcgill = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO kaggle_chars.mcgill(load_id,permno,stock_ticker,date) "
            "VALUES (%s,1,'OLD','1999-01-01'),(%s,1,'OLD','1999-02-01')",
            (mcgill, mcgill),
        )
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (mcgill,))
    db.commit()
    links, settings = _inputs()
    links["first_yf_date"] = "2000-01-02"
    links["last_yf_date"] = "2000-01-04"
    release = build_release(
        db, crsp_bars_load=load, mcgill_load=mcgill, links=links, settings=settings
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT symbol,source,valid_from,valid_to FROM ref.symbol_map "
            "WHERE release_id=%s ORDER BY symbol",
            (release,),
        )
        assert cur.fetchall() == [
            ("OLD", "mcgill", date(1999, 1, 1), date(1999, 2, 1)),
            ("TTWO", "return-match", date(2000, 1, 2), date(2000, 1, 4)),
        ]


def test_ttwo_and_aapl_are_both_linked(conn):
    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "VALUES (%s,3,1,'2000-01-02',1)",
            (load,),
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) "
            "VALUES ('2000-01-02','AAPL','1d','yfinance',10),"
            "('2000-01-03','AAPL','1d','yfinance',12)"
        )
    db.commit()
    links, settings = _inputs()
    links.loc[1] = ["AAPL", 3, "linked", 1.0, 1.0, 1.0]
    release = build_release(
        db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT symbol FROM market.stock_spans "
            "WHERE release_id=%s AND kind='linked_yahoo' ORDER BY symbol",
            (release,),
        )
        assert cur.fetchall() == [("AAPL",), ("TTWO",)]
    assert "PERMNO-2" not in stocks_alive_on(db, release, date(1999, 1, 2)).symbol.tolist()
