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


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,123,'other')",
        "INSERT INTO ref.symbol_map(release_id,symbol) VALUES (%s,'ZZZ')",
        "INSERT INTO ref.splice_links(release_id,permno) VALUES (%s,999)",
        "INSERT INTO market.stock_spans(release_id,symbol) VALUES (%s,'ZZZ')",
    ],
)
def test_published_release_rejects_unapproved_inserts(conn, statement):
    db, load = conn
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with pytest.raises(psycopg2.Error, match="building"), db.cursor() as cur:
        cur.execute(statement, (rid,))
    db.rollback()


def test_cannot_withdraw_load_until_release_withdrawn(conn):
    db, load = conn
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with pytest.raises(psycopg2.Error, match="published release"), db.cursor() as cur:
        cur.execute("UPDATE ref.loads SET status='withdrawn' WHERE load_id=%s", (load,))
    db.rollback()
    withdraw_release(db, rid)
    with db.cursor() as cur:
        cur.execute("UPDATE ref.loads SET status='withdrawn' WHERE load_id=%s", (load,))
    db.commit()


def test_release_with_incomplete_referenced_load_is_unavailable(conn):
    from ggTrader.data.kaggle_import.reader import read_combined

    db, load = conn
    links, settings = _inputs()
    with db.cursor() as cur:
        cur.execute("INSERT INTO ref.loads(status) VALUES ('running') RETURNING load_id")
        running = cur.fetchone()[0]

    def add_invalid(cur, rid):
        cur.execute(
            "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,%s,'mcgill')",
            (rid, running),
        )

    rid = build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=add_invalid,
    )
    with pytest.raises(ReleaseUnavailable):
        read_combined(db, rid, ["TTWO"], None, None)
    db.rollback()


def test_release_load_role_is_unique(conn):
    db, load = conn
    links, settings = _inputs()

    def duplicate(cur, rid):
        cur.execute(
            "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,%s,'crsp-bars')",
            (rid, load),
        )

    with pytest.raises(psycopg2.errors.UniqueViolation):
        build_release(
            db,
            crsp_bars_load=load,
            mcgill_load=None,
            links=links,
            settings=settings,
            before_publish=duplicate,
        )


def test_combined_helper_rejects_incomplete_reference(conn):
    db, load = conn
    links, settings = _inputs()
    with db.cursor() as cur:
        cur.execute("INSERT INTO ref.loads(status) VALUES ('running') RETURNING load_id")
        other = cur.fetchone()[0]

    def add_invalid(cur, rid):
        cur.execute(
            "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,%s,'mcgill')",
            (rid, other),
        )

    rid = build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=add_invalid,
    )
    with pytest.raises(psycopg2.Error, match="release .* is unavailable"), db.cursor() as cur:
        cur.execute("SELECT * FROM market._combined_linked(%s, ARRAY['TTWO'])", (rid,))
    db.rollback()


def test_zero_yahoo_anchor_demotes_link_at_build_not_at_read(conn):
    from ggTrader.data.kaggle_import.reader import read_combined

    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "UPDATE public.ohlcv SET close=0 WHERE symbol='TTWO' AND timestamp='2000-01-02'"
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute(
            "SELECT symbol, kind, possible_double FROM market.stock_spans "
            "WHERE release_id=%s AND symbol IN ('TTWO','PERMNO-1') ORDER BY symbol",
            (rid,),
        )
        assert cur.fetchall() == [("PERMNO-1", "placeholder", True), ("TTWO", "yahoo", True)]
        cur.execute(
            "SELECT settings->'demoted_links' FROM ref.releases WHERE release_id=%s", (rid,)
        )
        assert cur.fetchone()[0][0]["symbol"] == "TTWO"
    assert read_combined(db, rid, ["TTWO"], None, None).close.tolist() == [0, 11, 12]


def test_null_crsp_close_on_common_day_demotes_link(conn):
    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "UPDATE market.daily_bars SET close=NULL WHERE load_id=%s AND permno=1 "
            "AND date='2000-01-02'",
            (load,),
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute(
            "SELECT kind,possible_double FROM market.stock_spans "
            "WHERE release_id=%s AND symbol='TTWO'",
            (rid,),
        )
        assert cur.fetchone() == ("yahoo", True)
        cur.execute(
            "SELECT settings->'demoted_links' FROM ref.releases WHERE release_id=%s", (rid,)
        )
        assert cur.fetchone()[0][0]["reason"]


def test_first_candidate_day_skipped_when_zero_then_next_day_used(conn):
    db, load = conn
    with db.cursor() as cur:
        cur.execute(
            "UPDATE market.daily_bars SET close=0 WHERE load_id=%s AND permno=1 "
            "AND date='2000-01-02'",
            (load,),
        )
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "VALUES (%s,1,1,'2000-01-03',3)",
            (load,),
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute(
            "SELECT common_day FROM ref.splice_links WHERE release_id=%s AND symbol='TTWO'", (rid,)
        )
        assert cur.fetchone()[0] == date(2000, 1, 3)


@pytest.mark.parametrize(
    "table", ["ref.release_loads", "ref.symbol_map", "ref.splice_links", "market.stock_spans"]
)
def test_old_and_nonexistent_release_reject_children_during_new_build(conn, table):
    db, load = conn
    links, settings = _inputs()
    old = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)

    def probe(cur, new):
        for target in (old, -999):
            cur.execute("SAVEPOINT child_probe")
            with pytest.raises(psycopg2.Error, match="building"):
                cur.execute(f"INSERT INTO {table} (release_id) VALUES (%s)", (target,))
            cur.execute("ROLLBACK TO SAVEPOINT child_probe")

    build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=probe,
    )


def test_building_release_unavailable_until_published(conn):
    from ggTrader.data.kaggle_import.reader import read_combined

    db, _ = conn
    # A separate uncommitted building row is unavailable to ordinary resolution.
    with db.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        rid = cur.fetchone()[0]
    db.commit()
    with pytest.raises(ReleaseUnavailable):
        read_combined(db, rid, ["TTWO"], None, None)
    db.rollback()
    with pytest.raises(ReleaseUnavailable):
        stocks_alive_on(db, rid, date(2000, 1, 1))
