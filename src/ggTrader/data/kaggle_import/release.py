"""Publish immutable combinations of derived bars and Yahoo identities."""

import hashlib
from dataclasses import asdict
from datetime import date
from typing import Any

import pandas as pd
from psycopg2.extensions import connection
from psycopg2.extras import Json, execute_values

from ggTrader.data.kaggle_import.columns import plan_columns
from ggTrader.data.kaggle_import.links import LinkSettings, join_returns_hash
from ggTrader.data.kaggle_import.loadlog import LoadFailed


class ReleaseUnavailable(Exception):
    """A release does not exist or has been withdrawn."""


def _native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_release(  # noqa: PLR0912, PLR0915 — publication is one atomic operation
    conn: connection,
    *,
    crsp_bars_load: int,
    mcgill_load: int | None,
    links: pd.DataFrame,
    settings: LinkSettings,
) -> int:
    """Publish all release records atomically, or leave none of them behind."""
    try:
        with conn.cursor() as cur:
            loads = [(crsp_bars_load, "crsp-derived-bars")]
            if mcgill_load is not None:
                loads.append((mcgill_load, "mcgill-fiam"))
            versions = {}
            for load_id, dataset in loads:
                cur.execute(
                    "SELECT parser_version, file_path FROM ref.loads WHERE load_id=%s "
                    "AND dataset=%s AND status='complete'",
                    (load_id, dataset),
                )
                row = cur.fetchone()
                if row is None:
                    raise LoadFailed(f"Load {load_id} is not a complete {dataset} load")
                versions[dataset] = row[0]
                if dataset == "crsp-derived-bars":
                    source_path = row[1]
            cur.execute(
                "SELECT permno,spell,min(date),max(date) FROM market.daily_bars WHERE load_id=%s "
                "GROUP BY permno,spell",
                (crsp_bars_load,),
            )
            spells = pd.DataFrame(
                cur.fetchall(), columns=["permno", "spell", "first_date", "last_date"]
            )
            links = links.drop(columns=["first_yf_date", "last_yf_date"], errors="ignore")
            symbols = links.symbol.tolist()
            cur.execute(
                """WITH y AS (
                SELECT symbol,timestamp,close, row_number() OVER (PARTITION BY
                symbol ORDER BY timestamp) AS n
                FROM public.ohlcv WHERE venue='yfinance' AND interval='1d' AND symbol=ANY(%s)
            ), facts AS (
                SELECT symbol,min(timestamp)::date AS
                first_yf_date,max(timestamp)::date AS last_yf_date,
                       max(timestamp) AS newest,count(*) AS n_rows FROM y GROUP BY symbol
            ) SELECT f.symbol,f.first_yf_date,f.last_yf_date,f.newest,f.n_rows,
                array_agg(y.timestamp::date ORDER BY y.timestamp) FILTER (WHERE y.n <= %s),
                array_agg(y.close ORDER BY y.timestamp) FILTER (WHERE y.n <= %s)
              FROM facts f JOIN y USING(symbol)
              GROUP BY f.symbol,f.first_yf_date,f.last_yf_date,f.newest,f.n_rows""",
                (
                    symbols,
                    max(5, settings.hold_first_days + 1),
                    max(5, settings.hold_first_days + 1),
                ),
            )
            facts = pd.DataFrame(
                cur.fetchall(),
                columns=[
                    "symbol",
                    "first_yf_date",
                    "last_yf_date",
                    "newest",
                    "n_rows",
                    "dates",
                    "closes",
                ],
            )
            if len(facts) != len(links) or links.symbol.duplicated().any():
                raise ValueError("Yahoo symbols missing or duplicated")
            candidates = links.loc[links.status.eq("linked"), ["symbol", "permno"]].dropna()
            pairs = [(str(r.symbol), int(r.permno)) for r in candidates.itertuples(index=False)]
            common = []
            if pairs:
                cur.execute(
                    """WITH candidates AS (SELECT * FROM
                    unnest(%s::text[],%s::bigint[]) AS c(symbol,permno)),
                    y AS (SELECT o.symbol,o.timestamp::date AS day,
                    row_number() OVER (PARTITION BY o.symbol ORDER BY
                    o.timestamp) n
                          FROM public.ohlcv o JOIN candidates c ON
                          c.symbol=o.symbol WHERE o.venue='yfinance'
                          AND o.interval='1d')
                    SELECT DISTINCT ON (y.symbol) y.symbol,y.day,b.spell
                    FROM y JOIN candidates c USING(symbol)
                    JOIN market.daily_bars b ON b.load_id=%s AND b.permno=c.permno AND b.date=y.day
                    WHERE y.n<=5 ORDER BY y.symbol,y.day""",
                    ([p[0] for p in pairs], [p[1] for p in pairs], crsp_bars_load),
                )
                common = cur.fetchall()
            yahoo = links.merge(facts, on="symbol", validate="one_to_one").merge(
                pd.DataFrame(common, columns=["symbol", "common_day", "common_spell"]),
                on="symbol",
                how="left",
            )
            spans = plan_columns(spells, yahoo)
            joined = spans[spans.kind.eq("linked_yahoo")].merge(
                yahoo, on="symbol", suffixes=("_span", "")
            )
            returns = {}
            hashes = {}
            for row in joined.itertuples(index=False):
                closes = pd.Series(row.closes, index=pd.to_datetime(row.dates), dtype=float)
                vals = closes.pct_change(fill_method=None).iloc[1 : settings.hold_first_days + 1]
                if not vals.map(pd.notna).all():
                    raise ValueError(f"Invalid Yahoo join returns for {row.symbol}")
                returns[row.symbol] = [round(float(x), 8) for x in vals]
                hashes[row.symbol] = join_returns_hash(
                    closes, row.first_yf_date, settings.hold_first_days
                )
            fingerprint = hashlib.sha256(
                "\n".join(f"{x:.8f}" for symbol in sorted(returns) for x in returns[symbol]).encode(
                    "ascii"
                )
            ).hexdigest()
            config = {
                k: v.isoformat() if isinstance(v, date) else v for k, v in asdict(settings).items()
            }
            crsp_version = None
            if source_path and source_path.startswith("load:"):
                cur.execute(
                    "SELECT parser_version FROM ref.loads WHERE load_id=%s",
                    (int(source_path.removeprefix("load:")),),
                )
                source = cur.fetchone()
                crsp_version = source[0] if source else None
            config.update(
                gap_tolerance=0.02,
                max_gap=10,
                derive_parser_version=versions["crsp-derived-bars"],
                crsp_parser_version=crsp_version,
            )
            fingerprint_data = dict(
                sha256=fingerprint,
                n_links=len(returns),
                newest_yahoo_timestamp=facts.newest.max().isoformat() if len(facts) else None,
                n_yahoo_rows=int(facts.n_rows.sum()),
            )
            cur.execute(
                "INSERT INTO ref.releases(created_at,settings,status,yahoo_fingerprint) "
                "VALUES(now(),%s,'published',%s) RETURNING release_id",
                (Json(config), Json(fingerprint_data)),
            )
            release_id = cur.fetchone()[0]
            cur.executemany(
                "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,%s,%s)",
                [(release_id, crsp_bars_load, "crsp-bars")]
                + ([(release_id, mcgill_load, "mcgill")] if mcgill_load is not None else []),
            )
            cols = [
                "symbol",
                "permno",
                "spell",
                "first_date",
                "last_date",
                "linked",
                "kind",
                "crsp_until",
                "first_yf_date",
                "possible_double",
            ]
            if len(spans):
                execute_values(
                    cur,
                    "INSERT INTO market.stock_spans(release_id," + ",".join(cols) + ") VALUES %s",
                    [
                        (release_id, *(_native(v) for v in row))
                        for row in spans[cols].itertuples(index=False, name=None)
                    ],
                )
            for row in joined.itertuples(index=False):
                agreement = {
                    k: _native(getattr(row, k))
                    for k in ("corr", "corr_first_half", "corr_second_half")
                }
                cur.execute(
                    "INSERT INTO "
                    "ref.splice_links(release_id,permno,symbol,first_yf_date,common_day,"
                    "link_day_returns_hash,join_returns,agreement) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        release_id,
                        int(row.permno),
                        row.symbol,
                        _native(row.first_yf_date),
                        _native(row.common_day),
                        hashes[row.symbol],
                        returns[row.symbol],
                        Json(agreement),
                    ),
                )
            if pairs:
                execute_values(
                    cur,
                    "INSERT INTO "
                    "ref.symbol_map(release_id,permno,symbol,valid_from,valid_to,source) VALUES "
                    "%s",
                    [
                        (
                            release_id,
                            int(r.permno),
                            r.symbol,
                            _native(r.first_yf_date),
                            _native(r.last_yf_date),
                            "return-match",
                        )
                        for r in yahoo[yahoo.status.eq("linked")].itertuples(index=False)
                    ],
                )
            if mcgill_load is not None:
                cur.execute(
                    "INSERT INTO "
                    "ref.symbol_map(release_id,permno,symbol,valid_from,valid_to,source) SELECT "
                    "%s,permno,stock_ticker,min(date),max(date),'mcgill' FROM "
                    "kaggle_chars.mcgill_of(%s) WHERE permno IS NOT NULL AND stock_ticker IS NOT "
                    "NULL GROUP BY permno,stock_ticker",
                    (release_id, mcgill_load),
                )
        conn.commit()
        return release_id
    except Exception:
        conn.rollback()
        raise


def withdraw_release(conn: connection, release_id: int) -> None:
    """Withdraw a published release."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ref.releases SET status='withdrawn' WHERE release_id=%s AND "
                "status='published'",
                (release_id,),
            )
            if not cur.rowcount:
                raise ReleaseUnavailable(f"Release {release_id} is unavailable")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def stocks_alive_on(conn: connection, release_id: int, day: date) -> pd.DataFrame:
    """Return all release columns with data spanning the requested day."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM ref.releases WHERE release_id=%s AND status='published'", (release_id,)
        )
        if cur.fetchone() is None:
            raise ReleaseUnavailable(f"Release {release_id} is unavailable")
        cur.execute(
            "SELECT symbol,permno,spell,kind FROM market.stock_spans WHERE release_id=%s AND "
            "first_date<=%s AND last_date>=%s ORDER BY symbol",
            (release_id, day, day),
        )
        return pd.DataFrame(cur.fetchall(), columns=["symbol", "permno", "spell", "kind"])
