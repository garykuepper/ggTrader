"""Read daily bars from a published data release."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import psycopg2
from psycopg2.extensions import connection

from .release import ReleaseUnavailable


class ReleaseInputMissing(Exception):
    """A required input to a release is missing."""


def read_combined(
    conn: connection, release_id: int, symbols: Sequence[str], start: date | None, end: date | None
) -> pd.DataFrame:
    """Return the requested release bars in chronological order."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM market.ohlcv_daily_combined(%s, %s, %s, %s)",
                (release_id, list(symbols), start, end),
            )
            rows = cur.fetchall()
            columns = [column.name for column in cur.description]
    except psycopg2.Error as exc:
        if f"release {release_id} is unavailable" in str(exc):
            raise ReleaseUnavailable(f"release {release_id} is unavailable") from None
        if "release input missing" in str(exc):
            raise ReleaseInputMissing(str(exc).splitlines()[0]) from None
        raise
    result = pd.DataFrame(rows, columns=columns)
    result["ts"] = pd.to_datetime(result["ts"]).astype("datetime64[ns]")
    return result.sort_values(["ts", "symbol"]).reset_index(drop=True)
