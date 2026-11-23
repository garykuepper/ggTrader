"""Pin the default bars source against an explicitly selected scratch database."""

import pandas as pd
import pytest

from ggTrader.data.bars import BAR_FIELDS, load_bars
from ggTrader.utils import config

pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_new_default_equals_explicit_db(db_url, conn, monkeypatch) -> None:
    """Both source choices read the same seeded rows and record the same hash."""
    with conn.cursor() as cur:
        cur.execute(
            """CREATE TABLE ohlcv (
                timestamp timestamp, symbol text, interval text,
                open double precision, high double precision, low double precision,
                close double precision, volume double precision, trades integer, venue text
            )"""
        )
        cur.executemany(
            "INSERT INTO ohlcv VALUES (%s, %s, '1d', %s, %s, %s, %s, %s, 1, 'yfinance')",
            [
                ("2024-01-02", "AAA", 10, 12, 9, 11, 100),
                ("2024-01-03", "AAA", 11, 13, 10, 12, 110),
                ("2024-01-02", "BBB", 20, 22, 19, 21, 200),
                ("2024-01-03", "BBB", 21, 23, 20, 22, 210),
            ],
        )
    conn.commit()
    monkeypatch.setattr(config, "get_db_connection_string", lambda: db_url)

    default = load_bars(["AAA", "BBB"], "2024-01-01", "2024-01-31")
    explicit = load_bars(["AAA", "BBB"], "2024-01-01", "2024-01-31", source="db")

    assert default.symbols == ["AAA", "BBB"]
    for field in BAR_FIELDS:
        pd.testing.assert_frame_equal(default.frame(field), explicit.frame(field))
    assert default.provenance["bars_sha256"] == explicit.provenance["bars_sha256"]
