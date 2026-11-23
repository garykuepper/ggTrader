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
def test_combined_returns_five_fields_and_flags_from_one_call(published):
    import pandas as pd

    from ggTrader.data.bars import _read_combined_long

    _, _, rid = published
    flat, flags, _, resolved = _read_combined_long(
        rid,
        ["PERMNO-2", "TTWO"],
        pd.Timestamp("1990-01-01"),
        pd.Timestamp("2010-01-01"),
        batch_size=1,
    )
    assert resolved == rid
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
