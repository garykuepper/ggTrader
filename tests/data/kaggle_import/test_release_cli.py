"""Release CLI builds and withdraws only in a scratch database."""

import argparse
import os
from urllib.parse import urlparse, urlunparse

import pytest

from ggTrader.cli.cmd_db import register_db_parser, run_db
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture

conn = db_fixture
pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_build_list_withdraw(conn, monkeypatch, tmp_path, capsys):
    db, load = conn
    url = urlparse(os.environ["GGT_TEST_DB_URL"])
    monkeypatch.setenv("RELEASE_TEST_URL", urlunparse(url._replace(path="/" + db.info.dbname)))
    links, _ = _inputs()
    path = tmp_path / "links.csv"
    links.to_csv(path, index=False)
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))

    def invoke(*args):
        parsed = parser.parse_args(
            ["db", "import", "release", *args, "--db-url-env", "RELEASE_TEST_URL"]
        )
        run_db(parsed)
        return capsys.readouterr().out

    output = invoke("build", "--crsp-bars-load", str(load), "--links", str(path), "--yes")
    assert "Release 1" in output and "linked_yahoo" in output and "fingerprint" in output.lower()
    listing = invoke("list")
    assert "published" in listing and "1" in listing and "linked_yahoo" in listing
    assert "1" in invoke("withdraw", "--release-id", "1", "--yes")
    assert "withdrawn" in invoke("list")
