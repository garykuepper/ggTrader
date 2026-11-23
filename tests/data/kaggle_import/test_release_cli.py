"""Release CLI builds and withdraws only in a scratch database."""

import argparse
import hashlib
import json
import os
from urllib.parse import urlparse, urlunparse

import pytest

from ggTrader.cli.cmd_db import register_db_parser, run_db
from ggTrader.data.kaggle_import.guards import checked_pg_url
from ggTrader.data.kaggle_import.release import _verify_link_artifact as real_verify
from tests.data.kaggle_import.test_link_manifest import artifact
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture

conn = db_fixture
pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_build_list_withdraw(conn, monkeypatch, tmp_path, capsys):
    import ggTrader.data.kaggle_import.release as release_module

    db, load = conn
    monkeypatch.setattr(release_module, "_verify_link_artifact", real_verify)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,file_sha256) VALUES "
            "('crsp-daily','complete',%s) RETURNING load_id",
            ("a" * 64,),
        )
        raw = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,file_path,parser_version) VALUES "
            "('crsp-derived-bars','running',%s,'derive-1') RETURNING load_id",
            (f"load:{raw}",),
        )
        derived = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "SELECT %s,permno,spell,date,close FROM market.daily_bars WHERE load_id=%s",
            (derived, load),
        )
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (derived,))
    db.commit()
    url = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
    monkeypatch.setenv("RELEASE_TEST_URL", urlunparse(url._replace(path="/" + db.info.dbname)))
    links, _ = _inputs()
    path, manifest_path, manifest = artifact(tmp_path)
    links.to_csv(path, index=False)
    manifest["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))

    def invoke(*args):
        parsed = parser.parse_args(
            ["db", "import", "release", *args, "--db-url-env", "RELEASE_TEST_URL"]
        )
        run_db(parsed)
        return capsys.readouterr().out

    command = ("build", "--crsp-bars-load", str(derived), "--links", str(path), "--yes")
    manifest_path.unlink()
    with pytest.raises(ValueError, match="kaggle_find_links"):
        invoke(*command)
    relaxed = {**manifest, "settings": {**manifest["settings"], "agreement_bar": 0.99}}
    manifest_path.write_text(json.dumps(relaxed))
    with pytest.raises(ValueError, match="approved"):
        invoke(*command)
    manifest_path.write_text(json.dumps(manifest))
    output = invoke(*command)
    assert "Release 1" in output and "linked_yahoo" in output and "fingerprint" in output.lower()
    listing = invoke("list")
    assert "published" in listing and "1" in listing and "linked_yahoo" in listing
    assert "1" in invoke("withdraw", "--release-id", "1", "--yes")
    assert "withdrawn" not in invoke("list")
