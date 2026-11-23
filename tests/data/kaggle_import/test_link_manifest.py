"""Release link artifacts must match their approved settings and source."""

import hashlib
import json
from dataclasses import asdict

import pytest

from ggTrader.data.kaggle_import.links import APPROVED_LINK_SETTINGS


def artifact(tmp_path):
    csv = tmp_path / "links.csv"
    csv.write_text("symbol,permno,status\nABC,1,linked\n")
    settings = asdict(APPROVED_LINK_SETTINGS)
    for key in ("window_start", "window_end"):
        settings[key] = settings[key].isoformat()
    manifest = {
        "settings": settings,
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "crsp_file_sha256": "a" * 64,
        "yahoo_fingerprint": {
            "n_symbols": 1,
            "newest_timestamp": "2020-01-01",
            "symbols_sha256": "b" * 64,
        },
        "git_sha": "abc",
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    path = csv.with_name(csv.name + ".manifest.json")
    path.write_text(json.dumps(manifest))
    return csv, path, manifest


def test_matching_manifest_is_accepted(tmp_path):
    from ggTrader.data.kaggle_import.link_manifest import verify_link_manifest

    csv, _, manifest = artifact(tmp_path)
    assert verify_link_manifest(csv, "a" * 64) == manifest


def test_missing_manifest_says_how_to_create_it(tmp_path):
    from ggTrader.data.kaggle_import.link_manifest import verify_link_manifest

    csv, path, _ = artifact(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="kaggle_find_links"):
        verify_link_manifest(csv, "a" * 64)


@pytest.mark.parametrize("change", ["settings", "csv", "source"])
def test_manifest_mismatch_is_refused(tmp_path, change):
    from ggTrader.data.kaggle_import.link_manifest import verify_link_manifest

    csv, path, manifest = artifact(tmp_path)
    if change == "settings":
        manifest["settings"]["agreement_bar"] = 0.99
        path.write_text(json.dumps(manifest))
    elif change == "csv":
        csv.write_text(csv.read_text() + "XYZ,2,linked\n")
    with pytest.raises(ValueError):
        verify_link_manifest(csv, "c" * 64 if change == "source" else "a" * 64)


@pytest.mark.integration
@pytest.mark.db
def test_publication_requires_verified_manifest_and_records_it(tmp_path, request):
    from ggTrader.data.kaggle_import.release import build_release
    from tests.data.kaggle_import.test_release import _inputs

    db, original = request.getfixturevalue("conn")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,file_sha256,parser_version) "
            "VALUES ('crsp-daily','complete',%s,'p') RETURNING load_id",
            ("a" * 64,),
        )
        raw = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,file_path,parser_version) "
            "VALUES ('crsp-derived-bars','running',%s,'derive-1') RETURNING load_id",
            (f"load:{raw}",),
        )
        derived = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "SELECT %s,permno,spell,date,close FROM market.daily_bars WHERE load_id=%s",
            (derived, original),
        )
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (derived,))
    db.commit()
    links, _ = _inputs()
    csv, path, manifest = artifact(tmp_path)
    links.to_csv(csv, index=False)
    manifest["csv_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    kwargs = dict(
        crsp_bars_load=derived,
        mcgill_load=None,
        links=links,
        settings=APPROVED_LINK_SETTINGS,
        links_path=csv,
    )
    for key, value in (
        ("settings", {**manifest["settings"], "agreement_bar": 0.99}),
        ("crsp_file_sha256", "b" * 64),
        ("csv_sha256", "c" * 64),
    ):
        path.write_text(json.dumps({**manifest, key: value}))
        with pytest.raises(ValueError):
            build_release(db, **kwargs)
    path.unlink()
    with pytest.raises(ValueError, match="kaggle_find_links"):
        build_release(db, **kwargs)
    path.write_text(json.dumps(manifest))
    rid = build_release(db, **kwargs)
    with db.cursor() as cur:
        cur.execute("SELECT settings FROM ref.releases WHERE release_id=%s", (rid,))
        assert cur.fetchone()[0]["link_manifest"] == manifest


from tests.data.kaggle_import.test_release import conn  # noqa: E402,F401
