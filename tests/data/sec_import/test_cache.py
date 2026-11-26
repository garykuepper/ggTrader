"""Tests for the read-only SEC cache importer."""

import hashlib
import json
from pathlib import Path

import pytest

from ggTrader.data.sec_import import cache


def make_pair(folder: Path, url: str, body: bytes = b"body") -> Path:
    """Create a fake cache pair."""
    stem = hashlib.sha256(url.encode()).hexdigest()
    path = folder / f"{stem}.body"
    path.write_bytes(body)
    (folder / f"{stem}.meta.json").write_text(
        json.dumps(
            {
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "status": 200,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/master.zip", "quarterly_index"),
        ("https://data.sec.gov/submissions/CIK123.json", "submission_list"),
        ("https://www.sec.gov/Archives/edgar/data/123/", "directory_listing"),
        ("https://www.sec.gov/a.xml", "form25_xml"),
        ("https://www.sec.gov/Archives/edgar/data/123/a.txt", "filing_text"),
        ("https://www.sec.gov/a.pdf", "other"),
    ],
)
def test_classify_url_covers_every_kind(url: str, kind: str) -> None:
    assert cache.classify_url(url) == kind


def test_scan_cache_reads_pairs_and_checks_hash(tmp_path: Path) -> None:
    path = make_pair(tmp_path, "https://example.com/a")
    entries = cache.scan_cache(tmp_path)
    assert len(entries) == 1
    assert entries[0].body_path == path
    assert entries[0].bytes == 4
    assert entries[0].fetched_at.utcoffset() is not None


def test_scan_cache_raises_on_hash_mismatch(tmp_path: Path) -> None:
    path = make_pair(tmp_path, "https://example.com/a")
    path.write_bytes(b"changed")
    with pytest.raises(cache.CacheCorrupt):
        cache.scan_cache(tmp_path)


def test_scan_cache_raises_on_orphan_body(tmp_path: Path) -> None:
    (tmp_path / ("a" * 64 + ".body")).write_bytes(b"")
    with pytest.raises(cache.CacheCorrupt):
        cache.scan_cache(tmp_path)


def test_scan_raises_when_name_is_not_hash_of_url(tmp_path: Path) -> None:
    path = make_pair(tmp_path, "https://example.com/a")
    meta = path.with_name(path.stem + ".meta.json")
    data = json.loads(meta.read_text())
    data["url"] = "https://example.com/b"
    meta.write_text(json.dumps(data))
    with pytest.raises(cache.CacheCorrupt):
        cache.scan_cache(tmp_path)


def test_scan_ignores_stray_files_and_reports_them(tmp_path: Path) -> None:
    make_pair(tmp_path, "https://example.com/a")
    (tmp_path / "README.md").write_text("ignored")
    assert len(cache.scan_cache(tmp_path)) == 1
    assert cache.stray_files(tmp_path) == ["README.md"]


def test_merge_dedupes_identical_url_hash(tmp_path: Path) -> None:
    make_pair(tmp_path, "https://example.com/a")
    entries = cache.scan_cache(tmp_path)
    assert cache.merge_caches(entries, entries) == entries


def test_merge_conflict_on_different_content(tmp_path: Path) -> None:
    from dataclasses import replace

    make_pair(tmp_path, "https://example.com/a")
    entries = cache.scan_cache(tmp_path)
    with pytest.raises(cache.CacheConflict):
        cache.merge_caches(entries, [replace(entries[0], content_sha256="x")])


def test_manifest_is_order_independent(tmp_path: Path) -> None:
    make_pair(tmp_path, "https://example.com/a")
    make_pair(tmp_path, "https://example.com/b")
    entries = cache.scan_cache(tmp_path)
    assert cache.manifest_sha256(entries) == cache.manifest_sha256(entries[::-1])


def test_scan_never_opens_a_file_for_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pair(tmp_path, "https://example.com/a")
    original = Path.open
    modes = []

    def checked(path: Path, mode: str = "r", *args: object, **kwargs: object):
        modes.append(mode)
        assert mode in ("r", "rb")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked)
    cache.scan_cache(tmp_path)
    assert set(modes) == {"r", "rb"}


@pytest.mark.integration
@pytest.mark.db
def test_load_cache_registers_files_and_coverage_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry resumes a committed chunk and completes with a coverage label."""
    import psycopg2

    from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
    from ggTrader.data.sec_import.ddl import apply_ddl
    from tests.data.scratch_db import scratch_database

    folder = tmp_path / "cache"
    folder.mkdir()
    make_pair(folder, "https://example.com/a")
    (folder / "README.md").write_text("stray")
    monkeypatch.setattr(cache, "backup_fresh", lambda _: tmp_path)
    monkeypatch.setattr(cache, "free_space_ok", lambda *args, **kwargs: True)
    original = cache.complete_load

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted")

    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        apply_ddl(conn)
        monkeypatch.setattr(cache, "complete_load", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            cache.load_cache(conn, [folder], backup_dir=tmp_path, db_free_path=tmp_path)
        monkeypatch.setattr(cache, "complete_load", original)
        load_id = cache.load_cache(conn, [folder], backup_dir=tmp_path, db_free_path=tmp_path)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sec.fetched_files WHERE load_id=%s", (load_id,))
            assert cur.fetchone() == (1,)
            cur.execute("SELECT count(*) FROM ref.load_files WHERE load_id=%s", (load_id,))
            assert cur.fetchone() == (1,)
            cur.execute(
                "SELECT coverage, checks FROM ref.load_coverage JOIN ref.loads USING (load_id) "
                "WHERE load_id=%s",
                (load_id,),
            )
            coverage, checks = cur.fetchone()
            assert coverage == "not_a_stock_source"
            assert checks["stray_files"] == {str(folder): ["README.md"]}
