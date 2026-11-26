import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from scripts import db_dump


class Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query):
        pass

    def fetchone(self):
        return ("2.25.1",)


class Connection:
    def cursor(self):
        return Cursor()

    def close(self):
        pass


@pytest.fixture
def setup_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(db_dump, "THESIX", tmp_path.parent)
    monkeypatch.setenv("GGTRADER_DB_URL", "fake")
    monkeypatch.setattr(db_dump.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 10**9))
    monkeypatch.setattr(db_dump, "snapshot", lambda _: {"counts": {"x": 1}, "digests": {}})
    return tmp_path


def run(folder, size=1024 * 1024):
    def runner(command, stdout, check):
        stdout.write(b"a" * size)

    return db_dump.main(["--out-dir", str(folder)], connect=lambda _: Connection(), runner=runner)


def test_refuses_out_dir_outside_thesix(setup_dump, tmp_path):
    assert run(tmp_path.parent.parent) == 2


def test_refuses_when_free_space_low(setup_dump, monkeypatch):
    monkeypatch.setattr(db_dump.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    assert run(setup_dump) == 2


def test_writes_part_then_renames_and_records_sha256(setup_dump):
    assert run(setup_dump) == 0
    dump = setup_dump / db_dump.dump_filename(date.today())
    assert dump.exists()
    assert (
        setup_dump / "SHA256SUMS"
    ).read_text() == f"{hashlib.sha256(dump.read_bytes()).hexdigest()}  {dump.name}\n"
    assert not list(setup_dump.glob("*.part"))


def test_counts_before_after_mismatch_exits_3_and_keeps_part_file_out_of_final_name(
    setup_dump, monkeypatch
):
    values = iter([{"counts": {"x": 1}}, {"counts": {"x": 2}}])
    monkeypatch.setattr(db_dump, "snapshot", lambda _: next(values))
    assert run(setup_dump) == 3
    assert list(setup_dump.glob("*.mismatch"))
    assert not list(setup_dump.glob("*.dump"))


def test_dump_under_one_megabyte_exits_1(setup_dump):
    assert run(setup_dump, 10) == 1
    assert not list(setup_dump.glob("*.dump"))


def test_counts_json_records_timescaledb_version(setup_dump):
    assert run(setup_dump) == 0
    record = json.loads(next(setup_dump.glob("counts_*.json")).read_text())
    assert record["timescaledb"] == "2.25.1"


def test_prune_keeps_newest_and_ignores_other_files(tmp_path):
    for day in ("20260101", "20260102"):
        (tmp_path / f"ggtrader_{day}.dump").touch()
        (tmp_path / f"counts_{day}.json").touch()
    (tmp_path / "other.dump").touch()
    db_dump.prune(tmp_path, 0)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "counts_20260102.json",
        "ggtrader_20260102.dump",
        "other.dump",
    ]


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        db_dump.main(["--help"])
    assert exc.value.code == 0
