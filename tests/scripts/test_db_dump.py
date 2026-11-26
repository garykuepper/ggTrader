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
        self.closed = True


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


def test_mismatch_on_every_attempt_exits_3_and_keeps_the_last_file_out_of_final_name(
    setup_dump, monkeypatch
):
    values = iter([{"counts": {"x": n}} for n in range(10)])
    monkeypatch.setattr(db_dump, "snapshot", lambda _: next(values))
    assert run(setup_dump) == 3
    assert len(list(setup_dump.glob("*.mismatch"))) == 1
    assert not list(setup_dump.glob("*.dump"))


def test_mismatch_on_first_attempt_is_retaken_once(setup_dump, monkeypatch):
    # before, after (differs) -> retry: before, after (same)
    values = iter([{"counts": {"x": n}} for n in (1, 2, 3, 3)])
    monkeypatch.setattr(db_dump, "snapshot", lambda _: next(values))
    calls = []

    def runner(command, stdout, check):
        calls.append(1)
        stdout.write(b"a" * 1024 * 1024)

    code = db_dump.main(
        ["--out-dir", str(setup_dump)], connect=lambda _: Connection(), runner=runner
    )
    assert code == 0
    assert len(calls) == 2
    assert not list(setup_dump.glob("*.mismatch"))
    assert (setup_dump / db_dump.dump_filename(date.today())).exists()


def test_prune_removes_mismatch_files_of_earlier_days_only(tmp_path):
    today = f"{date.today():%Y%m%d}"
    old = tmp_path / "ggtrader_20200101.dump.mismatch"
    current = tmp_path / f"ggtrader_{today}.dump.mismatch"
    old.write_bytes(b"x")
    current.write_bytes(b"x")
    db_dump.prune(tmp_path, 4)
    assert not old.exists()
    assert current.exists()


def test_dump_under_one_megabyte_exits_1(setup_dump):
    assert run(setup_dump, 10) == 1
    assert not list(setup_dump.glob("*.dump"))


def test_counts_json_records_timescaledb_version(setup_dump):
    assert run(setup_dump) == 0
    record = json.loads(next(setup_dump.glob("counts_*.json")).read_text())
    assert set(record) == {"day", "dump_bytes", "seconds", "timescaledb", "before", "after"}
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


def test_before_and_after_snapshots_use_separate_connections(setup_dump, monkeypatch):
    connections = []
    snapshots = []

    def connect(_):
        conn = Connection()
        conn.closed = False
        connections.append(conn)
        return conn

    def snapshot(conn):
        snapshots.append(conn)
        return {"counts": {}, "digests": {}}

    def runner(command, stdout, check):
        assert len(connections) == 1 and connections[0].closed
        stdout.write(b"a" * (1024 * 1024))

    monkeypatch.setattr(db_dump, "snapshot", snapshot)
    assert db_dump.main(["--out-dir", str(setup_dump)], connect=connect, runner=runner) == 0
    assert len(connections) == 2 and snapshots == connections
    assert all(conn.closed for conn in connections)


def test_connections_are_autocommit(setup_dump, monkeypatch):
    connections = []

    def connect(_):
        conn = Connection()
        conn.autocommit = False
        connections.append(conn)
        return conn

    def snapshot(conn):
        assert conn.autocommit is True
        return {"counts": {}, "digests": {}}

    def runner(command, stdout, check):
        stdout.write(b"a" * (1024 * 1024))

    monkeypatch.setattr(db_dump, "snapshot", snapshot)
    assert db_dump.main(["--out-dir", str(setup_dump)], connect=connect, runner=runner) == 0
    assert len(connections) == 2 and all(conn.autocommit for conn in connections)


def test_refuses_when_newest_earlier_dump_requires_more_space(setup_dump, monkeypatch):
    older = setup_dump / "ggtrader_20000101.dump"
    newer = setup_dump / "ggtrader_20000102.dump"
    older.touch()
    newer.touch()
    original_stat = type(newer).stat

    def fake_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == newer:
            return SimpleNamespace(st_mode=result.st_mode, st_size=20 * 10**9)
        return result

    monkeypatch.setattr(type(newer), "stat", fake_stat)
    monkeypatch.setattr(db_dump.shutil, "disk_usage", lambda _: SimpleNamespace(free=35 * 10**9))
    assert run(setup_dump) == 2


def test_build_dump_command_uses_ggtrader_superuser():
    assert db_dump.build_dump_command() == [
        "docker",
        "exec",
        "ggtrader_db",
        "pg_dump",
        "-U",
        "ggtrader",
        "-Fc",
        "-Z",
        "1",
        "--no-owner",
        "ggtrader",
    ]


def test_failed_dump_removes_its_own_part_file(setup_dump):
    def fail(command, stdout, check):
        stdout.write(b"partial")
        raise RuntimeError("failed")

    assert (
        db_dump.main(["--out-dir", str(setup_dump)], connect=lambda _: Connection(), runner=fail)
        == 1
    )
    assert not list(setup_dump.glob("*.part"))
    assert run(setup_dump) == 0


def test_existing_part_is_not_removed_on_refusal(setup_dump):
    existing = setup_dump / (db_dump.dump_filename(date.today()) + ".part")
    existing.write_bytes(b"other run")
    assert run(setup_dump) == 1
    assert existing.read_bytes() == b"other run"


def test_failure_writing_checksum_leaves_no_dump_and_no_part_file(setup_dump, monkeypatch):
    final = setup_dump / db_dump.dump_filename(date.today())
    original_open = type(final).open

    def fail_sums(path, *args, **kwargs):
        if path.name == "SHA256SUMS":
            raise OSError("write failed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(final), "open", fail_sums)
    assert run(setup_dump) == 1
    assert not final.exists()
    assert not list(setup_dump.glob("*.part"))


def test_exception_after_rename_does_not_touch_final_dump(setup_dump, monkeypatch):
    final = setup_dump / db_dump.dump_filename(date.today())
    original_write_text = type(final).write_text
    original_unlink = type(final).unlink
    removed = []

    def fail_counts(path, *args, **kwargs):
        if path.name.startswith("counts_"):
            raise OSError("write failed")
        return original_write_text(path, *args, **kwargs)

    def track_unlink(path, *args, **kwargs):
        removed.append(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(final), "write_text", fail_counts)
    monkeypatch.setattr(type(final), "unlink", track_unlink)
    assert run(setup_dump) == 1
    assert final.exists()
    assert not removed


def test_failure_message_hides_connection_details(setup_dump, capsys):
    def failing(_):
        raise RuntimeError("could not connect postgresql://ggtrader:SECRET@localhost/ggtrader")

    code = db_dump.main(["--out-dir", str(setup_dump)], connect=failing, runner=None)
    assert code == 1
    assert "SECRET" not in capsys.readouterr().err
