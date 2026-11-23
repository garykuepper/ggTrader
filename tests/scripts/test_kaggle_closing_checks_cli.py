"""CLI contract for the read-only import checks."""

from datetime import date

import pytest

from scripts.kaggle_closing_checks import build_parser, default_output, initial_checks


def test_target_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_arguments_and_default_output():
    args = build_parser().parse_args(["--db-url-env", "TEST_URL", "--release-id", "3"])
    assert args.db_url_env == "TEST_URL" and args.release_id == 3
    assert (
        str(default_output(date(2026, 11, 23), "pilot"))
        == "results/kaggle_checks/2026-11-23-pilot.json"
    )


def test_registry():
    checks = initial_checks()
    assert [c["id"] for c in checks] == [f"check_{i:02}" for i in range(1, 14)]
    assert all(c["status"] == "covered_by_tests" for c in checks[8:])


def test_baseline_records_and_detects_changed_hash(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import kaggle_closing_checks as closing

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args):
            pass

        def fetchall(self):
            return [("AAA",), ("BBB",)]

    class Connection:
        def cursor(self):
            return Cursor()

    state = {"hash": "first"}
    monkeypatch.setattr(
        closing,
        "_load",
        lambda *a, **kw: SimpleNamespace(
            close=SimpleNamespace(shape=(2, 2)), provenance={"bars_sha256": state["hash"]}
        ),
    )
    path = tmp_path / "baseline.json"
    closing._baseline(Connection(), "unused", save=path)
    assert closing._baseline(Connection(), "unused", compare=path)["matches"] is True
    state["hash"] = "changed"
    assert closing._baseline(Connection(), "unused", compare=path)["matches"] is False
