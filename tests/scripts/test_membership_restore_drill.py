"""Fast tests of the restore-drill helpers and of the (uninstalled) weekly unit templates."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "membership_restore_drill.py"
UNITS = ROOT / "deploy" / "membership_weekly"


def _load():
    spec = importlib.util.spec_from_file_location("membership_restore_drill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_restore_command_keeps_the_password_out_of_the_arguments():
    drill = _load()
    url = drill.scratch_url(
        "postgresql://user:p%40ss@db.example:5433/ggtrader", "ggt_restore_drill_1"
    )
    assert url.endswith("/ggt_restore_drill_1")
    argv, env = drill.restore_command(url, Path("/x/universe.dump"))
    assert argv[0] == "pg_restore" and "--dbname=ggt_restore_drill_1" in argv
    assert "--host=db.example" in argv and "--port=5433" in argv and "--username=user" in argv
    assert "p@ss" not in " ".join(argv) and env == {"PGPASSWORD": "p@ss"}
    assert argv[-1] == "/x/universe.dump"


def test_compare_names_every_difference():
    drill = _load()
    assert drill.compare({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []
    problems = drill.compare({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert len(problems) == 2 and "b:" in problems[0] and "c:" in problems[1]


def test_help_runs_without_touching_anything():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd="/",
        check=False,
    )
    assert done.returncode == 0 and "usage" in done.stdout


def test_unit_templates_are_templates_that_do_not_notify_by_accident():
    service = (UNITS / "ggtrader-membership-weekly.service").read_text()
    timer = (UNITS / "ggtrader-membership-weekly.timer").read_text()
    assert "NOT INSTALLED" in service and "NOT INSTALLED" in timer
    assert "db membership weekly --notify" in service and "__REPO_ROOT__" in service
    assert "OnCalendar=Sat *-*-* 06:00:00 America/Los_Angeles" in timer
    readme = (UNITS / "README.md").read_text()
    assert "The files in this folder stay templates" in readme and "restore" in readme.lower()
