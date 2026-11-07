"""L0: the upgrade tooling (workflow, Dependabot, runbook skill) is consistent with the repo."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "vbt-upgrade-watch.yml"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
SKILL = ROOT / ".claude" / "skills" / "vbt-upgrade" / "SKILL.md"
ENGINE_STACK = ["vectorbt", "vectorbt-rust", "pandas", "numpy", "numba", "llvmlite", "plotly"]


def _referenced_repo_paths(text: str) -> set[str]:
    return set(re.findall(r"\b((?:scripts|tests|src|docs)/[\w./-]+\.(?:py|md|toml|txt))", text))


def test_dependabot_ignores_exactly_the_engine_stack() -> None:
    text = DEPENDABOT.read_text()
    uv_block = text.split('package-ecosystem: "github-actions"')[0]
    ignored = re.findall(r'dependency-name:\s*"([^"]+)"', uv_block)
    assert ignored == ENGINE_STACK
    assert 'package-ecosystem: "uv"' in text
    assert "github-actions" in text and "docker" in text


def test_workflow_never_merges() -> None:
    text = WORKFLOW.read_text()
    assert "gh pr merge" not in text and "--auto" not in text and "automerge" not in text.lower()
    assert "--draft" in text


def test_workflow_runs_daily_on_both_engines_with_the_suites() -> None:
    text = WORKFLOW.read_text()
    assert re.search(r'cron:\s*"\d+ \d+ \* \* \*"', text)
    assert "workflow_dispatch" in text
    assert "engine: [rust, numba]" in text
    for needle in (
        "python -m ggTrader.engine.selfcheck",
        "pytest tests/engine",
        "uv lock",
        "uv sync --frozen --extra test --extra dev",
    ):
        assert needle in text, needle


def test_workflow_bumps_the_pair_together() -> None:
    text = WORKFLOW.read_text()
    assert r"vectorbt\[rust\]==" in text and "vectorbt-rust==" in text


@pytest.mark.parametrize("path", [WORKFLOW, SKILL])
def test_referenced_repo_files_exist(path: Path) -> None:
    optional = {"docs/vbt_upgrade_log.md", "docs/vbt_upgrade_rejected.txt"}  # created on first use
    missing = [
        p
        for p in sorted(_referenced_repo_paths(path.read_text()))
        if p not in optional and not (ROOT / p).exists() and "_*" not in p
    ]
    assert not missing, missing


def test_skill_has_frontmatter_and_the_decision_words() -> None:
    text = SKILL.read_text()
    head = text.split("---")[1]
    assert re.search(r"^name:\s*vbt-upgrade$", head, re.M)
    assert re.search(r"^description:\s*\S", head, re.M)
    for word in ("ACCEPT", "RE-BASELINE", "REJECT", "--golden-reason", "ggtrader-deploy"):
        assert word in text, word
    assert "crontab" in text  # the red line


def test_skill_commands_match_the_engine_entry_points() -> None:
    text = SKILL.read_text()
    assert "python -m ggTrader.engine.selfcheck" in text
    assert "scripts/run_tests.sh tests/engine" in text
    assert "CONTRACT_VERSION" in text and "BASELINE_EPOCH" in text
    from ggTrader.engine import config

    assert hasattr(config, "CONTRACT_VERSION") and hasattr(config, "BASELINE_EPOCH")


# --- workflow trust boundaries (review MAJOR-3) ----------------------------------------------


def _jobs(text: str) -> dict[str, str]:
    """Job name -> its YAML block (two-space-indented keys under ``jobs:``)."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"(?m)^  (\w+):\n", body)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _job_permissions(block: str) -> dict[str, str]:
    match = re.search(r"(?m)^    permissions:\n((?:      \S.*\n)+)", block)
    assert match, "job has no permissions block"
    return dict(re.findall(r"(\S+):\s*(\S+)", match.group(1)))


def test_workflow_grants_nothing_by_default_and_write_only_to_report() -> None:
    text = WORKFLOW.read_text()
    top = text.split("\njobs:\n", 1)[0]
    assert re.search(r"(?m)^permissions:\s*\{\}\s*$", top)
    assert not re.search(r"(?m)^permissions:\n", top), "no workflow-level grants"
    jobs = _jobs(text)
    assert set(jobs) == {"detect", "try", "report"}
    assert _job_permissions(jobs["detect"]) == {"contents": "read"}
    assert _job_permissions(jobs["try"]) == {"contents": "read"}
    assert _job_permissions(jobs["report"]) == {
        "contents": "write",
        "pull-requests": "write",
        "issues": "write",
    }


def test_the_job_that_runs_the_new_release_has_no_token_secret_or_cache() -> None:
    try_job = _jobs(WORKFLOW.read_text())["try"]
    assert "secrets." not in try_job and "GH_TOKEN" not in try_job
    assert "persist-credentials: false" in try_job
    assert "enable-cache: false" in try_job and "enable-cache: true" not in try_job
    assert "git push" not in try_job and "gh " not in try_job.replace("github", "")


def test_report_job_never_installs_or_runs_the_new_release() -> None:
    report = _jobs(WORKFLOW.read_text())["report"]
    for forbidden in ("uv sync", "uv lock", "pip install", "pytest", "ggTrader.engine", "setup-uv"):
        assert forbidden not in report, forbidden
    assert "grep -vq 'vectorbt'" in report  # the artifact pyproject may change vectorbt lines only


def test_untrusted_checkouts_do_not_persist_credentials() -> None:
    jobs = _jobs(WORKFLOW.read_text())
    for name in ("detect", "try"):
        checkout = jobs[name].split("actions/checkout@", 1)[1].split("\n\n", 1)[0]
        assert "persist-credentials: false" in checkout, name


def test_every_action_is_pinned_to_a_full_commit_sha_with_a_release_comment() -> None:
    uses = re.findall(r"(?m)^\s*- uses: (\S+)(.*)$", WORKFLOW.read_text())
    assert len(uses) >= 7
    for ref, comment in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref
        assert re.search(r"# v\d+\.\d+\.\d+$", comment), (ref, comment)


def test_workflow_header_records_the_verified_pins() -> None:
    header = WORKFLOW.read_text().split("\non:\n", 1)[0]
    assert "VERIFIED 2026-11-06" in header
    for ref, _ in re.findall(r"(?m)^\s*- uses: (\S+)(.*)$", WORKFLOW.read_text()):
        assert ref.split("@")[1] in header, ref


def test_dependabot_keeps_the_docker_base_on_python_3_12() -> None:
    docker = DEPENDABOT.read_text().split('package-ecosystem: "docker"')[1]
    assert 'dependency-name: "python"' in docker
    assert "version-update:semver-major" in docker and "version-update:semver-minor" in docker


def test_dockerfile_base_image_is_pinned_by_digest() -> None:
    first = next(
        line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("FROM ")
    )
    assert re.fullmatch(r"FROM python:3\.12-slim@sha256:[0-9a-f]{64}", first), first
