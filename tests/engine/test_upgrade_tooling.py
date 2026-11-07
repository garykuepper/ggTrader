"""L0: the upgrade workflow's trust boundaries (security pins), kept from regressing.

Only the security properties of ``vbt-upgrade-watch.yml`` and the Dockerfile are tested here: the
workflow never merges, grants and holds the least token scope, runs the new release without any
secret, and pins every action by commit SHA. The runbook-wording and Dependabot-consistency
tests that used to live in this file were removed in S1 (docs change; these do not).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "vbt-upgrade-watch.yml"

pytestmark = pytest.mark.tooling


def test_workflow_never_merges() -> None:
    text = WORKFLOW.read_text()
    assert "gh pr merge" not in text and "--auto" not in text and "automerge" not in text.lower()
    assert "--draft" in text


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


def test_dockerfile_base_image_is_pinned_by_digest() -> None:
    first = next(
        line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("FROM ")
    )
    assert re.fullmatch(r"FROM python:3\.12-slim@sha256:[0-9a-f]{64}", first), first
