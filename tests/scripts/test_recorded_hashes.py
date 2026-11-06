"""Check that files whose sha256 is recorded in frozen evidence still match it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "research/audits/event-timing/_event_evidence_resolution_workbook.json"
PLAN = "docs/superpowers/plans/2026-10-22-event-evidence-resolution.md"


def _find_spec_hashes(node: object) -> list[dict[str, str]]:
    """Collect every ``spec_hashes`` mapping in a parsed JSON tree."""
    found: list[dict[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "spec_hashes" and isinstance(value, dict):
                found.append(value)
            else:
                found.extend(_find_spec_hashes(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_spec_hashes(item))
    return found


def _find_input_hashes(node: object, path: str) -> list[str]:
    """Collect every ``{"path": path, "sha256": ...}`` digest in a parsed JSON tree."""
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("path") == path and isinstance(node.get("sha256"), str):
            found.append(node["sha256"])
        for value in node.values():
            found.extend(_find_input_hashes(value, path))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_input_hashes(item, path))
    return found


def test_restored_plan_matches_recorded_hash() -> None:
    """The plan file is byte-identical to the digest the frozen workbook recorded."""
    tree = json.loads(WORKBOOK.read_text(encoding="utf-8"))
    recorded = {m[PLAN] for m in _find_spec_hashes(tree) if PLAN in m}
    recorded.update(_find_input_hashes(tree, PLAN))
    assert len(recorded) == 1, f"expected one recorded digest for {PLAN}, got {recorded}"
    actual = hashlib.sha256((ROOT / PLAN).read_bytes()).hexdigest()
    assert actual == recorded.pop()
