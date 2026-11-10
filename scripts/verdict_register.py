"""Validate and render the source-linked research verdict register."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("research/programme/verdict-framework")
OUTPUT = Path("research/VERDICT_REGISTER.md")
EVIDENCE = {
    "SUPPORTED",
    "EVIDENCE AGAINST THE REGISTERED BENEFIT",
    "INCONCLUSIVE",
    "UNRESOLVED",
    "DESCRIPTIVE",
}
ACTIONS = {
    "CONTINUE RESEARCH",
    "ELIGIBLE FOR NEXT STAGE",
    "HOLD / PARK THIS VERSION",
    "CLOSE THIS SPECIFICATION",
}
PROTOCOLS = {"MET", "NOT MET", "STOPPED", "UNRESOLVED"}


def validate(inventory: dict, annotations: dict, root: Path) -> list[str]:
    """Check structure, source links, identifiers and inventory coverage."""
    errors: list[str] = []
    root = root.resolve()

    def required(obj: dict, key: str, context: str) -> str:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{context}: missing {key}")
            return ""
        return value

    def check_date(value: str, context: str) -> None:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            errors.append(f"{context}: invalid date {value!r}")

    def check_source(value: object, context: str) -> None:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{context}: missing source")
            return
        try:
            path = Path(value)
            resolved = (root / path).resolve()
            valid = not path.is_absolute() and resolved.is_relative_to(root) and resolved.is_file()
        except (OSError, ValueError, RuntimeError):
            valid = False
        if not valid:
            errors.append(f"{context}: invalid source {value!r}")

    if not isinstance(inventory, dict) or not isinstance(annotations, dict):
        return ["inventory and annotations must be objects"]
    check_date(required(inventory, "cutoff", "inventory"), "inventory")
    sources = inventory.get("sources")
    if not isinstance(sources, list):
        errors.append("inventory: sources must be a list")
    else:
        for source in sources:
            check_source(source, "inventory")
    coverage_complete = inventory.get("coverage_complete", False)
    if not isinstance(coverage_complete, bool):
        errors.append("inventory: coverage_complete must be a boolean")
    elif coverage_complete:
        check_source(inventory.get("coverage_evidence"), "inventory coverage")
    records = inventory.get("records")
    verdicts = annotations.get("verdicts")
    if not isinstance(records, list):
        errors.append("inventory: records must be a list")
        records = []
    if not isinstance(verdicts, list):
        errors.append("annotations: verdicts must be a list")
        verdicts = []
    by_id: dict[str, dict] = {}
    for row in records:
        if not isinstance(row, dict):
            errors.append("inventory: record must be an object")
            continue
        ident = required(row, "inventory_id", "inventory record")
        if ident in by_id:
            errors.append(f"duplicate inventory ID: {ident}")
        by_id[ident] = row
        check_source(row.get("source"), ident)
        disposition = row.get("disposition")
        if not isinstance(disposition, str) or disposition not in {
            "pending",
            "included",
            "excluded",
        }:
            errors.append(f"{ident}: invalid disposition")
        if disposition == "excluded":
            required(row, "reason", ident)
    seen_verdicts: set[str] = set()
    seen_revisions: set[str] = set()
    covered: set[str] = set()
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            errors.append("verdict must be an object")
            continue
        ident = required(verdict, "verdict_id", "verdict")
        if ident in seen_verdicts:
            errors.append(f"duplicate verdict ID: {ident}")
        seen_verdicts.add(ident)
        inv_id = required(verdict, "inventory_id", ident)
        row = by_id.get(inv_id)
        if row is None:
            errors.append(f"{ident}: missing inventory link {inv_id}")
        elif row.get("disposition") != "included":
            errors.append(f"{ident}: linked inventory record is not included")
        covered.add(inv_id)
        for field in ("study_id", "specification_id", "stage_id", "original_verdict"):
            required(verdict, field, ident)
        check_source(verdict.get("source"), ident)
        if (
            not isinstance(verdict.get("protocol_result"), str)
            or verdict["protocol_result"] not in PROTOCOLS
        ):
            errors.append(f"{ident}: invalid protocol result")
        revisions = verdict.get("revisions")
        if not isinstance(revisions, list) or not revisions:
            errors.append(f"{ident}: revisions must be a nonempty list")
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                errors.append(f"{ident}: revision must be an object")
                continue
            rev_id = required(revision, "annotation_revision_id", ident)
            if rev_id in seen_revisions:
                errors.append(f"duplicate annotation revision ID: {rev_id}")
            seen_revisions.add(rev_id)
            check_date(required(revision, "date", rev_id), rev_id)
            required(revision, "rationale", rev_id)
            revision_sources = revision.get("sources")
            if not isinstance(revision_sources, list) or not revision_sources:
                errors.append(f"{rev_id}: sources must be a nonempty list")
            else:
                for source in revision_sources:
                    check_source(source, rev_id)
            for field in ("approval", "uncertainties"):
                if field not in revision:
                    errors.append(f"{rev_id}: missing {field} (use null if unknown)")
            approval = revision.get("approval")
            if approval is not None and (
                not isinstance(approval, dict)
                or any(
                    not isinstance(approval.get(k), str) or not approval[k].strip()
                    for k in ("reference", "scope")
                )
            ):
                errors.append(f"{rev_id}: approval needs reference and scope")
            if revision.get("action") == "CONTINUE RESEARCH" and approval is None:
                errors.append(f"{rev_id}: CONTINUE RESEARCH requires approval and scope")
            uncertainty = revision.get("uncertainties")
            if uncertainty is not None and (
                not isinstance(uncertainty, str) or not uncertainty.strip()
            ):
                errors.append(f"{rev_id}: uncertainties must be text or null")
            if "protocol_result" in revision and (
                not isinstance(revision["protocol_result"], str)
                or revision["protocol_result"] not in PROTOCOLS
            ):
                errors.append(f"{rev_id}: invalid protocol result")
            if (
                not isinstance(revision.get("evidence"), str)
                or revision["evidence"] not in EVIDENCE
            ):
                errors.append(f"{rev_id}: invalid evidence label")
            if not isinstance(revision.get("action"), str) or revision["action"] not in ACTIONS:
                errors.append(f"{rev_id}: invalid action label")
    verdict_by_id = {
        v["verdict_id"]: v
        for v in verdicts
        if isinstance(v, dict) and isinstance(v.get("verdict_id"), str)
    }
    for ident, verdict in verdict_by_id.items():
        parent = verdict.get("parent_verdict_id")
        if parent is None:
            continue
        if not isinstance(parent, str) or parent not in verdict_by_id:
            errors.append(f"{ident}: invalid parent verdict")
            continue
        if verdict.get("study_id") != verdict_by_id[parent].get("study_id"):
            errors.append(f"{ident}: parent belongs to a different study")
        visited = {ident}
        while isinstance(parent, str) and parent in verdict_by_id:
            if parent in visited:
                errors.append(f"{ident}: parent cycle")
                break
            visited.add(parent)
            parent = verdict_by_id[parent].get("parent_verdict_id")
    for ident, row in by_id.items():
        if row.get("disposition") == "included" and ident not in covered:
            errors.append(f"{ident}: included inventory record has no verdict")
    return errors


def validate_history(baseline: dict, current: dict) -> list[str]:
    """Reject changes or removals of baseline verdicts and their ordered revisions."""
    errors: list[str] = []
    if not isinstance(baseline, dict) or not isinstance(baseline.get("verdicts"), list):
        return ["invalid baseline annotations"]
    if not isinstance(current, dict) or not isinstance(current.get("verdicts"), list):
        return ["invalid current annotations"]
    existing = {
        v["verdict_id"]: v for v in current["verdicts"] if isinstance(v, dict) and "verdict_id" in v
    }
    for old in baseline["verdicts"]:
        if (
            not isinstance(old, dict)
            or not isinstance(old.get("verdict_id"), str)
            or not isinstance(old.get("revisions"), list)
        ):
            errors.append("invalid baseline verdict")
            continue
        ident = old["verdict_id"]
        new = existing.get(ident)
        if new is None:
            errors.append(f"{ident}: baseline verdict removed")
            continue
        if {k: v for k, v in old.items() if k != "revisions"} != {
            k: v for k, v in new.items() if k != "revisions"
        }:
            errors.append(f"{ident}: baseline verdict changed")
        revisions = new.get("revisions")
        if (
            not isinstance(revisions, list)
            or revisions[: len(old["revisions"])] != old["revisions"]
        ):
            errors.append(f"{ident}: baseline revision history changed")
    return errors


def validate_inventory_history(baseline: dict, current: dict) -> list[str]:
    """Keep discovered records and their source identity through review progression."""
    if not isinstance(baseline, dict) or not isinstance(baseline.get("records"), list):
        return ["invalid baseline inventory"]
    if not isinstance(current, dict) or not isinstance(current.get("records"), list):
        return ["invalid current inventory"]
    errors: list[str] = []
    existing = {
        r["inventory_id"]: r
        for r in current["records"]
        if isinstance(r, dict) and isinstance(r.get("inventory_id"), str)
    }
    for old in baseline["records"]:
        if not isinstance(old, dict) or not isinstance(old.get("inventory_id"), str):
            errors.append("invalid baseline inventory record")
            continue
        ident = old["inventory_id"]
        new = existing.get(ident)
        if new is None:
            errors.append(f"{ident}: baseline inventory record removed")
            continue
        if old.get("source") != new.get("source"):
            errors.append(f"{ident}: inventory source changed")
        previous = old.get("disposition")
        disposition = new.get("disposition")
        if previous != "pending" and disposition != previous:
            errors.append(f"{ident}: reviewed inventory disposition changed")
        if (
            previous == "pending"
            and disposition == "excluded"
            and (not isinstance(new.get("reason"), str) or not new["reason"].strip())
        ):
            errors.append(f"{ident}: exclusion requires a reason")
    return errors


def revision_anchor(ident: str) -> str:
    """Make a stable HTML-safe anchor from an arbitrary revision ID."""
    return "revision-" + hashlib.sha256(ident.encode("utf-8")).hexdigest()


def render(inventory: dict, annotations: dict) -> str:
    """Render a deterministic summary and the complete revision history."""
    records = inventory["records"]
    counts = {
        key: sum(r["disposition"] == key for r in records)
        for key in ("included", "pending", "excluded")
    }
    status = (
        "REVIEWED"
        if records and not counts["pending"] and inventory.get("coverage_complete") is True
        else "INCOMPLETE"
    )
    lines = [
        "# Research verdict register",
        "",
        f"Inventory cutoff: {inventory['cutoff']}  ",
        f"Status: **{status}**  ",
        f"Identified: {len(records)}; "
        + "; ".join(f"{key}: {value}" for key, value in counts.items()),
        "",
        "Original study records remain authoritative. An empty inventory does not establish "
        "coverage. Included counts contain summary records as well as individual records; "
        "they are not a count of independent tests. Pending rows transcribe discovery sources "
        "and have not been independently "
        "reviewed against the original study.",
        "",
        "| Study / specification / stage | Original verdict | Protocol | "
        "Evidence | Action | Source / annotation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    verdicts = sorted(annotations["verdicts"], key=lambda v: v["verdict_id"])
    for verdict in verdicts:
        latest = max(enumerate(verdict["revisions"]), key=lambda item: (item[1]["date"], item[0]))[
            1
        ]
        chronological = sorted(verdict["revisions"], key=lambda r: r["date"])
        current_protocol = verdict["protocol_result"]
        current_role = None
        for revision in chronological:
            current_protocol = revision.get("protocol_result", current_protocol)
            current_role = revision.get("record_role", current_role)
        label = " / ".join(cell(verdict[k]) for k in ("study_id", "specification_id", "stage_id"))
        if current_role == "summary":
            label += " (Summary, not an independent test)"
        anchor = revision_anchor(latest["annotation_revision_id"])
        lines.append(
            f"| {label} | {cell(verdict['original_verdict'])} | "
            f"{cell(current_protocol)} | {cell(latest['evidence'])} | "
            f"{cell(latest['action'])} | [source](../{quote(verdict['source'], safe='/')}) / "
            f"[{cell(latest['annotation_revision_id'])}](#{anchor}) ({latest['date']}) |"
        )
    for row in sorted(records, key=lambda r: r["inventory_id"]):
        if row["disposition"] != "pending":
            continue
        title = cell(row.get("title", row["inventory_id"]))
        original = cell(row.get("original_audit_verdict", "unknown"))
        source = quote(row["source"], safe="/")
        lines.append(
            f"| {title} | {original} | NOT REVIEWED | NOT REVIEWED | "
            f"NOT REVIEWED | [source](../{source}) |"
        )
    lines.extend(["", "## Detailed records", ""])
    for verdict in verdicts:
        lines.extend(
            [
                f"### {verdict['verdict_id']}",
                "",
                f"Inventory: {verdict['inventory_id']}  ",
                f"Initial protocol determination: {verdict['protocol_result']}  ",
                f"Parent: {verdict.get('parent_verdict_id') or 'none'}  ",
                f"Source: [original](../{quote(verdict['source'], safe='/')})",
                "",
            ]
        )
        for rev in sorted(verdict["revisions"], key=lambda r: r["date"]):
            lines.extend(
                [
                    f'- <a id="{revision_anchor(rev["annotation_revision_id"])}"></a>'
                    f"**{rev['annotation_revision_id']}** ({rev['date']}): "
                    f"{rev['evidence']} / {rev['action']}. "
                    f"Protocol update: {rev.get('protocol_result', 'unchanged')}. "
                    f"{rev['rationale']} "
                    f"Approval: {rev['approval'] or 'unknown'}; "
                    f"Uncertainties: {rev['uncertainties'] or 'unknown'}. "
                    + " ".join(
                        f"[evidence](../{quote(source, safe='/')})" for source in rev["sources"]
                    )
                ]
            )
        lines.append("")
    excluded = [r for r in records if r["disposition"] == "excluded"]
    if excluded:
        lines.extend(["## Exclusions", ""])
        lines.extend(f"- {r['inventory_id']}: {r['reason']}" for r in excluded)
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Validate input and check or atomically render the register."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "render"))
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare with trusted annotations JSON; otherwise use the tracked HEAD version.",
    )
    parser.add_argument(
        "--inventory-baseline",
        type=Path,
        help="Compare with trusted inventory JSON; otherwise use the tracked HEAD version.",
    )
    args = parser.parse_args(argv)
    try:
        inventory = json.loads((ROOT / DIRECTORY / "inventory.json").read_text())
        annotations = json.loads((ROOT / DIRECTORY / "annotations.json").read_text())
        errors = validate(inventory, annotations, ROOT)

        def read_baseline(filename: str, explicit: Path | None) -> dict | None:
            if explicit:
                return json.loads(explicit.read_text())
            path = (DIRECTORY / filename).as_posix()
            tracked = subprocess.run(
                ["git", "cat-file", "-e", f"HEAD:{path}"], cwd=ROOT, capture_output=True
            )
            if tracked.returncode != 0:
                return None
            return json.loads(
                subprocess.run(
                    ["git", "show", f"HEAD:{path}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )

        baseline = read_baseline("annotations.json", args.baseline)
        inventory_baseline = read_baseline("inventory.json", args.inventory_baseline)
        if baseline is not None and not errors:
            errors.extend(validate_history(baseline, annotations))
        if inventory_baseline is not None and not errors:
            errors.extend(validate_inventory_history(inventory_baseline, inventory))
        if errors:
            print("\n".join(errors))
            return 1
        content = render(inventory, annotations)
        output = ROOT / OUTPUT
        if args.command == "check":
            if not output.is_file() or output.read_text() != content:
                print("Register is missing or stale; run render.")
                return 1
            print("Register is valid and current.")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=output.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                os.replace(name, output)
            finally:
                Path(name).unlink(missing_ok=True)
            print(f"Wrote {output}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Register error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
