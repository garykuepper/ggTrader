"""Verify the verdict register contract without market data."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts import verdict_register as register


@pytest.fixture
def records(tmp_path):
    (tmp_path / "source.md").write_text("source")
    inventory = {
        "cutoff": "2026-11-10",
        "sources": ["source.md"],
        "records": [
            {"inventory_id": "i1", "disposition": "included", "source": "source.md"},
            {"inventory_id": "i2", "disposition": "pending", "source": "source.md"},
            {
                "inventory_id": "i3",
                "disposition": "excluded",
                "reason": "status only",
                "source": "source.md",
            },
        ],
    }
    annotations = {
        "verdicts": [
            {
                "verdict_id": "v1",
                "inventory_id": "i1",
                "study_id": "study",
                "specification_id": "spec",
                "stage_id": "stage",
                "original_verdict": "NO-GO",
                "source": "source.md",
                "protocol_result": "NOT MET",
                "revisions": [
                    {
                        "annotation_revision_id": "r1",
                        "date": "2026-01-01",
                        "evidence": "INCONCLUSIVE",
                        "action": "HOLD / PARK THIS VERSION",
                        "rationale": "Unknown power",
                        "approval": None,
                        "uncertainties": None,
                        "sources": ["source.md"],
                    },
                ],
            }
        ]
    }
    return inventory, annotations


def test_valid_revisions_and_stages_preserve_identity(records, tmp_path):
    inventory, annotations = records
    verdict = annotations["verdicts"][0]
    newer = dict(verdict["revisions"][0], annotation_revision_id="r2", date="2026-02-01")
    verdict["revisions"].append(newer)
    second = copy.deepcopy(verdict)
    second.update(verdict_id="v2", stage_id="another")
    second["revisions"] = [dict(newer, annotation_revision_id="r3")]
    inventory["records"].append(
        {"inventory_id": "i4", "disposition": "included", "source": "source.md"}
    )
    second["inventory_id"] = "i4"
    annotations["verdicts"].append(second)
    assert register.validate(inventory, annotations, tmp_path) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_verdict",
        "duplicate_revision",
        "missing_inventory",
        "missing_source",
        "uncovered",
        "pending_reviewed",
        "no_reason",
        "outside",
        "bad_label",
    ],
)
def test_invalid_records_are_rejected(records, tmp_path, mutation):
    inventory, annotations = records
    verdict = annotations["verdicts"][0]
    if mutation == "duplicate_verdict":
        annotations["verdicts"].append(copy.deepcopy(verdict))
    elif mutation == "duplicate_revision":
        verdict["revisions"].append(copy.deepcopy(verdict["revisions"][0]))
    elif mutation == "missing_inventory":
        verdict["inventory_id"] = "missing"
    elif mutation == "missing_source":
        verdict["source"] = "missing.md"
    elif mutation == "uncovered":
        annotations["verdicts"] = []
    elif mutation == "pending_reviewed":
        inventory["records"][0]["disposition"] = "pending"
    elif mutation == "no_reason":
        del inventory["records"][2]["reason"]
    elif mutation == "outside":
        verdict["source"] = "../source.md"
    else:
        verdict["revisions"][0]["evidence"] = "GOOD"
    assert register.validate(inventory, annotations, tmp_path)


def test_render_preserves_history_and_reports_coverage(records):
    inventory, annotations = records
    revisions = annotations["verdicts"][0]["revisions"]
    revisions.insert(
        0,
        dict(
            revisions[0],
            annotation_revision_id="r2",
            date="2026-02-01",
            evidence="DESCRIPTIVE",
            rationale="Later review",
        ),
    )
    output = register.render(inventory, annotations)
    assert output == register.render(inventory, annotations)
    for text in (
        "Original verdict",
        "Protocol",
        "Evidence",
        "Action",
        "Source",
        "INCOMPLETE",
        "included: 1",
        "pending: 1",
        "excluded: 1",
        "r1",
        "r2",
        "Unknown power",
        "Later review",
    ):
        assert text in output
    summary = output.split("## Detailed records")[0]
    assert "DESCRIPTIVE" in summary
    assert "INCONCLUSIVE" not in summary


def test_check_detects_stale_register_without_writing(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    output = tmp_path / "research/VERDICT_REGISTER.md"
    output.write_text("stale\n")
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["check"]) != 0
    assert output.read_text() == "stale\n"
    assert "stale" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("field", ["disposition", "protocol_result", "evidence", "action"])
def test_malformed_label_types_are_reported(records, tmp_path, field):
    inventory, annotations = records
    if field == "disposition":
        inventory["records"][0][field] = []
    elif field == "protocol_result":
        annotations["verdicts"][0][field] = []
    else:
        annotations["verdicts"][0]["revisions"][0][field] = []
    assert register.validate(inventory, annotations, tmp_path)


def test_revision_requires_explicit_approval_and_uncertainty(records, tmp_path):
    inventory, annotations = records
    revision = annotations["verdicts"][0]["revisions"][0]
    del revision["approval"]
    assert register.validate(inventory, annotations, tmp_path)
    revision.update(approval=None, uncertainties=None)
    assert not register.validate(inventory, annotations, tmp_path)
    output = register.render(inventory, annotations)
    assert "Approval: unknown" in output
    assert "Uncertainties: unknown" in output


def test_continue_research_requires_approved_scope(records, tmp_path):
    inventory, annotations = records
    revision = annotations["verdicts"][0]["revisions"][0]
    revision.update(action="CONTINUE RESEARCH", approval=None, uncertainties=None)
    assert register.validate(inventory, annotations, tmp_path)
    revision["approval"] = {"reference": "owner decision", "scope": "specific follow-up"}
    assert not register.validate(inventory, annotations, tmp_path)
    assert "specific follow-up" in register.render(inventory, annotations)


def test_history_disallows_edits_and_deletions(records):
    inventory, baseline = records
    updated = copy.deepcopy(baseline)
    updated["verdicts"][0]["revisions"].append(
        dict(updated["verdicts"][0]["revisions"][0], annotation_revision_id="new")
    )
    assert register.validate_history(baseline, updated) == []
    updated["verdicts"][0]["revisions"].pop(0)
    assert register.validate_history(baseline, updated)
    updated = copy.deepcopy(baseline)
    updated["verdicts"][0]["revisions"][0]["rationale"] = "rewritten"
    assert register.validate_history(baseline, updated)
    updated = copy.deepcopy(baseline)
    updated["verdicts"][0]["original_verdict"] = "GO"
    assert register.validate_history(baseline, updated)
    assert register.validate_history(baseline, {"verdicts": []})


def test_source_path_rejects_symlink_escape_and_invalid_path(records, tmp_path):
    inventory, annotations = records
    outside = tmp_path.parent / "outside-source.md"
    outside.write_text("outside")
    (tmp_path / "escape.md").symlink_to(outside)
    annotations["verdicts"][0]["source"] = "escape.md"
    assert register.validate(inventory, annotations, tmp_path)
    annotations["verdicts"][0]["source"] = "bad\x00path"
    assert register.validate(inventory, annotations, tmp_path)


def test_summary_links_to_safe_anchor_and_escaped_source(records, tmp_path):
    inventory, annotations = records
    revision = annotations["verdicts"][0]["revisions"][0]
    revision.update(annotation_revision_id="revision ) #1", approval=None, uncertainties=None)
    (tmp_path / "source (1).md").write_text("source")
    annotations["verdicts"][0]["source"] = "source (1).md"
    assert not register.validate(inventory, annotations, tmp_path)
    output = register.render(inventory, annotations)
    assert "../source%20%281%29.md" in output
    assert "#revision-" in output and '<a id="revision-' in output


def test_cli_rejects_history_changes_against_supplied_baseline(records, tmp_path, monkeypatch):
    inventory, baseline = records
    current = copy.deepcopy(baseline)
    current["verdicts"][0]["revisions"][0]["rationale"] = "rewritten"
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(current))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--baseline", str(baseline_path)]) != 0
    assert not (tmp_path / "research/VERDICT_REGISTER.md").exists()


def test_cli_uses_tracked_head_history(records, tmp_path, monkeypatch):
    inventory, baseline = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    annotations_path = folder / "annotations.json"
    annotations_path.write_text(json.dumps(baseline))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    current = copy.deepcopy(baseline)
    current["verdicts"][0]["revisions"][0]["rationale"] = "rewritten"
    annotations_path.write_text(json.dumps(current))
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render"]) != 0
    assert not (tmp_path / "research/VERDICT_REGISTER.md").exists()


def test_inventory_history_rejects_deleted_and_reidentified_records(records):
    inventory, _ = records
    changed = copy.deepcopy(inventory)
    changed["records"].pop(1)
    assert register.validate_inventory_history(inventory, changed)
    changed = copy.deepcopy(inventory)
    changed["records"].pop(2)
    assert register.validate_inventory_history(inventory, changed)
    changed = copy.deepcopy(inventory)
    changed["records"][0]["source"] = "another.md"
    assert register.validate_inventory_history(inventory, changed)


def test_inventory_history_allows_documented_pending_progression(records):
    inventory, annotations = records
    baseline = copy.deepcopy(inventory)
    baseline["records"][1]["source"] = "source.md"
    updated = copy.deepcopy(baseline)
    updated["records"][1].update(disposition="excluded", reason="status only")
    assert not register.validate_inventory_history(baseline, updated)
    updated["records"][1].pop("reason")
    assert register.validate_inventory_history(baseline, updated)
    updated["records"][1]["disposition"] = "included"
    # Included progression must also have a verdict, checked by validate().
    for row in updated["records"]:
        row.setdefault("source", "source.md")
    assert register.validate(updated, annotations, Path("/nonexistent"))
    assert not register.validate_inventory_history(baseline, updated)


def test_revision_sources_are_required_and_rendered(records, tmp_path):
    inventory, annotations = records
    revision = annotations["verdicts"][0]["revisions"][0]
    del revision["sources"]
    assert register.validate(inventory, annotations, tmp_path)
    revision["sources"] = ["source.md"]
    assert not register.validate(inventory, annotations, tmp_path)
    assert "[evidence](../source.md)" in register.render(inventory, annotations)
    revision["sources"] = []
    assert register.validate(inventory, annotations, tmp_path)
    revision["sources"] = ["../escape.md"]
    assert register.validate(inventory, annotations, tmp_path)
    outside = tmp_path.parent / "revision-outside.md"
    outside.write_text("outside")
    (tmp_path / "revision-link.md").symlink_to(outside)
    revision["sources"] = ["revision-link.md"]
    assert register.validate(inventory, annotations, tmp_path)


def test_inventory_inclusion_requires_verdict(records, tmp_path):
    inventory, annotations = records
    baseline = copy.deepcopy(inventory)
    inventory["records"][1]["disposition"] = "included"
    assert not register.validate_inventory_history(baseline, inventory)
    assert any("i2" in error for error in register.validate(inventory, annotations, tmp_path))
    new = copy.deepcopy(annotations["verdicts"][0])
    new.update(verdict_id="v2", inventory_id="i2")
    new["revisions"][0]["annotation_revision_id"] = "r2"
    annotations["verdicts"].append(new)
    assert not register.validate(inventory, annotations, tmp_path)


def test_cli_checks_tracked_inventory_history(records, tmp_path, monkeypatch):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    inventory_path = folder / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    inventory["records"].pop(1)
    inventory_path.write_text(json.dumps(inventory))
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render"]) != 0
    assert not (tmp_path / "research/VERDICT_REGISTER.md").exists()
