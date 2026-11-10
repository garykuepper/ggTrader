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


def test_pending_items_are_visible_without_invented_judgments(records):
    inventory, annotations = records
    inventory["records"][1].update(
        title="Strategy awaiting review",
        source="source.md",
        original_audit_verdict="NO-GO (audit transcription)",
    )
    output = register.render(inventory, annotations)
    assert "Strategy awaiting review" in output
    assert "NO-GO (audit transcription)" in output
    assert "NOT REVIEWED" in output
    assert "[source](../source.md)" in output


def test_replica_model_holdout_and_live_parity_are_distinct(records, tmp_path):
    inventory, annotations = records
    first = annotations["verdicts"][0]
    first.update(study_id="replica", specification_id="daily-model", stage_id="primary")
    for index, stage, original, protocol in (
        (2, "holdout", "INCONCLUSIVE", "STOPPED"),
        (3, "live-parity", "UNVERIFIED", "UNRESOLVED"),
    ):
        inventory["records"].append(
            {"inventory_id": f"replica-{index}", "disposition": "included", "source": "source.md"}
        )
        item = copy.deepcopy(first)
        item.update(
            inventory_id=f"replica-{index}",
            verdict_id=f"replica-v{index}",
            stage_id=stage,
            original_verdict=original,
            protocol_result=protocol,
        )
        item["revisions"][0]["annotation_revision_id"] = f"replica-r{index}"
        annotations["verdicts"].append(item)
    assert register.validate(inventory, annotations, tmp_path) == []
    result = register.render(inventory, annotations)
    assert all(stage in result for stage in ("primary", "holdout", "live-parity"))
    assert all(original in result for original in ("NO-GO", "INCONCLUSIVE", "UNVERIFIED"))


def test_same_day_append_is_current_even_if_revision_id_sorts_earlier(records):
    inventory, annotations = records
    revision = annotations["verdicts"][0]["revisions"][0]
    revision["annotation_revision_id"] = "z-first"
    annotations["verdicts"][0]["revisions"].append(
        dict(revision, annotation_revision_id="a-later", evidence="UNRESOLVED")
    )
    summary = register.render(inventory, annotations).split("## Detailed records")[0]
    assert "UNRESOLVED" in summary
    assert "a-later" in summary


def test_protocol_correction_is_append_only_and_visible(records, tmp_path):
    inventory, baseline = records
    updated = copy.deepcopy(baseline)
    earlier = updated["verdicts"][0]["revisions"][0]
    updated["verdicts"][0]["revisions"].append(
        dict(
            earlier,
            annotation_revision_id="correction",
            date="2026-02-01",
            protocol_result="UNRESOLVED",
        )
    )
    assert register.validate_history(baseline, updated) == []
    assert register.validate(inventory, updated, tmp_path) == []
    output = register.render(inventory, updated)
    assert "UNRESOLVED" in output.split("## Detailed records")[0]
    assert "NOT MET" in output.split("## Detailed records")[1]
    updated["verdicts"][0]["revisions"][-1]["protocol_result"] = "invalid"
    assert register.validate(inventory, updated, tmp_path)


def test_zero_pending_does_not_claim_complete_discovery_without_evidence(records, tmp_path):
    inventory, annotations = records
    inventory["records"] = [inventory["records"][0]]
    assert "INCOMPLETE" in register.render(inventory, annotations)
    inventory["coverage_complete"] = True
    assert register.validate(inventory, annotations, tmp_path)
    inventory["coverage_evidence"] = "source.md"
    assert register.validate(inventory, annotations, tmp_path) == []
    assert "REVIEWED" in register.render(inventory, annotations)


def test_parent_links_are_checked_and_summaries_are_labeled(records, tmp_path):
    inventory, annotations = records
    child = annotations["verdicts"][0]
    child["parent_verdict_id"] = "missing"
    assert register.validate(inventory, annotations, tmp_path)
    parent = copy.deepcopy(child)
    parent.update(verdict_id="parent", parent_verdict_id=None)
    parent["revisions"][0]["annotation_revision_id"] = "parent-revision"
    parent["revisions"][0]["record_role"] = "summary"
    annotations["verdicts"].append(parent)
    child["parent_verdict_id"] = "parent"
    assert not register.validate(inventory, annotations, tmp_path)
    result = register.render(inventory, annotations)
    assert "Summary, not an independent test" in result
    assert "Parent: parent" in result


def test_summary_role_survives_later_revision(records):
    inventory, annotations = records
    revisions = annotations["verdicts"][0]["revisions"]
    revisions[0]["record_role"] = "summary"
    later = dict(revisions[0], annotation_revision_id="later", date="2026-02-01")
    del later["record_role"]
    revisions.append(later)
    assert "Summary, not an independent test" in register.render(inventory, annotations)


def test_parent_cycles_and_cross_study_relationships_fail(records, tmp_path):
    inventory, annotations = records
    first = annotations["verdicts"][0]
    second = copy.deepcopy(first)
    second.update(verdict_id="v2", parent_verdict_id="v1")
    second["revisions"][0]["annotation_revision_id"] = "r2"
    annotations["verdicts"].append(second)
    first["parent_verdict_id"] = "v2"
    assert register.validate(inventory, annotations, tmp_path)
    del first["parent_verdict_id"]
    second["study_id"] = "other-study"
    assert register.validate(inventory, annotations, tmp_path)


def test_pending_source_review_is_linked_and_validated(records, tmp_path):
    inventory, annotations = records
    pending = inventory["records"][1]
    pending["review_source"] = "missing.md"
    assert register.validate(inventory, annotations, tmp_path)
    pending["review_source"] = "source.md"
    pending["review_note"] = "Weight-specific outcomes unavailable"
    assert not register.validate(inventory, annotations, tmp_path)
    rendered = register.render(inventory, annotations)
    assert "Weight-specific outcomes unavailable" in rendered
    assert "[review](../source.md)" in rendered


def test_history_check_rejects_committed_rewrite(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    path = folder / "annotations.json"
    path.write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "original"], cwd=tmp_path, check=True)
    annotations["verdicts"][0]["original_verdict"] = "GO"
    path.write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "rewrite"], cwd=tmp_path, check=True)
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--history"]) != 0
    assert "baseline verdict changed" in capsys.readouterr().out


def test_history_check_rejects_deleted_committed_register(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    path = folder / "annotations.json"
    path.write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "original"], cwd=tmp_path, check=True)
    path.unlink()
    subprocess.run(["git", "add", "-u"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "delete"], cwd=tmp_path, check=True)
    path.write_text(json.dumps(annotations))
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--history"]) != 0
    assert "Register error" in capsys.readouterr().out


def test_inventory_history_preserves_recorded_review_notes(records):
    inventory, _ = records
    inventory["records"][1].update(review_source="source.md", review_note="Unverified")
    changed = copy.deepcopy(inventory)
    changed["records"][1]["review_note"] = "Rewritten"
    assert register.validate_inventory_history(inventory, changed)
    changed["records"][1]["review_note"] = "Unverified"
    del changed["records"][1]["review_source"]
    assert register.validate_inventory_history(inventory, changed)


def test_history_check_refuses_shallow_clone(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    source = tmp_path / "source-repo"
    folder = source / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (source / "source.md").write_text("source")
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=source, check=True)
    (source / "extra").write_text("next")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "next"], cwd=source, check=True)
    clone = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth=1", source.as_uri(), str(clone)], check=True)
    monkeypatch.setattr(register, "ROOT", clone)
    assert register.main(["render", "--history"]) != 0
    assert "Full Git history is required" in capsys.readouterr().out


def test_history_check_compares_trusted_merge_base(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (tmp_path / "source.md").write_text("source")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    (tmp_path / "seed").write_text("seed")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "-qb", "feature"], cwd=tmp_path, check=True)
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "feature-register"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "-qb", "base", "HEAD^"], cwd=tmp_path, check=True)
    folder.mkdir(parents=True, exist_ok=True)
    base_inventory = copy.deepcopy(inventory)
    base_inventory["records"].append(
        {"inventory_id": "base-only", "disposition": "pending", "source": "source.md"}
    )
    (folder / "inventory.json").write_text(json.dumps(base_inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base-register"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "-q", "feature"], cwd=tmp_path, check=True)
    subprocess.run(["git", "merge", "-qs", "ours", "--no-edit", "base"], cwd=tmp_path, check=True)
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--history"]) == 0
    assert register.main(["render", "--history", "--base-ref", "base"]) != 0
    assert "base-only" in capsys.readouterr().out


def test_history_accepts_append_then_rejects_inventory_rewrite(
    records, tmp_path, monkeypatch, capsys
):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (tmp_path / "source.md").write_text("source")
    inventory_file = folder / "inventory.json"
    annotations_file = folder / "annotations.json"
    inventory_file.write_text(json.dumps(inventory))
    annotations_file.write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    revision = dict(
        annotations["verdicts"][0]["revisions"][0], annotation_revision_id="r2", date="2026-02-01"
    )
    annotations["verdicts"][0]["revisions"].append(revision)
    annotations_file.write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "append"], cwd=tmp_path, check=True)
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--history"]) == 0
    inventory["records"].pop(2)
    inventory_file.write_text(json.dumps(inventory))
    subprocess.run(["git", "add", str(inventory_file)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "remove"], cwd=tmp_path, check=True)
    assert register.main(["render", "--history"]) != 0
    assert "baseline inventory record removed" in capsys.readouterr().out


def test_trusted_base_null_documents_do_not_bootstrap(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (tmp_path / "source.md").write_text("source")
    inv = folder / "inventory.json"
    ann = folder / "annotations.json"
    inv.write_text("null")
    ann.write_text("null")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "invalid base"], cwd=tmp_path, check=True)
    inv.write_text(json.dumps(inventory))
    ann.write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "valid head"], cwd=tmp_path, check=True)
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--base-ref", "HEAD^"]) != 0
    assert "invalid baseline" in capsys.readouterr().out


def test_trusted_base_read_error_fails_closed(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (tmp_path / "source.md").write_text("source")
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "annotations.json").write_text(json.dumps(annotations))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    real_run = subprocess.run

    def fail_read(args, **kwargs):
        if args[:2] == ["git", "show"] and args[2].endswith(
            ":research/programme/verdict-framework/annotations.json"
        ):
            raise subprocess.CalledProcessError(128, args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(register, "ROOT", tmp_path)
    monkeypatch.setattr(register.subprocess, "run", fail_read)
    assert register.main(["render", "--base-ref", "HEAD"]) != 0
    assert "Register error" in capsys.readouterr().out


def test_trusted_base_bootstrap_partial_and_invalid_ref(records, tmp_path, monkeypatch, capsys):
    inventory, annotations = records
    folder = tmp_path / "research/programme/verdict-framework"
    folder.mkdir(parents=True)
    (tmp_path / "source.md").write_text("source")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "empty base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "empty-base"], cwd=tmp_path, check=True)
    inv = folder / "inventory.json"
    ann = folder / "annotations.json"
    inv.write_text(json.dumps(inventory))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "partial base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "partial-base"], cwd=tmp_path, check=True)
    ann.write_text(json.dumps(annotations))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "complete head"], cwd=tmp_path, check=True)
    monkeypatch.setattr(register, "ROOT", tmp_path)
    assert register.main(["render", "--base-ref", "empty-base"]) == 0
    assert register.main(["render", "--base-ref", "partial-base"]) != 0
    assert "incomplete register" in capsys.readouterr().out
    assert register.main(["render", "--base-ref", "no-such-ref"]) != 0
    assert "Register error" in capsys.readouterr().out
