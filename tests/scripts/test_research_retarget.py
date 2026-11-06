"""Tests for the research retarget tool (second move into sub-folders, reference rewrite)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from scripts.research_reorg import (
    DEFAULT_ALLOW_NEW,
    PINNED_PATHS,
    MoveRefusedError,
    rewrite_markdown,
)
from scripts.research_retarget import (
    MoveRow,
    apply_moves,
    check_moves,
    load_moves,
    main,
    rewrite_text,
    run_rewrite,
    snapshot_moves,
    update_manifest,
    verify_moved_tree,
)

S = "research/audits/s"
DEEPER = ("audits/s",)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def _write_moves(root: Path, pairs: list[tuple[str, str]], name: str = "moves.csv") -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("old_path", "new_path"))
    writer.writerows(pairs)
    path = root / name
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return path


def _rows(*pairs: tuple[str, str]) -> list[MoveRow]:
    return [MoveRow(old, new) for old, new in pairs]


SUB_MAP = {f"{S}/a.md": f"{S}/sub/a.md", f"{S}/b.md": f"{S}/sub/b.md"}


# --- rewrite_text -------------------------------------------------------------------------


def test_rewrite_text_reuses_or_matches_gen1_link_logic() -> None:
    text = (
        'See [b](b.md "T") and [c](c.md#frag) and [net](https://x.org/b.md) and [up](../README.md).'
    )
    expected = (
        'See [b](b.md "T") and [c](../c.md#frag) and [net](https://x.org/b.md) '
        "and [up](../../README.md)."
    )
    gen1_text, gen1_unresolved = rewrite_markdown(text, f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    new_text, unresolved = rewrite_text(text, f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert gen1_text == expected
    assert gen1_unresolved == []
    assert new_text == gen1_text
    assert unresolved == []


def test_token_with_line_suffix_is_rewritten() -> None:
    text = f"Cite `{S}/a.md:12-20` and {S}/a.md:5, 7 and ({S}/a.md) and {S}/a.md:9.\n"
    new_text, unresolved = rewrite_text(text, "docs/notes.md", "docs/notes.md", SUB_MAP)
    assert new_text == (
        f"Cite `{S}/sub/a.md:12-20` and {S}/sub/a.md:5, 7 and ({S}/sub/a.md) and {S}/sub/a.md:9.\n"
    )
    assert unresolved == []


def test_token_inside_longer_path_is_not_matched() -> None:
    text = f"old/{S}/a.md and {S}/a.md.bak and {S}/a.md/ and x{S}/a.md and {S}/a.mdx but {S}/a.md."
    new_text, _ = rewrite_text(text, "docs/notes.md", "docs/notes.md", SUB_MAP)
    assert new_text == (
        f"old/{S}/a.md and {S}/a.md.bak and {S}/a.md/ and x{S}/a.md and {S}/a.mdx but {S}/sub/a.md."
    )


def test_relative_link_from_moved_file_to_unmoved_file_gains_one_level() -> None:
    text = "[c](c.md) ![img](img/x.png) [same](sub/z.md) [top](../README.md)\n"
    new_text, unresolved = rewrite_text(text, f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert new_text == "[c](../c.md) ![img](../img/x.png) [same](z.md) [top](../../README.md)\n"
    assert unresolved == []
    # A link-looking string inside an inline code span is not a link: the same text
    # without backticks gets recomputed, so this proves the span is what protects it.
    plain, _ = rewrite_text("Write [c](c.md) here.\n", f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert plain == "Write [c](../c.md) here.\n"
    span, _ = rewrite_text("Write `[c](c.md)` here.\n", f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert span == "Write `[c](c.md)` here.\n"


def test_relative_link_between_two_moved_files_in_different_subfolders() -> None:
    mapping = {f"{S}/a.md": f"{S}/one/a.md", f"{S}/b.md": f"{S}/two/b.md"}
    from_a, _ = rewrite_text("[b](b.md#x)\n", f"{S}/a.md", f"{S}/one/a.md", mapping)
    from_b, _ = rewrite_text('[a](a.md "t")\n', f"{S}/b.md", f"{S}/two/b.md", mapping)
    assert from_a == "[b](../two/b.md#x)\n"
    assert from_b == '[a](../one/a.md "t")\n'


def test_relative_link_to_sibling_in_same_subfolder_is_unchanged_text() -> None:
    text = "[b](b.md) and [b again](./b.md#top)\n"
    new_text, unresolved = rewrite_text(text, f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert new_text == "[b](b.md) and [b again](b.md#top)\n"
    assert unresolved == []


def test_two_readme_files_do_not_collide() -> None:
    study_readme = {f"{S}/README.md": f"{S}/sub/README.md"}
    text = "[study](README.md) [root](../../README.md)\n"
    # Unmoved file: only the study README is in the mapping, so only it changes.
    unmoved, _ = rewrite_text(text, f"{S}/index.md", f"{S}/index.md", study_readme)
    assert unmoved == "[study](sub/README.md) [root](../../README.md)\n"
    # Moved file with neither README in the mapping: both are recomputed one level deeper.
    moved, _ = rewrite_text(text, f"{S}/a.md", f"{S}/sub/a.md", {f"{S}/a.md": f"{S}/sub/a.md"})
    assert moved == "[study](../README.md) [root](../../../README.md)\n"


def test_anchor_preserved() -> None:
    text = '[x](b.md#sec-1 "Title here") [y](<b.md#a>)\n'
    new_text, _ = rewrite_text(text, f"{S}/index.md", f"{S}/index.md", SUB_MAP)
    assert new_text == '[x](sub/b.md#sec-1 "Title here") [y](<sub/b.md#a>)\n'


def test_fenced_code_mentions_are_reported_not_rewritten() -> None:
    text = "\n".join(
        [
            f"Intro {S}/a.md here.",
            "```text",
            f"{S}/a.md",
            "[a](a.md)",
            "```",
            "Tail [a](a.md)",
        ]
    )
    new_text, unresolved = rewrite_text(text, f"{S}/index.md", f"{S}/index.md", SUB_MAP)
    assert new_text == "\n".join(
        [
            f"Intro {S}/sub/a.md here.",
            "```text",
            f"{S}/a.md",
            "[a](a.md)",
            "```",
            "Tail [a](sub/a.md)",
        ]
    )
    assert [(u.line, u.text, u.reason) for u in unresolved] == [(3, f"{S}/a.md", "fenced code")]


def test_directory_mention_is_untouched() -> None:
    text = f"Folder `{S}/` and {S} and {S}/a.md.\n"
    new_text, unresolved = rewrite_text(text, "docs/notes.md", "docs/notes.md", SUB_MAP)
    assert new_text == f"Folder `{S}/` and {S} and {S}/sub/a.md.\n"
    assert unresolved == []


def test_non_markdown_gets_token_rewrite_only() -> None:
    py = f'A = "{S}/a.md"  # see [a](a.md)\nB = "{S}/"\n'
    new_py, _ = rewrite_text(py, "scripts/tool.py", "scripts/tool.py", SUB_MAP)
    assert new_py == f'A = "{S}/sub/a.md"  # see [a](a.md)\nB = "{S}/"\n'
    sh = f"# [b](b.md)\ncat {S}/a.md\n"
    new_sh, _ = rewrite_text(sh, f"{S}/run.sh", f"{S}/sub/run.sh", SUB_MAP)
    assert new_sh == f"# [b](b.md)\ncat {S}/sub/a.md\n"


def test_reference_style_and_html_links_are_reported_not_guessed() -> None:
    text = '[ref]: b.md "T"\n<a href="b.md">b</a>\n'
    new_text, unresolved = rewrite_text(text, f"{S}/index.md", f"{S}/index.md", SUB_MAP)
    assert new_text == text
    assert [(u.line, u.reason) for u in unresolved] == [
        (1, "reference-style or html link"),
        (2, "reference-style or html link"),
    ]


def test_link_from_moved_file_to_old_tree_target_is_reported() -> None:
    text = "[old](../../../docs/research/gone.md)\n"
    new_text, unresolved = rewrite_text(text, f"{S}/a.md", f"{S}/sub/a.md", SUB_MAP)
    assert new_text == text
    assert [(u.line, u.reason) for u in unresolved] == [(1, "not in manifest")]
    # An unmoved file's old-tree link is not this tool's business.
    same, none = rewrite_text(text, f"{S}/index.md", f"{S}/index.md", SUB_MAP)
    assert same == text
    assert none == []


# --- check_moves / apply_moves ------------------------------------------------------------


def test_check_moves_accepts_clean_rows() -> None:
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"))
    assert check_moves(rows, {f"{S}/a.md"}, DEEPER) == []


def test_check_moves_rejects_basename_change() -> None:
    rows = _rows((f"{S}/a.md", f"{S}/sub/renamed.md"))
    problems = check_moves(rows, {f"{S}/a.md"}, DEEPER)
    assert problems == [f"basename changed: {S}/a.md -> {S}/sub/renamed.md"]


def test_check_moves_rejects_depth_over_limit() -> None:
    rows = _rows((f"{S}/a.md", f"{S}/x/y/a.md"))
    problems = check_moves(rows, {f"{S}/a.md"}, DEEPER)
    assert problems == [f"depth exceeds 5 components: {S}/x/y/a.md"]
    other = _rows(("research/audits/t/a.md", "research/audits/t/sub/a.md"))
    assert check_moves(other, {"research/audits/t/a.md"}, DEEPER) == [
        "depth exceeds 4 components: research/audits/t/sub/a.md"
    ]


def test_check_moves_rejects_existing_destination() -> None:
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/a.md"))
    tracked = {f"{S}/a.md", f"{S}/b.md", f"{S}/sub/a.md"}
    problems = check_moves(rows, tracked, DEEPER)
    assert f"destination already exists: {S}/sub/a.md" in problems
    assert f"destination used twice: {S}/sub/a.md" in problems


def test_check_moves_rejects_untracked_source_noop_and_outside_research() -> None:
    rows = _rows(
        (f"{S}/gone.md", f"{S}/sub/gone.md"), (f"{S}/a.md", f"{S}/a.md"), ("x/a.md", "y/a.md")
    )
    problems = check_moves(rows, {f"{S}/a.md", "x/a.md"}, DEEPER)
    assert problems == [
        f"source is not tracked: {S}/gone.md",
        f"source equals destination: {S}/a.md",
        "destination outside research/: y/a.md",
    ]


def test_apply_moves_dry_run_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    before = _git(root, "status", "--porcelain")
    pairs = apply_moves(_rows((f"{S}/a.md", f"{S}/sub/a.md")), root, True, DEEPER)
    assert pairs == [(f"{S}/a.md", f"{S}/sub/a.md")]
    assert f"PLAN {S}/a.md -> {S}/sub/a.md" in capsys.readouterr().out
    assert _git(root, "status", "--porcelain") == before == ""
    assert not (root / S / "sub").exists()


def test_apply_moves_renames_with_git_mv(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    pairs = apply_moves(
        _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/b.md")), root, False, DEEPER
    )
    assert len(pairs) == 2
    assert (root / S / "sub" / "a.md").read_text() == "A\n"
    assert not (root / S / "a.md").exists()
    status = sorted(_git(root, "status", "--porcelain").splitlines())
    assert status == [
        f"R  {S}/a.md -> {S}/sub/a.md",
        f"R  {S}/b.md -> {S}/sub/b.md",
    ]


def test_apply_moves_refuses_whole_run_on_one_problem(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/missing.md", f"{S}/sub/missing.md"))
    with pytest.raises(MoveRefusedError):
        apply_moves(rows, root, False, DEEPER)
    assert (root / S / "a.md").exists()
    assert _git(root, "status", "--porcelain") == ""


# --- run_rewrite --------------------------------------------------------------------------


def _rewrite_repo(tmp_path: Path) -> tuple[Path, list[MoveRow]]:
    files = {
        f"{S}/a.md": f"# A\n[b](b.md)\n[c](c.md)\nSee `{S}/b.md:3`.\n",
        f"{S}/b.md": "# B\n[a](a.md)\n",
        f"{S}/c.md": "# C\n[a](a.md)\n",
        f"{S}/_data.json": f'{{"p": "{S}/a.md"}}\n',
        "docs/notes.md": f"Look at {S}/a.md and [x](../{S}/b.md).\n",
        "scripts/tool.py": f'P = "{S}/c.md"\n',
        ".superpowers/note.md": f"{S}/a.md\n",
        "data/x.txt": f"{S}/a.md\n",
    }
    root = _make_repo(tmp_path, files)
    rows = _rows((f"{S}/a.md", f"{S}/one/a.md"), (f"{S}/b.md", f"{S}/two/b.md"))
    _write_moves(root, [(r.old, r.new) for r in rows])
    _git(root, "add", "moves.csv")
    _git(root, "commit", "-q", "-m", "moves")
    return root, rows


def test_run_rewrite_updates_links_and_tokens_after_move_and_is_idempotent(tmp_path: Path) -> None:
    root, rows = _rewrite_repo(tmp_path)
    apply_moves(rows, root, False, DEEPER)
    report_path = root / "report.json"
    dry = run_rewrite(root, rows, True, report_path, exclude=("moves.csv",))
    assert dry["status"] == "ok"
    assert (root / S / "c.md").read_text() == "# C\n[a](a.md)\n"  # dry run wrote nothing
    assert json.loads(report_path.read_text())["dry_run"] is True

    result = run_rewrite(root, rows, False, None, exclude=("moves.csv",))
    assert result["status"] == "ok"
    assert (root / S / "one" / "a.md").read_text() == (
        f"# A\n[b](../two/b.md)\n[c](../c.md)\nSee `{S}/two/b.md:3`.\n"
    )
    assert (root / S / "two" / "b.md").read_text() == "# B\n[a](../one/a.md)\n"
    assert (root / S / "c.md").read_text() == "# C\n[a](one/a.md)\n"
    assert (root / "docs/notes.md").read_text() == (
        f"Look at {S}/one/a.md and [x](../{S}/two/b.md).\n"
    )
    assert (root / "scripts/tool.py").read_text() == f'P = "{S}/c.md"\n'
    assert (root / S / "_data.json").read_text() == f'{{"p": "{S}/a.md"}}\n'
    assert (root / ".superpowers/note.md").read_text() == f"{S}/a.md\n"
    assert (root / "data/x.txt").read_text() == f"{S}/a.md\n"
    assert [c["file"] for c in result["changed"]] == [
        "docs/notes.md",
        f"{S}/c.md",
        f"{S}/one/a.md",
        f"{S}/two/b.md",
    ]
    assert result["rewrite_manifest"] == {
        "files": ["docs/notes.md", f"{S}/c.md", f"{S}/one/a.md", f"{S}/two/b.md"],
        "hand_edited": [],
    }

    again = run_rewrite(root, rows, False, None, exclude=("moves.csv",))
    assert again["changed"] == []
    assert again["rewrite_manifest"]["files"] == []


def test_frozen_paths_never_in_rewrite_scope(tmp_path: Path) -> None:
    pinned = PINNED_PATHS[0]
    files = {
        f"{S}/a.md": "# A\n",
        f"{S}/note.md": f"{S}/a.md\n",
        f"{S}/frozen.md": f"{S}/a.md\n",
        f"{S}/_res.json": f"{S}/a.md\n",
        "manifest.csv": (
            "current_path,proposed_path,folder,kind,notes\n"
            f'"{pinned}","{S}/frozen.md","{S}","report",""\n'
        ),
    }
    root = _make_repo(tmp_path, files)
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"))
    apply_moves(rows, root, False, DEEPER)
    result = run_rewrite(root, rows, False, None, manifest=root / "manifest.csv")
    assert (root / S / "frozen.md").read_text() == f"{S}/a.md\n"
    assert (root / S / "_res.json").read_text() == f"{S}/a.md\n"
    assert (root / S / "note.md").read_text() == f"{S}/sub/a.md\n"
    assert f"{S}/frozen.md" in result["excluded"]
    assert f"{S}/_res.json" in result["excluded"]
    assert [c["file"] for c in result["changed"]] == [f"{S}/note.md"]


# --- snapshot / verify --------------------------------------------------------------------


def _moved_repo(tmp_path: Path) -> tuple[Path, list[MoveRow], dict[str, str]]:
    files = {
        f"{S}/a.md": "A\n",
        f"{S}/_r.json": "{}\n",
        "docs/other.md": "other\n",
    }
    root = _make_repo(tmp_path, files)
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/_r.json", f"{S}/sub/_r.json"))
    before = snapshot_moves(rows, root)
    apply_moves(rows, root, False, DEEPER)
    return root, rows, before


def test_verify_accepts_pure_move(tmp_path: Path) -> None:
    root, rows, before = _moved_repo(tmp_path)
    assert verify_moved_tree(rows, before, set(), root) == []


def test_verify_moved_tree_flags_changed_moved_file(tmp_path: Path) -> None:
    root, rows, before = _moved_repo(tmp_path)
    (root / S / "sub" / "a.md").write_text("A changed\n")
    assert verify_moved_tree(rows, before, set(), root) == [
        f"moved file changed and not listed: {S}/sub/a.md"
    ]
    assert verify_moved_tree(rows, before, {f"{S}/sub/a.md"}, root) == []


def test_verify_moved_tree_flags_unlisted_edit(tmp_path: Path) -> None:
    root, rows, before = _moved_repo(tmp_path)
    (root / "docs/other.md").write_text("edited\n")
    (root / "docs/new.md").write_text("new\n")
    assert verify_moved_tree(rows, before, set(), root) == [
        "changed file is neither moved nor listed: docs/new.md",
        "changed file is neither moved nor listed: docs/other.md",
    ]
    assert verify_moved_tree(rows, before, {"docs/other.md"}, root, allow=("docs/new.md",)) == []


def test_verify_flags_listed_file_that_does_not_differ(tmp_path: Path) -> None:
    root, rows, before = _moved_repo(tmp_path)
    assert verify_moved_tree(rows, before, {"docs/other.md"}, root) == [
        "listed rewrite file does not differ from the base: docs/other.md"
    ]


def test_verify_flags_frozen_moved_file_that_changed(tmp_path: Path) -> None:
    root, rows, before = _moved_repo(tmp_path)
    (root / S / "sub" / "_r.json").write_text('{"x": 1}\n')
    assert verify_moved_tree(rows, before, set(), root) == [
        f"frozen moved file changed: {S}/sub/_r.json"
    ]
    # Listing a frozen file in the rewrite manifest does not excuse it.
    assert verify_moved_tree(rows, before, {f"{S}/sub/_r.json"}, root) == [
        f"frozen moved file changed: {S}/sub/_r.json",
        f"listed rewrite file is frozen: {S}/sub/_r.json",
    ]


def test_snapshot_records_sha256_per_old_path(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    before = snapshot_moves(_rows((f"{S}/a.md", f"{S}/sub/a.md")), root)
    assert before == {f"{S}/a.md": hashlib.sha256(b"A\n").hexdigest()}


# --- update-manifest ----------------------------------------------------------------------


def _manifest_files(root: Path) -> tuple[Path, Path, Path]:
    manifest = root / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        f'"docs/research/a.md","{S}/a.md","{S}","report","n, with comma"\n'
        f'"docs/research/c.md","{S}/c.md","{S}","report",""\n'
    )
    ledger = root / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "files": [
                    {"current_path": "docs/research/a.md", "proposed_path": f"{S}/a.md"},
                    {"current_path": "docs/research/c.md", "proposed_path": f"{S}/c.md"},
                ],
                "source_revision": "abc",
            },
            indent=2,
        )
        + "\n"
    )
    rewrite = root / "rewrite.json"
    rewrite.write_text(
        json.dumps(
            {"scope": "x", "hand_edited": [f"{S}/a.md"], "files": [f"{S}/a.md", f"{S}/c.md"]},
            indent=2,
        )
        + "\n"
    )
    return manifest, ledger, rewrite


def test_update_manifest_rewrites_proposed_paths(tmp_path: Path) -> None:
    root = tmp_path
    manifest, ledger, rewrite = _manifest_files(root)
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"))
    update_manifest(rows, manifest, ledger, rewrite)
    assert manifest.read_text() == (
        "current_path,proposed_path,folder,kind,notes\n"
        f'"docs/research/a.md","{S}/sub/a.md","{S}/sub","report","n, with comma"\n'
        f'"docs/research/c.md","{S}/c.md","{S}","report",""\n'
    )
    ledger_data = json.loads(ledger.read_text())
    assert [f["proposed_path"] for f in ledger_data["files"]] == [f"{S}/sub/a.md", f"{S}/c.md"]
    rewrite_data = json.loads(rewrite.read_text())
    assert rewrite_data["files"] == [f"{S}/c.md", f"{S}/sub/a.md"]
    assert rewrite_data["hand_edited"] == [f"{S}/sub/a.md"]
    snapshot = (manifest.read_bytes(), ledger.read_bytes(), rewrite.read_bytes())
    update_manifest(rows, manifest, ledger, rewrite)
    assert (manifest.read_bytes(), ledger.read_bytes(), rewrite.read_bytes()) == snapshot


def test_update_manifest_refuses_stale_row(tmp_path: Path) -> None:
    manifest, ledger, rewrite = _manifest_files(tmp_path)
    original = manifest.read_bytes()
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/zzz.md", f"{S}/sub/zzz.md"))
    with pytest.raises(ValueError, match=f"no manifest row has proposed_path {S}/zzz.md"):
        update_manifest(rows, manifest, ledger, rewrite)
    assert manifest.read_bytes() == original


# --- CLI ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command", [None, "check", "move", "rewrite", "snapshot", "verify", "update-manifest"]
)
def test_cli_help_exits_zero(command: str | None, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--help"] if command is None else [command, "--help"]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_load_moves_reads_csv_and_rejects_missing_columns(tmp_path: Path) -> None:
    good = _write_moves(tmp_path, [(f"{S}/a.md", f"{S}/sub/a.md")])
    assert load_moves(good) == [MoveRow(f"{S}/a.md", f"{S}/sub/a.md")]
    bad = tmp_path / "bad.csv"
    bad.write_text("old,new\nx,y\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_moves(bad)


def test_cli_check_and_snapshot_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    moves = _write_moves(root, [(f"{S}/a.md", f"{S}/sub/a.md")])
    assert (
        main(["check", "--moves", str(moves), "--root", str(root), "--allow-deeper", "audits/s"])
        == 0
    )
    out = tmp_path / "before.json"
    assert main(["snapshot", "--moves", str(moves), "--out", str(out), "--root", str(root)]) == 0
    assert list(json.loads(out.read_text())) == [f"{S}/a.md"]
    capsys.readouterr()
    bad = _write_moves(root, [(f"{S}/a.md", f"{S}/sub/renamed.md")], "bad.csv")
    assert (
        main(["check", "--moves", str(bad), "--root", str(root), "--allow-deeper", "audits/s"]) == 1
    )
    assert "PROBLEM basename changed" in capsys.readouterr().out


# --- review fixes -------------------------------------------------------------------------

CONSOLIDATED = "research/audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md"
CONSOLIDATED_NEW = CONSOLIDATED.replace("root-cause-2026-11-04/", "root-cause-2026-11-04/reviews/")


def test_update_manifest_skips_allow_new_file_without_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pre-move allow-list entry was removed with the sub-folder move; the filter keeps
    # the test independent of it, so only the destination entry may decide the skip.
    only_new = tuple(
        p
        for p in DEFAULT_ALLOW_NEW
        if p != "audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md"
    )
    monkeypatch.setattr("scripts.research_reorg.DEFAULT_ALLOW_NEW", only_new)
    manifest, ledger, rewrite = _manifest_files(tmp_path)
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (CONSOLIDATED, CONSOLIDATED_NEW))
    counts = update_manifest(rows, manifest, ledger, rewrite)
    assert counts["unlisted"] == 1
    assert counts["manifest"] == 1
    assert f"NOTE unlisted allow-new move: {CONSOLIDATED} -> {CONSOLIDATED_NEW}" in (
        capsys.readouterr().out
    )
    assert f'"{S}/sub/a.md"' in manifest.read_text()
    snapshot = (manifest.read_bytes(), ledger.read_bytes(), rewrite.read_bytes())
    update_manifest(rows, manifest, ledger, rewrite)
    assert (manifest.read_bytes(), ledger.read_bytes(), rewrite.read_bytes()) == snapshot
    # Only the allow-new list excuses a missing row: a stray destination is still refused.
    stray = _rows((CONSOLIDATED, f"{S}/sub/{CONSOLIDATED.rsplit('/', 1)[1]}"))
    with pytest.raises(ValueError, match="no manifest row"):
        update_manifest(stray, manifest, ledger, rewrite)


def test_tracked_moves_csv_elsewhere_is_never_rewritten(tmp_path: Path) -> None:
    csv_text = f"old_path,new_path\n{S}/a.md,{S}/sub/a.md\n"
    files = {
        f"{S}/a.md": "# A\n",
        "docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv": csv_text,
        "other/stray.csv": csv_text,
        "other/plain.csv": f"x,y\n{S}/a.md,1\n",
    }
    root = _make_repo(tmp_path, files)
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"))
    apply_moves(rows, root, False, DEEPER)
    result = run_rewrite(root, rows, False, None)
    assert (root / "other/stray.csv").read_text() == csv_text
    assert (
        root / "docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv"
    ).read_text() == (csv_text)
    assert (root / "other/plain.csv").read_text() == f"x,y\n{S}/sub/a.md,1\n"
    assert "other/stray.csv" in result["excluded"]
    assert "docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv" in result["excluded"]


def _failing_second_mv(monkeypatch: pytest.MonkeyPatch, fail_dest: str) -> None:
    real_run = subprocess.run

    def fake_run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if cmd[:3] == ["git", "mv", "--"] and cmd[4] == fail_dest:
            raise subprocess.CalledProcessError(1, cmd, stderr="simulated failure")
        return real_run(cmd, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr("scripts.research_retarget.subprocess.run", fake_run)


def test_apply_moves_undoes_completed_moves_when_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/b.md"))
    _failing_second_mv(monkeypatch, f"{S}/sub/b.md")
    with pytest.raises(MoveRefusedError, match=f"git mv {S}/b.md -> {S}/sub/b.md failed") as exc:
        apply_moves(rows, root, False, DEEPER)
    assert "undid 1 completed moves" in str(exc.value)
    assert _git(root, "status", "--porcelain") == ""
    assert (root / S / "a.md").read_text() == "A\n"
    assert not (root / S / "sub").exists()


def test_cli_move_failure_exits_one_with_recovery_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    moves = _write_moves(root, [(f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/b.md")])
    _failing_second_mv(monkeypatch, f"{S}/sub/b.md")
    argv = ["move", "--moves", str(moves), "--root", str(root), "--allow-deeper", "audits/s"]
    assert main(argv) == 1
    assert "git status" in capsys.readouterr().err


def test_cli_reports_bad_input_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    bad = tmp_path / "bad.csv"
    bad.write_text("old,new\nx,y\n")
    assert main(["check", "--moves", str(bad), "--root", str(root)]) == 1
    assert "FAIL" in capsys.readouterr().err
    moves = _write_moves(root, [(f"{S}/a.md", f"{S}/sub/a.md")])
    missing = str(tmp_path / "nope.csv")
    assert main(["rewrite", "--moves", str(moves), "--root", str(root), "--manifest", missing]) == 1
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "nope.csv" in err


def test_cli_verify_notes_allowed_frozen_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, rows, before = _moved_repo(tmp_path)
    moves = _write_moves(tmp_path, [(r.old, r.new) for r in rows])
    before_json = tmp_path / "before.json"
    before_json.write_text(json.dumps(before))
    rewrite_json = tmp_path / "rewrite.json"
    rewrite_json.write_text(json.dumps({"files": [], "hand_edited": []}))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("current_path,proposed_path,folder,kind,notes\n")
    (root / "research" / "MOVED.csv").write_text("old_path,new_path,renamed\n")
    argv = [
        "verify",
        "--moves",
        str(moves),
        "--root",
        str(root),
        "--before",
        str(before_json),
        "--rewrite-manifest",
        str(rewrite_json),
        "--manifest",
        str(manifest),
        "--allow",
        "research/MOVED.csv",
    ]
    assert main(argv) == 0
    assert "NOTE allowed frozen path: research/MOVED.csv" in capsys.readouterr().out


def _mkdir_fails_second_time(monkeypatch: pytest.MonkeyPatch) -> None:
    real_mkdir = Path.mkdir
    calls = {"n": 0}

    def fake_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated disk full")
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)


def test_apply_moves_undoes_completed_moves_when_mkdir_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    rows = _rows((f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/b.md"))
    _mkdir_fails_second_time(monkeypatch)
    with pytest.raises(MoveRefusedError, match="simulated disk full") as exc:
        apply_moves(rows, root, False, DEEPER)
    assert "undid 1 completed moves" in str(exc.value)
    assert _git(root, "status", "--porcelain") == ""
    assert not (root / S / "sub").exists()


def test_cli_move_mkdir_failure_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    moves = _write_moves(root, [(f"{S}/a.md", f"{S}/sub/a.md"), (f"{S}/b.md", f"{S}/sub/b.md")])
    _mkdir_fails_second_time(monkeypatch)
    argv = ["move", "--moves", str(moves), "--root", str(root), "--allow-deeper", "audits/s"]
    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("FAIL ")
    assert _git(root, "status", "--porcelain") == "?? moves.csv\n"


def test_undo_removes_created_folders_deepest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n", f"{S}/b.md": "B\n"})
    rows = _rows((f"{S}/a.md", f"{S}/x/y/a.md"), (f"{S}/b.md", f"{S}/sub/b.md"))
    _failing_second_mv(monkeypatch, f"{S}/sub/b.md")
    with pytest.raises(MoveRefusedError):
        apply_moves(rows, root, False, (*DEEPER, "audits/s"))
    assert not (root / S / "x").exists()
    assert not (root / S / "sub").exists()


def test_load_moves_rejects_short_and_long_rows(tmp_path: Path) -> None:
    short = tmp_path / "short.csv"
    short.write_text("old_path,new_path\nresearch/a.md\n")
    with pytest.raises(ValueError, match=r"short\.csv: row 1 needs old_path and new_path"):
        load_moves(short)
    long = tmp_path / "long.csv"
    long.write_text("old_path,new_path\na,b\nc,d,e\n")
    with pytest.raises(ValueError, match=r"long\.csv: row 2 needs old_path and new_path"):
        load_moves(long)


def test_cli_short_moves_row_prints_fail_not_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_repo(tmp_path, {f"{S}/a.md": "A\n"})
    short = tmp_path / "short.csv"
    short.write_text("old_path,new_path\nresearch/a.md\n")
    assert main(["check", "--moves", str(short), "--root", str(root)]) == 1
    assert "FAIL" in capsys.readouterr().err
