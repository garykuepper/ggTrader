"""Tests for the vectorbt upgrade-watch release picker (pure function, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts import vbt_latest

pytestmark = pytest.mark.tooling

NOW = datetime(2026, 11, 20, 6, 0, tzinfo=UTC)


def _release(days_ago: float, yanked: bool = False) -> list[dict]:
    when = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return [{"upload_time_iso_8601": when, "yanked": yanked}]


RELEASES = {
    "1.1.2": _release(40),
    "1.2.0": _release(30),
    "1.2.1": _release(5),
    "1.3.0": _release(1),
    "1.3.0rc1": _release(10),
    "1.2.2": _release(8, yanked=True),
}
RUST = {"1.1.2", "1.2.0", "1.2.1", "1.3.0", "1.3.0rc1", "1.2.2"}


def test_picks_the_newest_release_older_than_the_cooldown() -> None:
    assert vbt_latest.pick_version(RELEASES, RUST, "1.2.0", now=NOW, cooldown_days=3) == "1.2.1"


def test_a_release_inside_the_cooldown_is_not_taken_yet() -> None:
    assert vbt_latest.pick_version(RELEASES, RUST, "1.2.1", now=NOW, cooldown_days=3) == ""


def test_long_cooldown_holds_back_newer_releases() -> None:
    assert vbt_latest.pick_version(RELEASES, RUST, "1.1.2", now=NOW, cooldown_days=20) == "1.2.0"


def test_prereleases_and_yanked_releases_are_ignored() -> None:
    assert vbt_latest.pick_version(RELEASES, RUST, "1.2.1", now=NOW, cooldown_days=0) == "1.3.0"
    only = {"1.2.2": _release(8, yanked=True), "1.2.3rc1": _release(8)}
    assert (
        vbt_latest.pick_version(only, {"1.2.2", "1.2.3rc1"}, "1.2.0", now=NOW, cooldown_days=0)
        == ""
    )


def test_the_pair_is_atomic_no_rust_wheel_means_no_upgrade() -> None:
    assert (
        vbt_latest.pick_version(RELEASES, RUST - {"1.2.1"}, "1.2.0", now=NOW, cooldown_days=3) == ""
    )


def test_nothing_newer_than_the_lock_means_no_upgrade() -> None:
    assert vbt_latest.pick_version(RELEASES, RUST, "1.3.0", now=NOW, cooldown_days=0) == ""


def test_force_skips_the_cooldown_but_not_the_pair_rule() -> None:
    assert (
        vbt_latest.pick_version(RELEASES, RUST, "1.2.0", now=NOW, cooldown_days=3, force="1.3.0")
        == "1.3.0"
    )
    with pytest.raises(ValueError, match="vectorbt-rust"):
        vbt_latest.pick_version(
            RELEASES, RUST - {"1.3.0"}, "1.2.0", now=NOW, cooldown_days=3, force="1.3.0"
        )
    with pytest.raises(ValueError, match="not on PyPI"):
        vbt_latest.pick_version(RELEASES, RUST, "1.2.0", now=NOW, cooldown_days=3, force="9.9.9")


def test_a_rejected_version_is_not_proposed_and_neither_is_anything_older() -> None:
    """P8 R12: after the newest release was rejected the picker used to offer an older one."""
    picked = vbt_latest.pick_version(
        RELEASES, RUST, "1.2.0", now=NOW, cooldown_days=0, rejected={"1.3.0"}
    )
    assert picked == ""  # 1.2.1 is older than the rejected 1.3.0: not evaluated on its own


def test_a_release_newer_than_the_rejected_one_is_still_evaluated() -> None:
    releases = {**RELEASES, "1.3.1": _release(2)}
    picked = vbt_latest.pick_version(
        releases, RUST | {"1.3.1"}, "1.2.0", now=NOW, cooldown_days=0, rejected={"1.3.0"}
    )
    assert picked == "1.3.1"


def test_a_rejected_older_version_does_not_block_a_newer_candidate() -> None:
    picked = vbt_latest.pick_version(
        RELEASES, RUST, "1.1.2", now=NOW, cooldown_days=0, rejected={"1.2.0"}
    )
    assert picked == "1.3.0"


def test_the_reported_case_lock_1_1_1_with_1_2_0_rejected_proposes_nothing_older() -> None:
    """Verified in the P8 drill against live PyPI: it printed version=1.1.2."""
    releases = {"1.1.1": _release(60), "1.1.2": _release(40), "1.2.0": _release(30)}
    rust = set(releases)
    picked = vbt_latest.pick_version(
        releases, rust, "1.1.1", now=NOW, cooldown_days=3, rejected={"1.2.0"}
    )
    assert picked == ""
    forced = vbt_latest.pick_version(
        releases, rust, "1.1.1", now=NOW, cooldown_days=3, rejected={"1.2.0"}, force="1.1.2"
    )
    assert forced == "1.1.2"  # the owner can still ask for the older one explicitly


def test_main_reads_the_rejected_file_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "vectorbt"\nversion = "1.1.1"\n')
    rejected = tmp_path / "rejected.txt"
    rejected.write_text("1.2.0  # winner flips, https://example.invalid/issue\n")
    releases = {"1.1.2": _release(40), "1.2.0": _release(30)}
    monkeypatch.setattr(
        vbt_latest,
        "_fetch",
        lambda name: {"releases": releases if name == "vectorbt" else dict.fromkeys(releases, [])},
    )
    argv = ["--lock", str(lock), "--rejected-file", str(rejected)]
    assert vbt_latest.main(argv) == 0
    assert capsys.readouterr().out.strip() == "version="


def test_force_overrides_the_rejected_list() -> None:
    picked = vbt_latest.pick_version(
        RELEASES, RUST, "1.2.0", now=NOW, cooldown_days=0, force="1.3.0", rejected={"1.3.0"}
    )
    assert picked == "1.3.0"


def test_rejected_file_ignores_comments_and_blank_lines(tmp_path) -> None:
    f = tmp_path / "rejected.txt"
    f.write_text("# why\n1.2.1  # winner flips in 3 folds\n\n1.2.2\n")
    assert vbt_latest.read_rejected(f) == {"1.2.1", "1.2.2"}
    assert vbt_latest.read_rejected(tmp_path / "none.txt") == set()


def test_locked_version_is_read_from_uv_lock(tmp_path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "vectorbt"\nversion = "1.2.0"\n\n'
        '[[package]]\nname = "x"\nversion = "9"\n'
    )
    assert vbt_latest.locked_version(lock) == "1.2.0"
    with pytest.raises(ValueError):
        vbt_latest.locked_version(tmp_path / "missing.lock")


def test_help_works(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        vbt_latest.main(["--help"])
    assert info.value.code == 0
    assert "cooldown" in capsys.readouterr().out
