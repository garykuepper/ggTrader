"""Check that research driver scripts read and write inside the per-study research/ tree.

The drivers are not run: most of them fetch data or run walk-forward optimisation on
import of ``main``. The checks read each script's source text. The fold-attribution driver's
own source check lives with it in research/programme/wfo-methods/code/tests, and the Lens 2
inventory check in research/audits/root-cause-2026-11-04/code/tests.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv"
#: Closed-study drivers that need the legacy engine were archived here in P5c (they run only at
#: git tag legacy-engine-vbt0285); their default paths are still checked.
ARCHIVE = ROOT / "research/programme/vectorbt-engine/legacy-scripts"
CODEMOVED = ROOT / "research/CODEMOVED.csv"

#: Scripts re-pointed from docs/research/ to the per-study research/ folders (2026-11-05).
CHANGED_SCRIPTS = (
    "scripts/anchor_fix_reproduction_wfo.py",
    "scripts/audit_reruns_wfo.py",
    "scripts/century_leverage.py",
    "scripts/century_leverage_a23_x.py",
    "scripts/century_leverage_calibrate.py",
    "scripts/core_spy_blend.py",
    "scripts/cross_asset_sleeve_wfo.py",
    "scripts/factor_french_screen.py",
    "scripts/fomc_clustered_event_study.py",
    "scripts/ic_kelly_rebaseline_wfo.py",
    "scripts/lens2_build.py",
    "scripts/leveraged_trend_research.py",
    "scripts/live_indicator_warmup.py",
    "scripts/momentum_french_screen.py",
    "scripts/month_end_treasury_wfo.py",
    "scripts/net_issuance_screen.py",
    "scripts/pairs_stat_arb_research.py",
    "scripts/position_sizing_regimes.py",
    "scripts/position_sizing_wfo.py",
    "scripts/production_replica_audit.py",
    "scripts/replica_parity_check.py",
    "scripts/run_position_sizing_wfo_scheduled.sh",
    "scripts/shadow_cross_asset.py",
    "scripts/static_leverage_aftertax.py",
    "scripts/wfo_fold_attribution.py",
    "scripts/wfo_policy_synthetic.py",
)

#: Outputs a driver names that do not exist yet (the file name is built at run time).
#: Each must sit in an existing manifest study folder.
DOCUMENTED_NEW_OUTPUTS = {
    # Date-stamped re-run output; the 2026-10-19 run lives in the same folder.
    "research/audits/production-replica/_audit_reruns_<YYYYMMDD>.json",
}

#: A research/ path in source text, not preceded by another path segment (so the gitignored
#: results/research/... run folders are skipped).
RESEARCH_PATH_RE = re.compile(r"(?<![\w/.-])research/[\w./<>{}%:-]*")
#: Old location in any spelling: a literal path or a split ``"docs" / "research"`` join.
OLD_LOCATION_RE = re.compile(r"docs/research|[\"']docs[\"']\s*/\s*[\"']research")


def _manifest_folders() -> set[str]:
    with MANIFEST.open(newline="") as fh:
        return {row["folder"] for row in csv.DictReader(fh)}


def _research_paths(text: str) -> list[str]:
    """Return the research/ paths cited in ``text``, without line suffixes or punctuation."""
    paths = []
    for raw in RESEARCH_PATH_RE.findall(text):
        path = raw
        for _ in range(2):
            path = re.sub(r":[\d-]+$", "", path).rstrip(".,;:)")
        paths.append(path)
    return paths


def _in_manifest_folder(path: str, folders: set[str]) -> bool:
    """True if ``path`` is research/ itself, a manifest folder, or a file in one."""
    stripped = path.rstrip("/")
    if stripped == "research" or stripped in folders:
        return True
    if path.endswith("/"):
        # A folder cited on its own (e.g. a path wrapped across a docstring line).
        return any(folder.startswith(path) for folder in folders)
    parent = str(Path(stripped).parent)
    return parent == "research" or parent in folders


def _moved_to(script: str) -> Path | None:
    """Where research/CODEMOVED.csv says a moved script now lives (``None`` if it never moved)."""
    if not CODEMOVED.is_file():
        return None
    with CODEMOVED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["old_path"] == script and row["new_path"].endswith((".py", ".sh")):
                return ROOT / row["new_path"]
    return None


def _script_path(script: str) -> Path:
    """The driver where it lives now: scripts/, its CODEMOVED new path, then the legacy archive."""
    live = ROOT / script
    if live.exists():
        return live
    moved = _moved_to(script)
    if moved is not None and moved.exists():
        return moved
    return ARCHIVE / Path(script).name


@pytest.mark.parametrize("script", CHANGED_SCRIPTS)
def test_default_paths_point_into_research_tree(script: str) -> None:
    text = _script_path(script).read_text()
    assert not OLD_LOCATION_RE.search(text), f"{script} still names docs/research"

    paths = _research_paths(text)
    assert paths, f"{script} names no research/ path"
    folders = _manifest_folders()
    for path in paths:
        assert _in_manifest_folder(path, folders), f"{script}: {path} is outside a study folder"
        if path in DOCUMENTED_NEW_OUTPUTS:
            assert (ROOT / path).parent.is_dir(), f"{script}: folder for {path} missing"
        else:
            assert (ROOT / path).exists(), f"{script}: {path} does not exist"
