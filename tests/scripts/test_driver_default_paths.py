"""Check that research driver scripts read and write inside the per-study research/ tree.

The drivers are not run: most of them fetch data or run walk-forward optimisation on
import of ``main``. The checks read each script's source text, and import only the two
modules whose top level holds constants alone (``lens2_build``, ``wfo_fold_attribution``).
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv"

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


@pytest.mark.parametrize("script", CHANGED_SCRIPTS)
def test_default_paths_point_into_research_tree(script: str) -> None:
    text = (ROOT / script).read_text()
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


def _load_script(name: str) -> ModuleType:
    """Import a script whose top level only defines constants and functions."""
    spec = importlib.util.spec_from_file_location(f"_driver_{name}", ROOT / "scripts" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lens2_resolves_every_inventory_artifact() -> None:
    lens2 = _load_script("lens2_build.py")
    inventory = lens2.load(lens2.INVENTORY_JSON)
    listed = {r["raw_artifact"] for r in inventory if r.get("raw_artifact")}
    for art in listed:
        loc = lens2.ARTIFACT_LOCATIONS.get(Path(art).name, art)
        if loc.startswith("results/"):
            continue  # gitignored run output; absent on a fresh clone by design
        assert (ROOT / loc).is_file(), f"{art} -> {loc} missing"
    for key in lens2.HANDLERS:
        if not key.startswith("results/"):
            assert key in lens2.ARTIFACT_LOCATIONS.values(), key


def test_fold_attribution_sources_exist() -> None:
    attribution = _load_script("wfo_fold_attribution.py")
    for filename, _ in attribution.RUN_SOURCES:
        path = attribution.ARTIFACT_PATHS[filename]
        assert Path(path).name == filename
        assert (ROOT / path).is_file(), path
    assert attribution.DEFAULT_OUT.parent.is_dir()
