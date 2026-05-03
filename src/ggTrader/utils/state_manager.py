"""Automatic state and results discovery for the Unified CLI."""

import json
from pathlib import Path
from typing import Optional

VALID_ASSET_CLASSES = ("crypto", "stocks")


def _read_asset_class(run_results_path: Path) -> str:
    """Extract asset_class from a run_results.json with legacy fallback.

    Order of precedence:
      1. Top-level "asset_class" (current writer — see ResultsManager)
      2. configuration._raw_config.ASSET_CLASS (older runs from after stocks support landed)
      3. "crypto" (truly legacy runs, predating multi-asset)
    """
    try:
        with open(run_results_path) as f:
            data = json.load(f)
    except Exception:
        return "crypto"

    if isinstance(data.get("asset_class"), str):
        return data["asset_class"]

    raw = data.get("configuration", {}).get("_raw_config", {})
    if isinstance(raw.get("ASSET_CLASS"), str):
        return raw["ASSET_CLASS"]

    return "crypto"


def get_latest_research_run(
    results_dir: str = "results",
    asset_class: Optional[str] = None,
) -> Optional[Path]:
    """Find the most recent run_results.json, optionally filtered by asset_class.

    When ``asset_class`` is provided, runs with a different asset class are skipped
    so the live trader (or backtest/production CLIs) can't accidentally pick up a
    research run from a different asset universe.
    """
    base_path = Path(results_dir)
    if not base_path.exists():
        return None

    candidates = []

    # Check new structure (results/research/)
    research_path = base_path / "research"
    if research_path.exists():
        for d in research_path.iterdir():
            if d.is_dir():
                res_json = d / "run_results.json"
                if res_json.exists():
                    candidates.append(res_json)

    # Check legacy structure (results root)
    for d in base_path.iterdir():
        if d.is_dir() and d.name not in ["research", "backtest", "production", "trade"]:
            res_json = d / "run_results.json"
            if res_json.exists() and "recalibration" not in d.name:
                candidates.append(res_json)

    if not candidates:
        return None

    if asset_class is not None:
        if asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(
                f"Invalid asset_class {asset_class!r}; expected one of {VALID_ASSET_CLASSES}"
            )
        candidates = [c for c in candidates if _read_asset_class(c) == asset_class]
        if not candidates:
            return None

    # Sort by mtime, then by parent directory name as tiebreaker (handles identical mtimes)
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.parent.name), reverse=True)
    return candidates[0]


def get_latest_production_weights(
    results_dir: str = "results",
    asset_class: Optional[str] = None,
) -> Optional[Path]:
    """Find the most recent portfolio_weights.json, optionally filtered by asset_class.

    The asset_class is read from the sibling run_results.json (one directory up
    from the portfolio_analysis/ folder).
    """
    base_path = Path(results_dir)
    if not base_path.exists():
        return None

    candidates = []

    # Check new structure
    prod_path = base_path / "production"
    if prod_path.exists():
        for d in prod_path.iterdir():
            if d.is_dir():
                weight_json = d / "portfolio_analysis" / "portfolio_weights.json"
                if weight_json.exists():
                    candidates.append(weight_json)

    # Check legacy structure
    for d in base_path.iterdir():
        if d.is_dir() and d.name not in ["research", "backtest", "production", "trade"]:
            if "recalibration" in d.name:
                weight_json = d / "portfolio_analysis" / "portfolio_weights.json"
                if weight_json.exists():
                    candidates.append(weight_json)

    if not candidates:
        return None

    if asset_class is not None:
        if asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(
                f"Invalid asset_class {asset_class!r}; expected one of {VALID_ASSET_CLASSES}"
            )

        def _matches(weight_path: Path) -> bool:
            sibling = weight_path.parent.parent / "run_results.json"
            if not sibling.exists():
                # No sibling run_results — assume crypto for legacy compatibility
                return asset_class == "crypto"
            return _read_asset_class(sibling) == asset_class

        candidates = [c for c in candidates if _matches(c)]
        if not candidates:
            return None

    # Sort by mtime, then by parent directory name as tiebreaker (handles identical mtimes)
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.parent.parent.name), reverse=True)
    return candidates[0]


def validate_results_asset_class(run_results_path: Path, expected: str) -> None:
    """Raise SystemExit with a clear message if a run_results.json doesn't match
    the expected asset class.

    Used by CLIs that accept ``--results PATH`` to refuse cross-class loads
    (e.g. trying to start a crypto trader against a stocks research run).
    """
    if expected not in VALID_ASSET_CLASSES:
        raise ValueError(
            f"Invalid expected asset_class {expected!r}; "
            f"expected one of {VALID_ASSET_CLASSES}"
        )

    path = Path(run_results_path)
    if not path.exists():
        raise SystemExit(f"ERROR: --results path does not exist: {path}")

    actual = _read_asset_class(path)
    if actual != expected:
        raise SystemExit(
            f"ERROR: --results points to a {actual!r} run, but --asset-class is {expected!r}.\n"
            f"  Path: {path}\n"
            f"  Either pass --asset-class {actual}, or pick a {expected} run."
        )
