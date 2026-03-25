"""Automatic state and results discovery for the Unified CLI."""

from pathlib import Path
from typing import Optional


def get_latest_research_run(results_dir: str = "results") -> Optional[Path]:
    """Find the most recent pipeline directory that contains a valid run_results.json."""
    base_path = Path(results_dir)
    if not base_path.exists():
        return None

    candidates = []
    # Look for both pipeline_ and isolated WFO runs if they exist
    for d in base_path.iterdir():
        if d.is_dir():
            res_json = d / "run_results.json"
            # We specifically want research runs, not recalibration runs here
            if res_json.exists() and "recalibration" not in d.name:
                candidates.append(res_json)

    if not candidates:
        return None

    # Sort chronologically by directory name if timestamped, or modified time
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def get_latest_production_weights(results_dir: str = "results") -> Optional[Path]:
    """Find the most recent portfolio_weights.json from recalibration runs."""
    base_path = Path(results_dir)
    if not base_path.exists():
        return None

    candidates = []
    for d in base_path.iterdir():
        if d.is_dir() and "recalibration" in d.name:
            weight_json = d / "portfolio_analysis" / "portfolio_weights.json"
            if weight_json.exists():
                candidates.append(weight_json)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]
