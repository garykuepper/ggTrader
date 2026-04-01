"""Automatic state and results discovery for the Unified CLI."""

from pathlib import Path
from typing import Optional


def get_latest_research_run(results_dir: str = "results") -> Optional[Path]:
    """Find the most recent pipeline directory that contains a valid run_results.json."""
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

    # Sort by mtime, then by parent directory name as tiebreaker (handles identical mtimes)
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.parent.name), reverse=True)
    return candidates[0]


def get_latest_production_weights(results_dir: str = "results") -> Optional[Path]:
    """Find the most recent portfolio_weights.json from recalibration runs."""
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

    # Sort by mtime, then by parent directory name as tiebreaker (handles identical mtimes)
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.parent.parent.name), reverse=True)
    return candidates[0]
