"""Latest-research-run discovery for the unified CLI (DB-backed)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LatestResearchRun:
    """A pointer to the most recent research run that the live trader / backtest
    CLIs use to load per-coin optimized parameters from.

    Source of truth is the ``runs`` table in TimescaleDB. ``run_dir`` is kept as
    an optional reference to the on-disk artifacts (markdown report, plots) when
    they exist; the live trader doesn't need it.
    """

    run_id: str
    per_coin: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_dir: Optional[Path] = None

    def __fspath__(self) -> str:
        return str(self.run_dir) if self.run_dir else self.run_id

    def __str__(self) -> str:
        return self.__fspath__()


def _per_coin_from_strategy_params(strategy_params: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize whatever shape ``runs.strategy_params`` holds to a flat per-coin dict.

    Accepts both the historical layout (``{"per_coin": {...}}``) and the simpler
    flat layout we now write.
    """
    if not isinstance(strategy_params, dict):
        return {}
    if "per_coin" in strategy_params and isinstance(strategy_params["per_coin"], dict):
        return strategy_params["per_coin"]
    return strategy_params


def _per_coin_from_legacy_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read per-coin results from a legacy ``run_results.json`` on disk."""
    with open(path) as f:
        data = json.load(f)
    sp = data.get("strategy_parameters", {})
    return sp.get("per_coin", data.get("per_coin_results", {}))


def from_run_dir(run_dir: Path | str) -> LatestResearchRun:
    """Build a LatestResearchRun from a results/research/<run>/ directory."""
    run_dir = Path(run_dir)
    res_json = run_dir / "run_results.json" if run_dir.is_dir() else run_dir
    per_coin: Dict[str, Dict[str, Any]] = {}
    run_id = run_dir.name
    if res_json.exists():
        try:
            with open(res_json) as f:
                data = json.load(f)
            per_coin = _per_coin_from_legacy_json(res_json)
            run_id = data.get("run_id", run_id)
        except Exception:
            pass
    return LatestResearchRun(
        run_id=run_id, per_coin=per_coin, run_dir=run_dir if run_dir.is_dir() else None
    )


def get_latest_research_run(results_dir: str = "results") -> Optional[LatestResearchRun]:
    """Return the latest crypto research run.

    Queries the ``runs`` table first; falls back to scanning ``results/research/``
    if the DB has no matching row (e.g. very old runs not yet backfilled).
    """
    # 1) DB lookup
    try:
        from sqlalchemy import text
        from ggTrader.utils.result_db_manager import ResultDBManager

        m = ResultDBManager()
        sql = """
            SELECT run_id, strategy_params, run_dir
            FROM runs
            WHERE run_type = 'research'
              AND COALESCE(status, 'success') = 'success'
              AND strategy_params IS NOT NULL
              AND COALESCE(asset_class, 'crypto') = 'crypto'
            ORDER BY timestamp DESC LIMIT 1
        """
        with m.engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        if row is not None:
            run_id, strategy_params, run_dir = row
            run_dir_path = Path(run_dir) if run_dir else None
            return LatestResearchRun(
                run_id=run_id,
                per_coin=_per_coin_from_strategy_params(strategy_params),
                run_dir=run_dir_path if (run_dir_path and run_dir_path.exists()) else None,
            )
    except Exception as e:
        print(f"[state_manager] DB lookup failed ({e!r}); falling back to disk")

    # 2) Disk fallback (legacy)
    base_path = Path(results_dir)
    if not base_path.exists():
        return None
    candidates: list[Path] = []
    research_path = base_path / "research"
    if research_path.exists():
        for d in research_path.iterdir():
            if d.is_dir() and (d / "run_results.json").exists():
                candidates.append(d / "run_results.json")
    for d in base_path.iterdir():
        if d.is_dir() and d.name not in {"research", "backtest", "production", "trade", "reports"}:
            res = d / "run_results.json"
            if res.exists() and "recalibration" not in d.name:
                candidates.append(res)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.parent.name), reverse=True)
    return from_run_dir(candidates[0].parent)


def get_latest_production_weights(results_dir: str = "results") -> Optional[Path]:
    """Return the most recent ``portfolio_weights.json`` (still file-based)."""
    base_path = Path(results_dir)
    if not base_path.exists():
        return None
    candidates: list[Path] = []
    prod_path = base_path / "production"
    if prod_path.exists():
        for d in prod_path.iterdir():
            wp = d / "portfolio_analysis" / "portfolio_weights.json"
            if d.is_dir() and wp.exists():
                candidates.append(wp)
    for d in base_path.iterdir():
        if (
            d.is_dir()
            and d.name not in {"research", "backtest", "production", "trade", "reports"}
            and "recalibration" in d.name
        ):
            wp = d / "portfolio_analysis" / "portfolio_weights.json"
            if wp.exists():
                candidates.append(wp)
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.parent.parent.name), reverse=True)
    return candidates[0]
