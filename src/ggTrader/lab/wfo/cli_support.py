"""Glue between ``ggt lab --engine vbt`` and the vectorbt-engine walk-forward.

Loads the bars read-only (frozen snapshot by default), builds the point-in-time mask and runs
:func:`~ggTrader.lab.wfo.run.run_wfo`; the legacy CLI path is untouched. Every result is
written as JSON with a provenance block (``utils.results.write_result_json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ggTrader.data.bars import Bars, legacy_exclusive_end, load_bars
from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import inputs
from ggTrader.lab.wfo.report import WfoResult, result_payload
from ggTrader.lab.wfo.run import run_wfo

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "lab"


def load_run_bars(
    universes: List[str],
    cfg: LabConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    snapshot: str | None = None,
    source: str = "snapshot",
) -> Bars:
    """Bars for the union of ``universes`` plus SPY, over the warm-up and evaluation window.

    The window ends on the legacy exclusive end so a run is comparable with a legacy run of the
    same ``--eval-end``. ``max_stocks`` truncates each universe alphabetically, as the legacy CLI.
    """
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    members: set[str] = {"SPY"}
    for universe in universes:
        names = equity_universe_between(eval_start, eval_end, universe=universe)
        if cfg.max_stocks and len(names) > cfg.max_stocks:
            names = names[: cfg.max_stocks]
        members.update(names)
    return load_bars(
        sorted(members),
        start=str(data_start.date()),
        end=legacy_exclusive_end(eval_end),
        source=source,
        snapshot_path=snapshot,
    )


def sleeve_inputs(
    bars: Bars, universe: str, cfg: LabConfig, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> tuple[Bars, pd.DataFrame]:
    """The sleeve's bars (SPY removed) and its daily eligibility mask."""
    names = equity_universe_between(eval_start, eval_end, universe=universe)
    if cfg.max_stocks and len(names) > cfg.max_stocks:
        names = names[: cfg.max_stocks]
    symbols = [s for s in names if s in set(bars.symbols)]
    sleeve = inputs.subset_bars(bars, symbols)
    mask = inputs.universe_mask(
        sleeve.close, universe, min_history_bars=cfg.min_history_bars, max_stocks=cfg.max_stocks
    )
    return sleeve, mask


def run_vbt_wfo(
    strategy: str,
    universe: str,
    cfg: LabConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    *,
    market: str,
    base_config: Dict[str, Any],
    overrides: Dict[str, list] | None = None,
    snapshot: str | None = None,
    result_json: str | None = None,
) -> WfoResult | str:
    """``ggt lab --wfo --engine vbt``: load, mask, run, print and write the provenance JSON."""
    from ggTrader.lab.strategies import registry

    try:
        instance = registry.get(strategy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cls = type(instance)
    bars = load_run_bars([universe], cfg, eval_start, eval_end, snapshot)
    spy_close = bars.close["SPY"].dropna()
    sleeve, mask = sleeve_inputs(bars, universe, cfg, eval_start, eval_end)
    grid = cls.grid.override(overrides) if overrides else cls.grid
    print(f"  [universe] {universe}: {len(sleeve.symbols)} symbols (engine vbt)", flush=True)
    result = run_wfo(
        strategy,
        cls,
        cfg,
        sleeve,
        spy_close,
        eval_start=str(eval_start.date()),
        eval_end=str(eval_end.date()),
        market=market,
        base_config=dict(base_config),
        grid=grid,
        mask=mask,
    )
    if isinstance(result, str):
        print(result)
        return result
    write_wfo_json(
        result,
        bars,
        result_json
        or str(
            DEFAULT_RESULTS_DIR
            / f"wfo_{strategy}_{universe}_{eval_start.date()}_{eval_end.date()}_vbt.json"
        ),
        strategy=strategy,
        universe=universe,
        eval_start=str(eval_start.date()),
        eval_end=str(eval_end.date()),
    )
    return result


def write_wfo_json(result: WfoResult, bars: Bars, path: str, **fields: Any) -> Path:
    """Write the run record with a provenance block that names the tape and the definitions."""
    from ggTrader.engine.provenance import provenance
    from ggTrader.lab.strategies import registry
    from ggTrader.utils.results import write_result_json

    tape = (bars.provenance or {}).get("snapshot") or {}
    cls = registry.REGISTRY.get(str(fields.get("strategy")))
    # The strategy's own source is part of what produced the number; hash it so an edited
    # or out-of-tree strategy is identified by its result.
    block = provenance(
        tape_sha256=tape.get("sha256"), extra=registry.source_sha256(cls) if cls else None
    )
    payload = result_payload(result, bars_summary=bars.summary(), tape=tape or None, **fields)
    out = write_result_json(path, payload, provenance=block)
    print(f"  result JSON: {out}", flush=True)
    return out
