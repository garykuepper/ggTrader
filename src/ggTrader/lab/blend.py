"""Portfolio-of-sleeves blend: run sleeves through the gated WFO and combine
their OOS curves with the validated inverse-vol / target-vol overlay.

blend_curves is the pure math (no I/O); run_blend orchestrates data load, WFO,
blend, and persistence.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, NamedTuple

import pandas as pd

from ggTrader.lab import persist
from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import STOCK_BASE_CONFIG
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo.report import curve_summary


def blend_curves(
    curves: dict[str, pd.Series],
    *,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Align sleeve OOS equity curves on common dates, blend to a target vol.

    Returns (blend_equity, returns_df, diag). blend_equity is a cumprod curve
    starting at START_CASH; returns_df is the aligned per-sleeve daily returns.
    """
    common = reduce(lambda a, b: a.intersection(b), (c.index for c in curves.values()))
    returns_df = pd.DataFrame(
        {label: curves[label].reindex(common).pct_change(fill_method=None) for label in curves}
    ).dropna()
    blended, diag = combine_sleeves(
        returns_df, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    start_cash = float(STOCK_BASE_CONFIG["START_CASH"])
    blend_equity = (1.0 + blended).cumprod() * start_cash
    return blend_equity, returns_df, diag


class BlendResult(NamedTuple):
    blended_equity: pd.Series
    sleeve_equity: dict[str, pd.Series]
    diag: pd.DataFrame
    table: str
    run_id: str


def _row(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {s['cagr_pct']:.2f}% | {s['sharpe']:.2f} | {s['sortino']:.2f} "
        f"| {s['ann_vol_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% |"
    )


def _engine_curves(
    sleeves: list[tuple[str, str]],
    cfg: LabConfig,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: dict,
    snapshot: str | None,
) -> tuple[dict[str, pd.Series], pd.Series, dict[str, Any]]:
    """Per-sleeve OOS curves, the SPY close and the provenance block, on the engine."""
    from ggTrader.engine.provenance import provenance as build_provenance
    from ggTrader.lab.strategies import registry
    from ggTrader.lab.wfo import WfoResult, run_wfo
    from ggTrader.lab.wfo.cli_support import load_run_bars, sleeve_inputs

    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    bars = load_run_bars(sorted({u for _, u in sleeves}), cfg, es, ee, snapshot)
    spy_close = bars.close["SPY"].dropna()
    curves: dict[str, pd.Series] = {}
    sources: dict[str, Any] = {}
    for strategy, universe in sleeves:
        label = f"{strategy}@{universe}"
        if label in curves:
            raise SystemExit(f"duplicate sleeve {label!r}; each strategy@universe must be unique")
        try:
            cls = type(registry.get(strategy))
        except ValueError as exc:
            raise SystemExit(f"blend sleeve {label!r}: {exc}") from exc
        sources[label] = registry.source_sha256(cls)
        sleeve, mask = sleeve_inputs(bars, universe, cfg, es, ee)
        if not sleeve.symbols:
            raise SystemExit(
                f"blend sleeve {label!r}: no symbols available for universe {universe!r}"
            )
        result = run_wfo(
            strategy,
            cls,
            cfg,
            sleeve,
            spy_close,
            eval_start=eval_start,
            eval_end=eval_end,
            market=market,
            base_config=base_config,
            grid=cls.grid,
            mask=mask,
        )
        if not isinstance(result, WfoResult):
            raise SystemExit(f"blend sleeve {label!r}: WFO produced no valid folds ({result})")
        curves[label] = result.oos_equity
    tape = (bars.provenance or {}).get("snapshot") or {}
    stamp = build_provenance(tape_sha256=tape.get("sha256"), extra={"strategy_sources": sources})
    return curves, spy_close, stamp


def run_blend(
    sleeves: list[tuple[str, str]],
    cfg: LabConfig,
    eval_start: str,
    eval_end: str,
    *,
    market: str,
    base_config: dict,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 1.0,
    snapshot: str | None = None,
) -> BlendResult:
    """Run each (strategy, universe) sleeve through the gated WFO, blend the OOS
    curves (inverse-vol -> target-vol), persist as a lab run, return BlendResult.

    Each sleeve runs through the vectorbt-engine walk-forward (``lab.wfo``) on the frozen tape
    (``snapshot`` overrides the file); the ``lab_runs`` row carries the engine provenance block.
    Only strategies ported to the engine contract can be sleeves.

    NOTE: max_leverage defaults to 1.0, the deployed value
    (src/ggTrader/paper/overlay.py). Until 2026-11 the default was 2.0 (the idealised research
    value); a run quoted from that default overstated the deployed blend (the trap was hit on
    2026-07-13 and 2026-08-18). Pass 2.0 explicitly for the unconstrained comparison.
    """
    curves, spy_close, stamp = _engine_curves(
        sleeves, cfg, eval_start, eval_end, market, base_config, snapshot
    )

    blend_eq, returns_df, diag = blend_curves(
        curves, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    common = returns_df.index
    start_cash = float(STOCK_BASE_CONFIG["START_CASH"])
    spy_common = spy_close.reindex(common).ffill()
    spy_bench = start_cash * (spy_common / spy_common.dropna().iloc[0])
    spy_stats = curve_summary(spy_common)

    # Build the report table.
    rows = [
        "| Strategy | CAGR | Sharpe | Sortino | Vol | Max DD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label in curves:
        rows.append(_row(label, curve_summary(curves[label].reindex(common))))
    rows.append(_row("Inverse-vol + target-vol blend", curve_summary(blend_eq)))
    rows.append(_row("SPY", spy_stats))
    table = "\n".join(rows)

    # Persist as a normal lab run.
    persist.init_schema()
    labels = list(curves)
    run_id = persist.start_run(
        f"blend:{','.join(labels)}",
        market,
        "blend",
        eval_start,
        eval_end,
        params={
            "sleeves": labels,
            "target_vol": target_vol,
            "window": window,
            "max_leverage": max_leverage,
            "engine": "vbt",
        },
        provenance=stamp,
    )
    for label in labels:
        sleeve_eq = curves[label].reindex(common)
        persist.write_returns_equity(
            run_id, label, sleeve_eq.pct_change(fill_method=None).dropna(), sleeve_eq, spy_bench
        )
    persist.write_returns_equity(
        run_id, "blend", blend_eq.pct_change(fill_method=None).dropna(), blend_eq, spy_bench
    )
    persist.write_summary(
        run_id,
        "blend",
        curve_summary(blend_eq),
        spy_stats,
        {
            "avg_leverage": float(diag["scale"].mean()),
            "max_leverage": float(diag["scale"].max()),
            "sleeves": labels,
        },
    )
    persist.finish_run(run_id)

    return BlendResult(
        blended_equity=blend_eq,
        sleeve_equity={label: curves[label].reindex(common) for label in labels},
        diag=diag,
        table=table,
        run_id=run_id,
    )
