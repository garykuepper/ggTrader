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
from ggTrader.lab.data import STOCK_BASE_CONFIG, equity_universe_between, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategies import STRATEGY_REGISTRY
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo


def blend_curves(
    curves: dict[str, pd.Series],
    *,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Align sleeve OOS equity curves on common dates, blend to a target vol.

    Returns (blend_equity, returns_df, diag). blend_equity is a cumprod curve
    starting at START_CASH; returns_df is the aligned per-sleeve daily returns.
    """
    common = reduce(lambda a, b: a.intersection(b), (c.index for c in curves.values()))
    returns_df = pd.DataFrame(
        {label: curves[label].reindex(common).pct_change() for label in curves}
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
    max_leverage: float = 2.0,
) -> BlendResult:
    """Run each (strategy, universe) sleeve through the gated WFO, blend the OOS
    curves (inverse-vol -> target-vol), persist as a lab run, return BlendResult.
    """
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    universes = sorted({u for _, u in sleeves})
    members = {u: equity_universe_between(es, ee, universe=u) for u in universes}
    all_symbols = sorted({s for ms in members.values() for s in ms} | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    spy_close = ohlcv["SPY"]["close"].dropna()

    curves: dict[str, pd.Series] = {}
    for strategy, universe in sleeves:
        label = f"{strategy}@{universe}"
        syms = [x for x in members[universe] if x in available]
        if not syms:
            raise SystemExit(
                f"blend sleeve {label!r}: no symbols available for universe {universe!r}"
            )
        if label in curves:
            raise SystemExit(f"duplicate sleeve {label!r}; each strategy@universe must be unique")
        cls = STRATEGY_REGISTRY[strategy]
        result = run_wfo(
            strategy,
            cls,
            cfg,
            ohlcv[syms],
            spy_close,
            eval_start=eval_start,
            eval_end=eval_end,
            market=market,
            base_config=base_config,
            grid=build_grid(cls),
        )
        if not isinstance(result, WfoResult):
            raise SystemExit(f"blend sleeve {label!r}: WFO produced no valid folds ({result})")
        curves[label] = result.oos_equity

    blend_eq, returns_df, diag = blend_curves(
        curves, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    common = returns_df.index
    start_cash = float(STOCK_BASE_CONFIG["START_CASH"])
    spy_common = spy_close.reindex(common).ffill()
    spy_bench = start_cash * (spy_common / spy_common.dropna().iloc[0])
    spy_stats = curve_stats(spy_common)

    # Build the report table.
    rows = [
        "| Strategy | CAGR | Sharpe | Sortino | Vol | Max DD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label in curves:
        rows.append(_row(label, curve_stats(curves[label].reindex(common))))
    rows.append(_row("Inverse-vol + target-vol blend", curve_stats(blend_eq)))
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
        },
    )
    for label in labels:
        sleeve_eq = curves[label].reindex(common)
        persist.write_returns_equity(
            run_id, label, sleeve_eq.pct_change().dropna(), sleeve_eq, spy_bench
        )
    persist.write_returns_equity(
        run_id, "blend", blend_eq.pct_change().dropna(), blend_eq, spy_bench
    )
    persist.write_summary(
        run_id,
        "blend",
        curve_stats(blend_eq),
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
