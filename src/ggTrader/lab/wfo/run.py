"""Walk-forward optimisation on the vectorbt engine: folds, selection, gates, anchor, breaker.

``run_wfo`` keeps the legacy ``lab.wfo.run_wfo`` call signature and result so callers switch by
import. What changed is how the numbers are made: signal strategies run **one** grouped
full-history simulation (every grid combination x every symbol, the point-in-time mask tiled
onto the parameter columns) that is sliced per window; weight strategies run fresh-start
windows stacked into one call per window kind (see :mod:`ggTrader.lab.wfo.scorers`).
Statistics come from ``engine.stats``; everything below is policy glue.

Definitions (owner decision 2026-11-06): vectorbt's own Sortino and CAGR; Sharpe, volatility
and drawdown follow the ``curve_stats`` window convention. The legacy forms were used once, as a
diagnostic, and are gone (see research/programme/vectorbt-engine/P5-completion.md).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars, to_legacy_frame
from ggTrader.engine import config as engine_config
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import inputs
from ggTrader.lab.wfo.anchor import NO_ANCHOR, AnchorSet, pick_anchor
from ggTrader.lab.wfo.breaker import (
    WfoPolicy,
    WfoState,
    check_circuit_breaker,
    check_shadow_reentry,
    compute_wfe,
)
from ggTrader.lab.wfo.folds import TEST_MONTHS, TRAIN_MONTHS, Fold, generate_folds
from ggTrader.lab.wfo.gates import run_dsr, run_ndh
from ggTrader.lab.wfo.report import WfoResult, anchor_history, curve_summary, format_wfo_table
from ggTrader.lab.wfo.scorers import Scorer, SignalScorer, WeightScorer, WindowReq
from ggTrader.lab.wfo.select import best_index, composite_score, pick_live_winner

#: Config keys of ``LabConfig`` a strategy may take as defaults (the legacy merged config).
CFG_DEFAULT_KEYS = ("top_n", "lookback", "skip")


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux ``/proc``; 0 elsewhere)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _instantiate(strategy_name: str, strategy_cls: Any) -> Any:
    """A strategy instance on the engine contract (``kind`` and ``build``), from a class,
    a zero-argument factory or the strategy registry when ``strategy_cls`` is ``None``."""
    if strategy_cls is None:
        from ggTrader.lab.strategies import registry

        return registry.get(strategy_name)
    contract_cls = isinstance(strategy_cls, type) and hasattr(strategy_cls, "build")
    strategy = strategy_cls() if (contract_cls or not isinstance(strategy_cls, type)) else None
    if strategy is None or getattr(strategy, "kind", None) not in ("signals", "weights"):
        raise ValueError(
            f"{strategy_name!r} is not on the engine contract (needs `kind` and `build`); "
            "legacy strategy classes run through ggTrader.lab.wfo.run_wfo or at git tag "
            "legacy-engine-vbt0285"
        )
    return strategy


def _normalise_config(base_config: Mapping[str, Any]) -> Dict[str, Any]:
    """The run's base config; ``"1d"`` becomes the engine's ``"1D"`` (pandas 3 wants upper)."""
    config = dict(base_config)
    if str(config.get("FREQ", "")).lower() == "1d":
        config["FREQ"] = engine_config.BAR_FREQ
    return config


def _resolve_mask(
    bars: Bars,
    ohlcv: Any,
    mask: pd.DataFrame | None,
    universe_fn: Any,
    kind: str,
    name: str,
    allow_unmasked: bool = False,
) -> pd.DataFrame | None:
    """The point-in-time mask from an explicit frame, a legacy ``universe_fn`` or nothing.

    Raises:
        ValueError: neither a mask nor a ``universe_fn`` was given, and ``allow_unmasked`` is
            False. A run without membership data trades symbols on days they were not in the
            index (survivorship bias), so it has to be asked for.
    """
    if mask is not None:
        return mask.reindex(index=bars.close.index, columns=bars.close.columns, fill_value=False)
    if universe_fn is not None:
        legacy = ohlcv if isinstance(ohlcv, pd.DataFrame) else to_legacy_frame(bars)
        return inputs.mask_from_universe_fn(legacy, bars.symbols, universe_fn)
    if kind == "weights":
        raise ValueError(
            f"{name}: weight strategies require universe_fn or mask "
            "(e.g. lambda asof, past: eligible_at(asof, past, cfg, universe=...)[0])"
        )
    if not allow_unmasked:
        raise ValueError(
            f"{name}: no point-in-time mask or universe_fn given, so entries would not be limited "
            "to the index members of each day (survivorship bias). Pass mask=lab.universe."
            "pit_mask(...) or, for a synthetic or single-symbol run, allow_unmasked=True"
        )
    print(f"  WARNING: {name} runs with allow_unmasked=True; entries are NOT PIT-masked")
    return None


def _make_scorer(
    strategy: Any,
    bars: Bars,
    grid: ParamGrid,
    names: Sequence[str],
    combos: Sequence[Mapping[str, Any]],
    mask: pd.DataFrame | None,
    base_config: Mapping[str, Any],
    max_bytes: int,
    min_losing_days: int,
) -> Scorer:
    cls = SignalScorer if strategy.kind == "signals" else WeightScorer
    return cls(strategy, bars, grid, names, combos, mask, base_config, max_bytes, min_losing_days)


def run_wfo(
    strategy_name: str,
    strategy_cls: Any,
    cfg: LabConfig,
    ohlcv: Bars | pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: ParamGrid | Sequence[Dict[str, Any]],
    ndh_threshold: float = 0.85,
    dsr_threshold: float = 0.80,
    universe_fn: Any = None,
    policy: WfoPolicy = WfoPolicy(),  # noqa: B008  (frozen dataclass, as the legacy signature)
    *,
    mask: pd.DataFrame | None = None,
    allow_unmasked: bool = False,
    max_bytes: int = engine_portfolio.DEFAULT_MAX_BYTES,
    min_losing_days: int = engine_config.MIN_LOSING_DAYS,
    verbose: bool = True,
) -> WfoResult | str:
    """Main WFO entry point: fold, train, test, concatenate, report.

    Args:
        strategy_name: registry name; labels combinations.
        strategy_cls: class or zero-argument factory of a strategy on the engine contract
            (``None`` looks the name up in the strategy registry).
        cfg: lab config (``min_history_bars`` etc.; its ``top_n``/``lookback``/``skip`` override
            the strategy defaults, as the legacy merged config did).
        ohlcv: ``Bars`` or a legacy ``(symbol, field)`` frame with a tz-aware UTC index.
        spy_close: benchmark close, scored over the OOS days.
        eval_start, eval_end: evaluation window (pin both for any cited number).
        market: label only.
        base_config: ``START_CASH``, ``FEES``, ``SLIPPAGE``, ``FREQ``, ``SIGNAL_POSITION_SIZE``.
        grid: a ``ParamGrid`` or the legacy list of combination dicts.
        ndh_threshold, dsr_threshold: gate thresholds.
        universe_fn: legacy ``(asof, past) -> symbols`` membership function (slow path).
        policy: breaker semantics.
        mask: boolean ``dates x symbols`` eligibility (``lab.universe.pit_mask``); preferred.
        allow_unmasked: run a signal strategy with neither ``mask`` nor ``universe_fn`` (for
            synthetic data or a fixed single-symbol tape). Without it such a run raises; a weight
            strategy always needs a mask.
        max_bytes: engine chunking budget.
        min_losing_days: a selection window needs this many losing days for its Sortino to
            count (degeneracy guard, ``engine.config.MIN_LOSING_DAYS``; 0 turns it off).
            Selection windows also blank Sharpe and Sortino of a combination with no entry
            inside them (spec 5.6). The values used are stamped in ``diagnostics["guards"]``.
        verbose: print the per-fold progress and the table, as the legacy run did.

    Returns:
        A :class:`WfoResult`, or the legacy "no valid folds" string.
    """
    started = time.time()
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)
    if not folds:
        return (
            f"WFO: {strategy_name} | no valid folds (need >= {TRAIN_MONTHS + TEST_MONTHS} months)"
        )

    bars = inputs.as_bars(ohlcv)
    strategy = _instantiate(strategy_name, strategy_cls)
    for key in CFG_DEFAULT_KEYS:
        if key in getattr(strategy, "defaults", {}):
            strategy.defaults = {**strategy.defaults, key: getattr(cfg, key)}
    param_grid, combos = inputs.resolve_grid(type(strategy), grid)
    names = [inputs.combo_name(strategy_name, c) for c in combos]
    name_to_idx = {n: i for i, n in enumerate(names)}
    base = _normalise_config(base_config)
    start_cash = float(base["START_CASH"])
    pit = _resolve_mask(
        bars, ohlcv, mask, universe_fn, strategy.kind, strategy_name, allow_unmasked
    )

    scorer = _make_scorer(
        strategy, bars, param_grid, names, combos, pit, base, max_bytes, min_losing_days
    )
    live_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    live_req = WindowReq("live", 0, live_start, eval_end_ts)
    requests: List[WindowReq] = [live_req]
    for i, fold in enumerate(folds):
        requests += [
            WindowReq("anchor", i, None, fold.train_end),
            WindowReq("train", i, fold.train_start, fold.train_end),
            WindowReq("test", i, fold.test_start, fold.test_end),
        ]
    scorer.prepare(requests)

    fold_results: List[Dict[str, Any]] = []
    skipped_folds: List[Dict[str, Any]] = []
    oos_curves: List[pd.Series] = []
    oos_running_value = start_cash
    fold_winners: List[Dict[str, Any]] = []
    od12_flips: List[int] = []
    wfo_state = WfoState()
    anchor: AnchorSet = NO_ANCHOR
    n_test_runs = 0
    fold_started = time.time()

    for i, fold in enumerate(folds):
        # Anchor: fit only on data available as of this fold's train_end (expanding window).
        anchor_req = WindowReq("anchor", i, None, fold.train_end)
        train_req = WindowReq("train", i, fold.train_start, fold.train_end)
        test_req = WindowReq("test", i, fold.test_start, fold.test_end)
        anchor = pick_anchor(scorer.metrics(anchor_req))
        if verbose:
            print(
                f"  Fold {i + 1}/{len(folds)}: train {fold.train_start.date()}→"
                f"{fold.train_end.date()} | test {fold.test_start.date()}→{fold.test_end.date()}"
                f" | anchor {anchor.combo} (MaxDD {anchor.max_drawdown_pct:.1f}%,"
                f" CAGR {anchor.cagr_pct:.1f}%)",
                end="",
                flush=True,
            )
        train_metrics = scorer.metrics(train_req)
        if not train_metrics:
            skipped_folds.append(
                {"fold_num": i + 1, **fold._asdict(), "reason": "no_train_metrics"}
            )
            continue

        scores = composite_score(train_metrics)
        best_idx = best_index(scores)
        winner = train_metrics[best_idx]
        winner_idx = name_to_idx[winner["combo"]]

        ndh = run_ndh(train_metrics, combos, best_idx, ndh_threshold)
        dsr_passed, dsr_value = run_dsr(
            scorer.returns(train_req, winner_idx), len(combos), dsr_threshold
        )
        gates_passed = ndh.passed and dsr_passed

        # OD12: the NDH expectancy divides by trades; compare the legacy whole-simulation count
        # with the per-window count (only the sign reaches the gate, so no flip is expected).
        whole = scorer.trade_counts(train_req, cumulative=True)
        legacy_metrics = [dict(m, n_trades=int(whole[j])) for j, m in enumerate(train_metrics)]
        ndh_whole = run_ndh(legacy_metrics, combos, best_idx, ndh_threshold)
        od12_flip = ndh_whole.passed != ndh.passed
        if od12_flip:
            od12_flips.append(i + 1)

        halted_entering = wfo_state.halted
        deploy_params = winner["params"]
        used_anchor = False
        anchor_reason: str | None = None
        if not gates_passed or wfo_state.halted:
            deploy_params = anchor.params
            used_anchor = True
            anchor_reason = "halted" if wfo_state.halted else "gates_failed"

        # Test: the deployed combination's test window, from the same simulation.
        deploy_idx = name_to_idx.get(inputs.combo_name(strategy_name, deploy_params))
        test_metrics = scorer.metrics(test_req, [deploy_idx]) if deploy_idx is not None else []
        oos_score = 0.0
        if test_metrics:
            n_test_runs += 1
            oos_score = composite_score(test_metrics)[0]
            eq_test = scorer.equity(test_req, deploy_idx)
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])
        elif deploy_idx is None:
            # No anchor exists (no combination had traded by train_end): the book stays in cash,
            # a flat curve, so the out-of-sample record and the SPY comparison keep their days.
            dates = scorer.window_dates(test_req)
            if len(dates) > 0:
                oos_curves.append(pd.Series(oos_running_value, index=dates))

        is_sharpe = winner.get("sharpe", float("nan"))
        oos_sharpe = float("nan")
        if test_metrics:
            oos_sharpe = test_metrics[0].get("sharpe", float("nan"))

        wfe_val = compute_wfe(is_sharpe, oos_sharpe)
        wfo_state.wfe_history.append(
            None if used_anchor and policy.anchor_fold_wfe == "exclude_from_breaker" else wfe_val
        )
        if np.isfinite(oos_sharpe):
            wfo_state.oos_sharpes.append(oos_sharpe)

        if not wfo_state.halted:
            if policy.nan_wfe == "legacy":
                wfo_state = check_circuit_breaker(wfo_state)
            else:
                wfo_state = check_circuit_breaker(wfo_state, nan_policy=policy.nan_wfe)
        else:
            wfo_state = check_shadow_reentry(wfo_state, ndh.passed, dsr_passed, wfe_val)

        fold_results.append(
            {
                "fold_num": i + 1,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "winner_combo": winner["combo"],
                "winner_params": winner["params"],
                "train_score": scores[best_idx],
                "oos_score": oos_score,
                "ndh_passed": ndh.passed,
                "dsr_passed": dsr_passed,
                "gates_passed": gates_passed,
                "ndh_density": ndh.density,
                "dsr_value": dsr_value,
                "wfe": wfe_val,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "halted": wfo_state.halted,
                "halted_entering": halted_entering,
                "used_anchor": used_anchor,
                # the fallback parameter set of this fold (fit on data up to train_end) and
                # what was deployed: the whole OOS book when the anchor is used (review MAJOR-3)
                "anchor_combo": anchor.combo,
                "anchor_params": anchor.params,
                "anchor_reason": anchor_reason,
                "deployed_combo": names[deploy_idx] if deploy_idx is not None else "none",
                "halt_cleared": bool(halted_entering and not wfo_state.halted),
                "winner_trades_in_window": winner["n_trades"],
                "winner_trades_whole_simulation": int(whole[winner_idx]),
                "od12_ndh_flip": od12_flip,
            }
        )
        fold_winners.append(winner)
        if verbose:
            oos_s = f" OOS Sharpe {oos_sharpe:.2f}" if np.isfinite(oos_sharpe) else ""
            wfe_s = f" WFE {wfe_val:.2f}" if wfe_val is not None else ""
            gate_s = (
                f" [NDH dens {ndh.density:.2f} var {ndh.variance_ratio:.2f}/0.20 "
                f"{'✓' if ndh.passed else '✗'} DSR {dsr_value:.2f}{'✓' if dsr_passed else '✗'}]"
            )
            print(f" → done{oos_s}{wfe_s}{gate_s}", flush=True)

    nan_metrics = {
        "sharpe": float("nan"),
        "cagr_pct": float("nan"),
        "max_drawdown_pct": float("nan"),
    }
    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_metrics = curve_summary(oos_equity)
        spy_oos = spy_close.reindex(oos_equity.index).ffill().dropna()
        if len(spy_oos) > 1:
            spy_metrics = curve_summary(start_cash * (spy_oos / spy_oos.iloc[0]))
        else:
            spy_metrics = dict(nan_metrics)
    else:
        oos_equity = pd.Series(dtype=float)
        oos_metrics = dict(nan_metrics)
        spy_metrics = dict(nan_metrics)

    live = _select_live_params(scorer, live_req, live_start, eval_end_ts, fold_winners, names)
    table = format_wfo_table(
        fold_results,
        oos_metrics,
        spy_metrics,
        live,
        strategy_name,
        len(combos),
        len(folds),
        halted=wfo_state.halted,
        anchor=anchor,
        anchor_history=anchor_history(fold_results),
    )
    if verbose:
        print(table)

    n_folds = len(folds)
    diagnostics = {
        "engine": engine_config.active_engine(),
        "definitions": "vbt",
        "strategy_kind": strategy.kind,
        "n_combos": len(combos),
        "n_symbols": len(bars.symbols),
        "n_bars": len(bars.close),
        "n_folds": n_folds,
        "n_simulations": scorer.n_simulations,
        "n_chunks": scorer.n_chunks,
        "legacy_equivalent_simulations": n_folds * 2 + n_test_runs + 1,
        "od12_ndh_flips_folds": od12_flips,
        "timings_s": {
            **{k: round(v, 3) for k, v in scorer.timings.items()},
            "folds_s": round(time.time() - fold_started, 3),
            "total_s": round(time.time() - started, 3),
        },
        "peak_rss_mb": round(_peak_rss_mb()),
        "max_bytes": max_bytes,
        "oos_metrics": oos_metrics,
        "spy_metrics": spy_metrics,
        "pit_masked": pit is not None,
        "guards": {
            "min_losing_days": min_losing_days,
            "zero_trade_rule": "selection windows (anchor, train, live)",
        },
    }
    return WfoResult(
        oos_equity=oos_equity,
        fold_results=fold_results,
        live_params=live,
        table=table,
        skipped_folds=tuple(skipped_folds),
        diagnostics=diagnostics,
    )


def _select_live_params(
    scorer: Scorer,
    live_req: WindowReq,
    live_start: pd.Timestamp,
    eval_end_ts: pd.Timestamp,
    fold_winners: List[Dict[str, Any]],
    names: Sequence[str],  # noqa: ARG001
) -> Dict[str, Any]:
    """Score the most recent ``TRAIN_MONTHS`` window and pick the durable winner."""
    train_metrics = scorer.metrics(live_req)
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}
    scores = composite_score(train_metrics)
    fold_win_counts: Dict[str, int] = {}
    for fw in fold_winners:
        fold_win_counts[fw["combo"]] = fold_win_counts.get(fw["combo"], 0) + 1
    best_idx, stability = pick_live_winner(train_metrics, scores, fold_win_counts)
    winner = train_metrics[best_idx]
    return {
        "combo": winner["combo"],
        "params": winner["params"],
        "train_metrics": {
            k: winner[k] for k in ("sharpe", "cagr_pct", "max_drawdown_pct") if k in winner
        },
        "stability": stability,
        "train_start": live_start,
        "train_end": eval_end_ts,
    }


__all__ = ["Fold", "WfoPolicy", "WfoResult", "run_wfo"]
