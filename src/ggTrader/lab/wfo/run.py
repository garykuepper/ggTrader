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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

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
from ggTrader.lab.wfo.fold_record import FoldRecord
from ggTrader.lab.wfo.folds import TEST_MONTHS, TRAIN_MONTHS, Fold, generate_folds
from ggTrader.lab.wfo.gates import run_dsr, run_ndh
from ggTrader.lab.wfo.report import WfoResult, anchor_history, curve_summary, format_wfo_table
from ggTrader.lab.wfo.scorers import (
    Scorer,
    ScoringInputs,
    SignalScorer,
    SignalSimOptions,
    WeightScorer,
    WindowReq,
)
from ggTrader.lab.wfo.select import best_index, composite_score, pick_live_winner
from ggTrader.utils.json_safe import canonical_hash

#: Config keys of ``LabConfig`` a strategy may take as defaults (the legacy merged config).
CFG_DEFAULT_KEYS = ("top_n", "lookback", "skip")


def _halt_mask_hash(mask: pd.DataFrame) -> str:
    """Hash mask dates, ordered columns and row-major boolean states."""
    return canonical_hash(
        {
            "format": "halt-mask-v1",
            "dates": [timestamp.isoformat() for timestamp in mask.index],
            "columns": list(mask.columns),
            "flags": np.packbits(mask.to_numpy(dtype=np.uint8).ravel()).tobytes().hex(),
        }
    )


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


def _normalise_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
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
    intraday_stops: bool,
    order_sequence: str | None = None,
    stop_exit_price: str | None = None,
    halt_mask: pd.DataFrame | None = None,
) -> Scorer:
    """Build a scorer from the old thirteen-argument call; ``run_wfo`` uses ``_scorer_for``."""
    inputs = ScoringInputs(
        strategy, bars, grid, names, combos, mask, base_config, max_bytes, min_losing_days
    )
    options = SignalSimOptions(intraday_stops, halt_mask, order_sequence, stop_exit_price)
    return _scorer_for(strategy, inputs, options)


def _scorer_for(strategy: Any, inputs: ScoringInputs, options: SignalSimOptions) -> Scorer:
    """The scorer of the strategy's kind; options only a signal strategy can honor are refused."""
    if strategy.kind == "signals":
        return SignalScorer.from_inputs(inputs, options)
    options.require_signal_strategy()
    return WeightScorer.from_inputs(inputs)


@dataclass(frozen=True, eq=False)
class _FoldContext:
    """The per-run constants ``_score_fold`` reads (nothing in it changes from fold to fold)."""

    scorer: Scorer
    strategy_name: str
    names: Sequence[str]
    name_to_idx: Mapping[str, int]
    combos: Sequence[Mapping[str, Any]]
    ndh_threshold: float
    dsr_threshold: float
    policy: WfoPolicy
    n_folds: int
    verbose: bool


class _FoldOutcome(NamedTuple):
    """What the fold loop needs back from ``_score_fold``."""

    anchor: AnchorSet
    record: FoldRecord | None  # None when the train window had no metrics
    skipped: dict[str, Any] | None  # the skipped_folds entry when record is None
    winner: dict[str, Any] | None
    oos_curve: pd.Series | None
    running_value: float
    ran_test: bool
    od12_flip: bool
    state: WfoState


def _score_fold(
    ctx: _FoldContext, i: int, fold: Fold, state: WfoState, running_value: float
) -> _FoldOutcome:
    """Score one fold: anchor, train, gates, the deployed combination's test window, breaker.

    It prints the two verbose fragments itself (the header with ``end=""`` before the train
    window is read, the result line at the end): the interleaving with the scorer calls is part
    of the behavior. It appends to ``state.wfe_history`` and ``state.oos_sharpes`` **in place**
    and returns the breaker's copy as ``outcome.state``. A skipped fold (no train metrics)
    returns the state and the running value it was given, with the anchor picked for it.
    ``run_dsr``, ``run_ndh`` and the other helpers are looked up in this module at call time, so
    tests and frozen studies can replace them.
    """
    scorer = ctx.scorer
    # Anchor: fit only on data available as of this fold's train_end (expanding window).
    anchor_req = WindowReq("anchor", i, None, fold.train_end)
    train_req = WindowReq("train", i, fold.train_start, fold.train_end)
    test_req = WindowReq("test", i, fold.test_start, fold.test_end)
    anchor = pick_anchor(scorer.metrics(anchor_req))
    if ctx.verbose:
        print(
            f"  Fold {i + 1}/{ctx.n_folds}: train {fold.train_start.date()}→"
            f"{fold.train_end.date()} | test {fold.test_start.date()}→{fold.test_end.date()}"
            f" | anchor {anchor.combo} (MaxDD {anchor.max_drawdown_pct:.1f}%,"
            f" CAGR {anchor.cagr_pct:.1f}%)",
            end="",
            flush=True,
        )
    train_metrics = scorer.metrics(train_req)
    if not train_metrics:
        skipped = {"fold_num": i + 1, **fold._asdict(), "reason": "no_train_metrics"}
        return _FoldOutcome(anchor, None, skipped, None, None, running_value, False, False, state)

    scores = composite_score(train_metrics)
    best_idx = best_index(scores)
    winner = train_metrics[best_idx]
    winner_idx = ctx.name_to_idx[winner["combo"]]

    ndh = run_ndh(train_metrics, ctx.combos, best_idx, ctx.ndh_threshold)
    dsr_passed, dsr_value = run_dsr(
        scorer.returns(train_req, winner_idx), len(ctx.combos), ctx.dsr_threshold
    )
    gates_passed = ndh.passed and dsr_passed

    # OD12: the NDH expectancy divides by trades; compare the legacy whole-simulation count
    # with the per-window count (only the sign reaches the gate, so no flip is expected).
    whole = scorer.trade_counts(train_req, cumulative=True)
    legacy_metrics = [dict(m, n_trades=int(whole[j])) for j, m in enumerate(train_metrics)]
    ndh_whole = run_ndh(legacy_metrics, ctx.combos, best_idx, ctx.ndh_threshold)
    od12_flip = ndh_whole.passed != ndh.passed

    halted_entering = state.halted
    deploy_params = winner["params"]
    used_anchor = False
    anchor_reason: str | None = None
    if not gates_passed or state.halted:
        deploy_params = anchor.params
        used_anchor = True
        anchor_reason = "halted" if state.halted else "gates_failed"

    # Test: the deployed combination's test window, from the same simulation.
    deploy_idx = ctx.name_to_idx.get(inputs.combo_name(ctx.strategy_name, deploy_params))
    test_metrics = scorer.metrics(test_req, [deploy_idx]) if deploy_idx is not None else []
    oos_score = 0.0
    ran_test = False
    oos_curve: pd.Series | None = None
    if test_metrics:
        ran_test = True
        oos_score = composite_score(test_metrics)[0]
        eq_test = scorer.equity(test_req, deploy_idx)
        if len(eq_test) > 0:
            normalized = running_value * (eq_test / eq_test.iloc[0])
            oos_curve = normalized
            running_value = float(normalized.iloc[-1])
    elif deploy_idx is None:
        # No anchor exists (no combination had traded by train_end): the book stays in cash,
        # a flat curve, so the out-of-sample record and the SPY comparison keep their days.
        dates = scorer.window_dates(test_req)
        if len(dates) > 0:
            oos_curve = pd.Series(running_value, index=dates)

    is_sharpe = winner.get("sharpe", float("nan"))
    oos_sharpe = float("nan")
    if test_metrics:
        oos_sharpe = test_metrics[0].get("sharpe", float("nan"))

    wfe_val = compute_wfe(is_sharpe, oos_sharpe)
    state.wfe_history.append(
        None if used_anchor and ctx.policy.anchor_fold_wfe == "exclude_from_breaker" else wfe_val
    )
    if np.isfinite(oos_sharpe):
        state.oos_sharpes.append(oos_sharpe)

    if not state.halted:
        if ctx.policy.nan_wfe == "legacy":
            new_state = check_circuit_breaker(state)
        else:
            new_state = check_circuit_breaker(state, nan_policy=ctx.policy.nan_wfe)
    else:
        new_state = check_shadow_reentry(state, ndh.passed, dsr_passed, wfe_val)

    record = FoldRecord(
        fold_num=i + 1,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        winner_combo=winner["combo"],
        winner_params=winner["params"],
        train_score=scores[best_idx],
        oos_score=oos_score,
        ndh_passed=ndh.passed,
        dsr_passed=dsr_passed,
        gates_passed=gates_passed,
        ndh_density=ndh.density,
        dsr_value=dsr_value,
        wfe=wfe_val,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        halted=new_state.halted,
        halted_entering=halted_entering,
        used_anchor=used_anchor,
        anchor_combo=anchor.combo,
        anchor_params=anchor.params,
        anchor_reason=anchor_reason,
        deployed_combo=ctx.names[deploy_idx] if deploy_idx is not None else "none",
        halt_cleared=bool(halted_entering and not new_state.halted),
        winner_trades_in_window=winner["n_trades"],
        winner_trades_whole_simulation=int(whole[winner_idx]),
        od12_ndh_flip=od12_flip,
    )
    if ctx.verbose:
        oos_s = f" OOS Sharpe {oos_sharpe:.2f}" if np.isfinite(oos_sharpe) else ""
        wfe_s = f" WFE {wfe_val:.2f}" if wfe_val is not None else ""
        gate_s = (
            f" [NDH dens {ndh.density:.2f} var {ndh.variance_ratio:.2f}/0.20 "
            f"{'✓' if ndh.passed else '✗'} DSR {dsr_value:.2f}{'✓' if dsr_passed else '✗'}]"
        )
        print(f" → done{oos_s}{wfe_s}{gate_s}", flush=True)
    return _FoldOutcome(
        anchor, record, None, winner, oos_curve, running_value, ran_test, od12_flip, new_state
    )


def run_wfo(
    strategy_name: str,
    strategy_cls: Any,
    cfg: LabConfig,
    ohlcv: Bars | pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: dict[str, Any],
    grid: ParamGrid | Sequence[dict[str, Any]],
    ndh_threshold: float = 0.85,
    dsr_threshold: float = 0.80,
    universe_fn: Any = None,
    policy: WfoPolicy = WfoPolicy(),  # noqa: B008  (frozen dataclass, as the legacy signature)
    *,
    mask: pd.DataFrame | None = None,
    allow_unmasked: bool = False,
    max_bytes: int = engine_portfolio.DEFAULT_MAX_BYTES,
    min_losing_days: int = engine_config.MIN_LOSING_DAYS,
    intraday_stops: bool = False,
    halt_mask: pd.DataFrame | None = None,
    order_sequence: str | None = None,
    stop_exit_price: str | None = None,
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
        intraday_stops: simulate signal-strategy stops using complete OHLC bars; opt-in only.
        halt_mask: verified boolean halt bars; reject positions overlapping flagged bars.
            Coverage, identity and timing are caller responsibilities; no fills are modeled.
        order_sequence: ``default`` or ``auto`` for signal strategies; omitted selects the
            existing mode-specific default.
        stop_exit_price: ``stoplimit`` or ``stopmarket`` for signal strategies; omitted uses
            the existing mode-specific default.
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

    scoring_inputs = ScoringInputs(
        strategy, bars, param_grid, names, combos, pit, base, max_bytes, min_losing_days
    )
    sim_options = SignalSimOptions(intraday_stops, halt_mask, order_sequence, stop_exit_price)
    scorer = _scorer_for(strategy, scoring_inputs, sim_options)
    live_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    live_req = WindowReq("live", 0, live_start, eval_end_ts)
    requests: list[WindowReq] = [live_req]
    for i, fold in enumerate(folds):
        requests += [
            WindowReq("anchor", i, None, fold.train_end),
            WindowReq("train", i, fold.train_start, fold.train_end),
            WindowReq("test", i, fold.test_start, fold.test_end),
        ]
    scorer.prepare(requests)

    fold_results: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    oos_curves: list[pd.Series] = []
    oos_running_value = start_cash
    fold_winners: list[dict[str, Any]] = []
    od12_flips: list[int] = []
    wfo_state = WfoState()
    anchor: AnchorSet = NO_ANCHOR
    n_test_runs = 0
    fold_started = time.time()

    ctx = _FoldContext(
        scorer=scorer,
        strategy_name=strategy_name,
        names=names,
        name_to_idx=name_to_idx,
        combos=combos,
        ndh_threshold=ndh_threshold,
        dsr_threshold=dsr_threshold,
        policy=policy,
        n_folds=len(folds),
        verbose=verbose,
    )
    for i, fold in enumerate(folds):
        out = _score_fold(ctx, i, fold, wfo_state, oos_running_value)
        anchor = out.anchor
        if out.record is None:
            skipped_folds.append(out.skipped)
            continue
        n_test_runs += int(out.ran_test)
        if out.od12_flip:
            od12_flips.append(i + 1)
        if out.oos_curve is not None:
            oos_curves.append(out.oos_curve)
        oos_running_value = out.running_value
        wfo_state = out.state
        fold_results.append(out.record.as_dict())
        fold_winners.append(out.winner)

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

    live = _select_live_params(scorer, live_req, live_start, eval_end_ts, fold_winners)
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
    stop_settings = getattr(scorer, "stop_settings", None)
    price_stop_active = stop_settings is not None and any(
        stop_settings[key] is not None
        for key in ("trailing_fraction", "take_profit_fraction", "atr_multiplier")
    )
    diagnostics = {
        "engine": engine_config.active_engine(),
        "definitions": "vbt",
        "strategy_kind": strategy.kind,
        "intraday_stops": intraday_stops,
        "halt_mask_supplied": halt_mask is not None,
        "halt_mask_sha256": _halt_mask_hash(halt_mask) if halt_mask is not None else None,
        "order_sequence": (
            order_sequence or ("auto" if intraday_stops else "default")
            if strategy.kind == "signals"
            else "auto"
        ),
        "stop_exit_price": (
            stop_exit_price or ("stopmarket" if intraday_stops else "stoplimit")
            if price_stop_active
            else None
        ),
        "stop_entry_price": (base_config.get("stop_entry_price") or "close")
        if price_stop_active
        else None,
        "stop_settings": stop_settings,
        "time_stop_warning": (
            engine_portfolio.TIME_STOP_WARNING
            if stop_settings is not None and stop_settings["time_stop_bars"] is not None
            else None
        ),
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
    fold_winners: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the most recent ``TRAIN_MONTHS`` window and pick the durable winner."""
    train_metrics = scorer.metrics(live_req)
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}
    scores = composite_score(train_metrics)
    fold_win_counts: dict[str, int] = {}
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
