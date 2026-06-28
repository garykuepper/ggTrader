# Exit-Rule Sweep for the Reversion Ensemble — Design

**Date:** 2026-06-28
**Status:** Approved design, pending implementation plan
**Author:** research session (Claude + Flynn)

## Problem

The 5-voter ensemble (`bb+rsi+ema+macd+vbb`, 3% sizing) exits purely on
**signals**: the RSI voter fires an independent exit when RSI crosses back above
50, or enough voters cast exit votes (`min_agree_exit`). This exit surface has
**never been swept**. For a mean-reversion book, P&L is dominated by *when you
sell* — exiting too early gives back the bounce, exiting too late lets winners
round-trip. Entry-side selection has been exhausted (the ML/feature gate was
falsified three ways on 2026-06-28; see `project_ml_gate_falsified` memory and
roadmap §5), so exits are the largest remaining untested lever, and they carry
no machine-learning overfit risk.

## Goal

Determine, via **honest walk-forward** on S&P 500, whether adding a
**take-profit (TP)** and/or **time-stop (max-hold)** exit improves the 5-voter
ensemble's out-of-sample performance versus the current RSI-exit baseline —
judged through the existing NDH + Deflated-Sharpe gates and circuit breaker, not
idealized numbers.

**Non-goal / explicit constraint:** This is *parallel research*. The nearer
roadmap goal is going live with the current validated config (5-voter / 3% /
RSI-exit). This work produces a **candidate** config; it does **not** mutate the
deployed default, change the live `paper` path, or alter `DEFAULT_VOTERS` /
default exit behavior. The deployed system only changes later, in a separate
explicit step, and only if a candidate clears the deploy bar below.

## Deploy bar (when does an arm "win")

An exit configuration is a deploy candidate only if, on the SP500 honest WFO, it:

1. **Beats the baseline OOS Sharpe** (current ~1.12), and
2. **Survives the gates without anchor-fallback dominating** (i.e. the winning
   params deploy on their own merit across most folds, not via the defensive
   anchor), and
3. Does not materially worsen max drawdown vs baseline (DD ≈ −11%).

If no arm clears the bar, the finding is "exits don't help — RSI-exit is already
good," documented in the roadmap, and the live config is unchanged. A negative
result is a valid, valuable outcome.

## Two arms

Per decision, both are tested:

- **Additive** — indicator exits stay ON; TP and/or time-stop cap winners /
  stale trades on top. vectorbt closes the position on whichever fires first.
- **Replacement** — indicator exits OFF (`exits_enabled=False`); the position
  exits purely on TP and/or time-stop.

The baseline control (no TP, no time-stop, indicator exits ON) is included in the
grid so every comparison is apples-to-apples within one WFO run.

## Architecture & touch points

The lab is already built for this. `from_signals` is called once in
`simulate.py` with a `stop_kwargs` block (currently trailing/ATR `sl_stop`).
`sweep.py::split_params` auto-routes "overlay" params (`STOP_PARAMS`) to the sim
config; `wfo.py` already loops stop params per fold. No new engine.

**Key API constraint:** installed `vectorbt==0.28.5` has native **`tp_stop`** but
**no time/duration stop** (`td_stop`/`dt_stop` are vectorbtpro-only, confirmed by
introspecting `Portfolio.from_signals`). So:

- **Take-profit** is a portfolio-side kwarg (like existing `sl_stop`).
- **Time-stop** is implemented **signal-side** as a forced exit N bars after each
  entry: `exits = exits | entries.shift(td_stop)`. This is correct under vbt's
  `from_signals` semantics because an exit signal on a flat position is ignored —
  so if an indicator exit already closed the trade, the time-exit is a no-op.

### Four changes

1. **`src/ggTrader/lab/simulate.py`** — in the `stop_kwargs` block, read
   `tp_stop` from `base_config` and pass `tp_stop=float(...)` (plus
   `stop_exit_price="close"` for fill realism) to `from_signals`. TP coexists
   with an existing `sl_stop` only if we ever combine them; for this work TP is
   used alone (no stop-loss).

2. **`src/ggTrader/lab/strategies/ensemble.py`** — `EnsembleSignal.__init__`
   gains two params:
   - `td_stop: int | None = None` — max-hold in bars; when set, apply
     `exits = exits | entries.shift(td_stop).fillna(False)` at the end of
     `_generate_signals`.
   - `exits_enabled: bool = True` — when `False`, the indicator exits matrix is
     replaced with an all-`False` frame *before* the time-stop OR (so the
     replacement arm exits only on TP/time). The independent RSI exit and vote
     exits are both suppressed.
   Both are threaded through `sweep_signals` (read from each combo with
   `.get(..., default)`).

3. **`src/ggTrader/lab/sweep.py`** — add `tp_stop` to `STOP_PARAMS` (routes it to
   the sim config / overlay). `td_stop` and `exits_enabled` are **signal** params
   and flow through combos to `EnsembleSignal` unchanged — no registry change.
   Extend `valid_combo` to reject the degenerate **"no exit at all"**:
   `exits_enabled is False AND td_stop is None AND tp_stop is None` → invalid (a
   position that can never close).

4. **Sweep grid** — exposed via `--sweep-param` (no permanent change to
   `sweep_params()` defaults, keeping the deployed default untouched):
   - `tp_stop ∈ {None, 0.03, 0.05, 0.08}`
   - `td_stop ∈ {None, 5, 10, 20}`
   - `exits_enabled ∈ {True, False}`
   - Baseline `(tp_stop=None, td_stop=None, exits_enabled=True)` always present.
   Invalid/degenerate combos filtered by `valid_combo`. Grid size is modest
   (≈ 4×4×2 minus degenerate = ~31 combos) — well within WFO budget.

## Data flow (one fold)

```
combo {min_agree..., tp_stop, td_stop, exits_enabled}
  → split_params → (signal_params incl. td_stop/exits_enabled, overlay incl. tp_stop)
  → EnsembleSignal(**signal_params).sweep_signals → SignalTargets(entries, exits')
       where exits' = (exits_enabled ? indicator_exits : all-False) | entries.shift(td_stop)
  → simulate_signals(targets, prices, {**base_config, tp_stop})
       → from_signals(entries, exits', tp_stop=…, size=3%, …)
  → metrics → gates (NDH, DSR) → OOS score → WFE → circuit breaker
```

## Testing (TDD)

Unit tests (native `.venv`, `pytest`), written before implementation:

- **time-stop shift**: an entry with no indicator exit closes exactly
  `td_stop` bars later; verify on a tiny synthetic price/entries frame.
- **time-stop no-op when already exited**: indicator exit before `td_stop` →
  position closed by the indicator exit, the later shifted exit changes nothing.
- **`exits_enabled=False`**: indicator exits are fully suppressed; with a
  `td_stop` set, the *only* exits are the shifted ones.
- **`tp_stop` plumbing**: `simulate_signals` with `tp_stop` set produces a TP
  exit on a synthetic price that rises past the target; absent `tp_stop`,
  behavior is unchanged (regression guard on the existing path).
- **`valid_combo` guard**: the degenerate no-exit combo is rejected; all other
  combinations pass.
- **default unchanged**: `EnsembleSignal()` with no new args yields byte-identical
  `entries`/`exits` to the pre-change implementation (protects the live default).

## Validation run

`ggt lab --strategy ensemble --wfo --universe sp500 --sweep-param tp_stop=…
--sweep-param td_stop=… --sweep-param exits_enabled=…` on the 5-voter. Compare
each arm's OOS Sharpe / CAGR / DD / WFE against the in-grid baseline. Record the
result (win or "no improvement") in the roadmap and a memory file. Only on a
clear, gate-honest win do we open a *separate* change to promote the candidate;
this spec does not include that promotion.

## Out of scope (YAGNI)

- **Band-touch exit** (exit at BB midline) — deferred; needs more exits-matrix
  work and adds parameters. Revisit only if TP/time look promising.
- **Trailing / hard stop-loss** — already proven harmful for this reversion book
  (roadmap §2b); excluded.
- **Per-symbol or regime-adaptive exits** — premature.
- **Any change to the live `paper` path, default voters, or default exits.**

## Risks & mitigations

- **Overfitting the exit grid** — mitigated by the deflated-Sharpe gate
  (accounts for number of trials), the modest grid, and the honest WFO that the
  whole lab is built around.
- **Time-stop edge case** (a second entry signal vbt ignored while in a position
  still casts a shifted exit) — documented approximation; standard for
  vectorized max-hold; covered by the "already exited" test. Refine only if it
  proves material to results.
- **Destabilizing go-live** — mitigated structurally: no defaults change, sweep
  values are passed at the CLI, and the deployed config is untouched until a
  separate, explicit promotion step.
