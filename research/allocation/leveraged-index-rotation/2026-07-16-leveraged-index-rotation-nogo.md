# Leveraged/Inverse Index Rotation: First Honest WFO — NO-GO Across All Three Universes

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-16
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

The leveraged/inverse index rotation strategy (`src/ggTrader/lab/strategies/leveraged_rotation.py`,
built 2026-07-14) rotates a universe's 2x/3x leveraged-long ETF, its inverse ETF, and
cash, driven by the monthly breadth of the existing validated `EnsembleSignal` computed
across that universe's own constituent stocks. It was implemented and unit-tested
(TDD, breadth + hysteresis-rotation helpers, per-universe subclasses for SP500,
Nasdaq100, Russell2000) but **had never been run through the lab's walk-forward
harness (`run_wfo`) against real leveraged-ETF price history before this report**. No
prior Sharpe/CAGR/MaxDD numbers exist for this strategy in the codebase, design spec,
or task reports — earlier framing in this session that conflated it with the
2026-07-13 equity ensemble-reversion "3-sleeve blend" (Sharpe 1.14 vs SPY 0.97, that
work is a different strategy family — signal-sleeve diversification across
SP500/MidCap/Nasdaq — not leveraged/inverse ETF rotation) was incorrect and is
retracted here.

This report is the first honest measurement, run today. The orchestration script
(`scripts/leveraged_rotation_research.py`) had an uncommitted, unrun fix on disk at
session start that (a) switched universe membership from `equity_universe_between`
(all tickers that existed *anywhere* in the eval span) to `universe_members_asof`
called with `pd.Timestamp.now()`, and (b) shortened the eval window from 2010-06-30
to 2019-01-01. Running it produced a clean **NO-GO across all three universes**:
every sleeve underperforms SPY by a wide margin with drawdowns of -80% to -89%, and
the regime halt is active for nearly every fold, meaning the WFO harness itself
judged the strategy too unstable to trade its own selected parameters live.

**2026-08-18 correction:** the (a) change above was described here as a
"point-in-time correct membership" fix. That was backwards. Calling
`universe_members_asof(universe, pd.Timestamp.now())` applies *today's* index
membership uniformly across the entire 2019→present backtest — it's survivorship
bias in the classic sense (any company that left the index between the point it
traded and today is silently excluded from the whole span), and it makes the run
non-reproducible (the member set changes every time the script is re-run, since
`now()` moves). `equity_universe_between(es, ee)` — the union of everything that was
a member *anywhere* in `[es, ee]` — is the less-biased of the two and is not
point-in-time exact either, but doesn't retroactively strip out real historical
members. The universe call has been reverted to `equity_universe_between(es, ee)`
(`scripts/leveraged_rotation_research.py`, see the fix commit and
`tests/scripts/test_leveraged_rotation_research.py`). The ablation below already
showed this variable wasn't the operative one for the NO-GO verdict, so **the
NO-GO stands** — this correction only fixes the mischaracterization of which
membership function is more correct, not the result.

A follow-up ablation (SP500 sleeve only, isolating the two changes) shows the
**eval-window shortening drove the headline number, not the universe-membership
function**: swapping only the universe-membership function (833 members via
`equity_universe_between` vs. 503 members via `universe_members_asof(..., now())`)
produced byte-identical results, while swapping only the eval window changed Sharpe
from -0.56 (2010+) to -0.24 (2019+). Both windows are decisively NO-GO.

This closes the leveraged/inverse rotation arc. It joins the equity-diversification
book (SP500/MidCap/Nasdaq blend, closed 2026-06-27) and crypto-carry (closed
2026-07-07) as an explored-and-rejected direction. The live SP500 core (Sharpe 1.12,
CAGR 16.3%, DD -11%) remains the only validated, deployed configuration.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **Leveraged rotation — SP500** (2019+, span-union universe) | -0.24 | -14.5% | -80.8% | 3/26 folds | anchor fallback (regime halt) |
| **Leveraged rotation — Nasdaq100** (2019+, span-union universe) | -0.37 | -23.7% | -88.5% | 3/26 folds | anchor fallback (regime halt) |
| **Leveraged rotation — Russell2000** (2019+, span-union universe) | -0.14 | -15.3% | -84.2% | 4/26 folds | anchor fallback (regime halt) |
| **Leveraged rotation — SP500** (2010+, span-union or today's-membership universe — identical) | -0.56 | -22.6% | -98.6% | n/a | anchor fallback (regime halt) |
| **SPY benchmark** | 0.65–0.77 | 14.1–15.3% | -33.7% | N/A | Buy-and-hold |
| *For reference — live SP500 core (different strategy, deployed)* | *1.12* | *16.3%* | *-11%* | *16/17 folds* | *3% flat* |

Aggregate WFE ranged 0.07 to 1.15 against a ≥0.50 target — the pass/fail signal is
noisy precisely because so few folds gate-pass at all; the strategy spends most of
its life on the defensive anchor combo, not its WFO-selected winner.

## 3. Actionable Research Directions

None ranked. This is a closure report, not a continuation — see §6 for the one
open question worth a decisive check before fully parking the idea.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Leveraged/inverse index rotation (breadth-driven, all 3 universes) — REJECTED.**
OOS Sharpe -0.24 (SP500) / -0.37 (Nasdaq100) / -0.14 (Russell2000) vs. SPY 0.65-0.77;
MaxDD -80.8% to -88.5% vs. SPY -33.7%; regime halt active on 22-23 of 26 folds in
every universe (strategy trades its defensive anchor set, not its selected
parameters, almost the entire backtest). Confirmed via ablation that this holds
under both the 2010+ and 2019+ eval windows and is independent of which universe
membership function is used (identical results with 833 span-union members vs.
503 today's-membership members — see the 2026-08-18 correction in §1: the
`universe_members_asof(..., now())` variant is the survivorship-biased one and
has been reverted, but it was never the operative variable for this verdict).
The core mechanism — using cross-sectional signal breadth as a timing
input for 2x/3x leveraged directional bets — does not survive contact with real
leveraged-ETF decay and whipsaw.

**B. Equity ensemble-reversion diversification (SP500+MidCap+Nasdaq sleeves,
"1.14 Sharpe 3-sleeve blend") — separate arc, already closed 2026-06-27.** Not
superseded or contradicted by this report; noted here only because it was
mistakenly conflated with (A) earlier in this session's discussion.

## 5. Operational Roadmap: Recommended First Action

**Do not pursue leveraged/inverse index rotation further; remove it from the active
research queue.** No parameter re-sweep or gate-tuning is likely to close an 80+
point Sharpe gap against SPY that is stable across three independent universes and
two eval windows. Effort is better spent elsewhere in the roadmap (crypto pivot,
exit-rule levers already identified as exhausted, or new signal families).

The uncommitted survivorship-bias fix in `scripts/leveraged_rotation_research.py`
(point-in-time `universe_members_asof` instead of `equity_universe_between`) is
methodologically correct and should still be committed on its own merits — it just
wasn't the variable that mattered for this particular verdict.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** the regime halt firing on 85%+ of folds means the WFO
harness is mostly scoring the *defensive anchor combo*, not a genuinely optimized
rotation policy — is it possible a properly-gated version (tighter breadth
thresholds, longer min-hold, or a non-leveraged pair) would behave differently, and
this result just indicts the specific parameter grid rather than the mechanism?

**Resolution:** Weak case for one more look, not enough to keep the arc open by
default. The MaxDD figures (-80% to -98%) are driven by leveraged-ETF decay during
whipsaw regimes, which is a structural property of 2x/3x products under any
breadth-timing signal with a 3-month rolling WFO window and no cost-of-leverage
term — a real edge would need to show up as *reduced* whipsaw exposure, and it
doesn't in any of the 26 folds per universe. Closing this arc; would only revisit
if a fundamentally different entry/exit mechanism (e.g., realized-vol-gated
leverage sizing rather than breadth-only) is proposed, and that should be scoped
as new research, not a re-sweep of this design.

### Parked Direction: Realized-vol-gated leverage sizing
Instead of a binary long/inverse/cash breadth signal, scale leveraged-ETF exposure
inversely to realized volatility (similar to the vol-targeting overlay already used
in the live SP500 blend). Not executable as a quick re-run of this script — needs a
new strategy class and its own WFO grid. Gate: only revisit if a specific signal
candidate is proposed; not worth building speculatively given (A)'s magnitude of
failure.
