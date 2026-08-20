# Anchor-Fix Reproduction: The 1.12 SP500-Core Baseline Does Not Reproduce

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-08-19
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

`run_wfo` (`src/ggTrader/lab/wfo.py`) walks a strategy through rolling
12-month-train / 3-month-test folds, applying an NDH (neighborhood-density)
plateau-robustness gate and a DSR (deflated Sharpe ratio) gate to each fold's
in-sample winner before deploying it out-of-sample; a fold that fails either
gate, or trades while the regime-halt circuit breaker is tripped, deploys a
defensive "anchor" combo instead. On 2026-08-18 a leak was fixed
(commit `2eafab1`, `docs/research/2026-08-18-wfo-anchor-leakage-fix.md`):
previously the anchor was fit once on the *entire* eval window before the
fold loop started, so a fold's defensive fallback could be chosen with
knowledge of price action years past that fold's own test window. The fix
scopes anchor computation to each fold's own expanding `[:train_end]` slice.

The live production configuration is the **3-sleeve leverage-realistic
blend** (SP500 + MidCap400 + Nasdaq100, inverse-vol/target-vol weighted,
`target_vol=0.068`, `window=60`, `max_leverage=1.0`), validated 2026-07-13
at Sharpe 1.14 / MaxDD -5.39%, still running live in `PaperTrader`
regardless of anything in this report — **that does not change here; this
report does not touch or recommend touching the live trader.** Layered
beneath it is the **5-voter `EnsembleSignal` SP500 core**, cited everywhere
in this project's history as OOS Sharpe **1.12** / CAGR **16.3%** / MaxDD
**-11.0%** vs. SPY 0.58, 17-fold walk-forward, gates **16/17**
(`docs/research/RESEARCH_SNAPSHOT.md:21`). This is the single most-cited
number in the project — it is the bar every one of nine diversification-
sleeve candidates was measured and rejected against, and the number the
July-13 leverage-realistic blend headline is itself benchmarked against.

**This report's finding: neither cited number reproduces.** A fresh, honest,
anchor-leak-fixed 17-fold WFO of the identical strategy/universe/window
scores OOS Sharpe **0.97**, CAGR **7.8%**, MaxDD **-7.6%** for the SP500
core, with only **12 of 17 folds** clearing gates (vs. the claimed 16/17).
The 3-sleeve blend at the deployable `max_leverage=1.0`, same window, scores
Sharpe **0.68**, CAGR **4.76%**, MaxDD **-6.70%** — well below the cited
1.14 / -5.39% headline. Neither gap is small: core CAGR is roughly half the
cited figure and its gate-fail rate is more than 4x higher than reported;
the blend's Sharpe shortfall (0.46) is even larger than the core's (0.15).
Section 2 gives the full comparison and explains, with direct evidence, why
this is *not* a clean before/after test of the anchor fix specifically: the
benchmark SPY row computed inside this same run scores Sharpe 0.78, not the
0.58 the core baseline cites for "the same" window — proof the two runs are
not actually on the same eval window, because SPY's own return stream cannot
change.

This report closes no research arc and rejects no strategy; the live
trader's configuration is unaffected either way (see §5). It is an
engineering-provenance finding: **the 1.12 baseline number that gated a live
deployment decision was never reproducible after the fact**, because the
run that produced it was never pinned to an explicit eval window.

## 2. Quantitative Performance Context

Config matched to the baseline as closely as documentation permits:
`ggt.py lab --strategy ensemble --universe sp500 --wfo --eval-start
2021-01-31 --eval-end 2026-04-30`. `--eval-start 2021-01-31` is the CLI's
own default and is used unconditionally everywhere the 1.12 baseline is
cited. `--eval-end 2026-04-30` was chosen because it is verified (directly
against `generate_folds()`, 12mo-train/3mo-test rolling) to produce exactly
17 folds, and because it is the exact window `docs/roadmap.md` (lines
117-119) and `RESEARCH_SNAPSHOT.md` §3 repeatedly identify as "the deployed
blend's actual" window — the same window used to catch three prior
eval-window-drift false positives (PEAD, insider-cluster-buy,
congress-trades sleeves, all of which looked good standalone and evaporated
on this exact window). It is the best-pinned, most-audited 17-fold window
available in the project's history. It is explicitly **not** proven to be
bit-identical to whatever undocumented window originally produced 1.12 —
see the eval-window-drift discussion below, which is the reason it cannot
be proven identical.

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **SP500 core, cited baseline (pre-anchor-fix, undocumented eval-end)** | 1.12 | 16.3% | -11.0% | 16/17 | 3.3% flat |
| **SP500 core, this run (post-anchor-fix, 2021-01-31→2026-04-30)** | **0.97** | **7.8%** | **-7.6%** | **12/17** | 3.3% flat |
| SPY, cited baseline's own window | 0.58 | ~14-15% | ~-34% | N/A | Buy-and-hold |
| SPY, this run's window (2021-01-31→2026-04-30) | **0.78** | 13.0% | -22.1% | N/A | Buy-and-hold |
| 3-sleeve blend, lev=1.0, cited headline (2026-07-13, pre-anchor-fix) | 1.14 | 9.93% | -5.39% | -- | inverse-vol/target-vol |
| **3-sleeve blend, lev=1.0, this run (post-anchor-fix, same window)** | **0.68** | **4.76%** | **-6.70%** | -- | inverse-vol/target-vol |

**Per-sleeve breakdown (this run, feeding the blend above):**

| Sleeve | CAGR | Sharpe | Sortino | Vol | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ensemble@sp500` | 7.78% | 0.97 | 1.35 | 8.12% | -7.63% |
| `ensemble@midcap400` | 8.42% | 0.84 | 1.12 | 10.27% | -11.24% |
| `ensemble@nasdaq100` | 3.41% | 0.50 | 0.58 | 7.21% | -8.60% |
| Inverse-vol + target-vol blend (lev=1.0) | 4.76% | 0.68 | 0.88 | 7.21% | -6.70% |
| SPY (this run's window) | 12.79% | 0.77 | 1.06 | 17.72% | -22.09% |

**The 3-sleeve blend also does not reproduce.** Sharpe 0.68 vs. the cited
1.14 (a 0.46 shortfall, far larger than the core's 0.15 gap), and MaxDD is
*worse* than cited (-6.70% vs -5.39%, though still much shallower than SPY's
-22.09%). Every individual sleeve scored below its own cross-sleeve
comparator in prior single-window studies: `midcap400`'s solo Sharpe (0.84)
and `nasdaq100`'s solo Sharpe (0.50) are both well below the ~1.24
`nasdaq100` figure the June-27 diversification study reported. The same
eval-window-drift and gate-fail-rate issues found for the core (§3) apply
here — `nasdaq100`'s 0.50 in particular is the softest sleeve driving the
blend down.

**Per-fold gate detail (this run, sp500 core, 17 folds):**

| Fold | Test window | NDH | DSR | Gate | Anchor used? | OOS Sharpe |
|---|---|---|---|---|---|---|
| 1 | 2022-01→2022-04 | ✓ (1.00) | ✓ | PASS | No | 1.11 |
| 2 | 2022-04→2022-07 | ✓ (1.00) | ✓ | PASS | No | 0.99 |
| 3 | 2022-07→2022-10 | ✓ (1.00) | ✓ | PASS | No | 0.51 |
| 4 | 2022-10→2023-01 | ✗ (0.67, var 0.36) | ✓ | **FAIL** | **Yes** | 1.97 |
| 5 | 2023-01→2023-04 | ✓ (1.00) | ✓ | PASS | No | 0.92 |
| 6 | 2023-04→2023-07 | ✗ (0.67, var 0.43) | ✓ | **FAIL** | **Yes** | 1.37 |
| 7 | 2023-07→2023-10 | ✗ (0.67, var 0.49) | ✓ | **FAIL** | **Yes** | -2.98 |
| 8 | 2023-10→2024-01 | ✗ (0.33, var 0.54) | ✓ | **FAIL** | **Yes** | 5.24 |
| 9 | 2024-01→2024-04 | ✓ (1.00) | ✓ | PASS | No | -0.73 |
| 10 | 2024-04→2024-07 | ✓ (1.00) | ✓ (0.98) | PASS | No | 2.15 |
| 11 | 2024-07→2024-10 | ✓ (1.00) | ✓ | PASS | No | 0.16 |
| 12 | 2024-10→2025-01 | ✓ (1.00) | ✓ | PASS | No | 0.58 |
| 13 | 2025-01→2025-04 | ✓ (1.00) | ✓ | PASS | No | 0.30 |
| 14 | 2025-04→2025-07 | ✓ (1.00) | ✓ | PASS | No | 2.37 |
| 15 | 2025-07→2025-10 | ✓ (1.00) | ✓ | PASS | No | 2.27 |
| 16 | 2025-10→2026-01 | ✗ (0.67, var 0.82) | ✓ | **FAIL** | **Yes** | 1.42 |
| 17 | 2026-01→2026-04 | ✓ (1.00) | ✓ | PASS | No | 2.21 |

**12/17 gate passes (folds 4, 6, 7, 8, 16 fail on the NDH plateau-density
check; DSR passes every fold).** The recommended live combo was stable in
only **6/17 folds**. Aggregate WFE 1.23.

## 3. Findings

**Finding 1: the 1.12 baseline is not reproducible, and this run proves why
via a control that cannot lie — SPY.** SPY's historical return stream is
fixed; it cannot change between runs of the same code on the same dates.
The baseline's own citation gives SPY Sharpe 0.58 for its comparison window;
this run's SPY row (computed inside the identical harness, same
`2021-01-31→2026-04-30` window) is Sharpe **0.78**. Since SPY cannot itself
change, this is direct, unambiguous proof the two runs are evaluated over
*different* calendar windows, not the same window pre/post-fix. This
confirms, with hard evidence rather than inference, what `docs/roadmap.md`
line 117 already suspected from indirect audit: the CLI's `--eval-end`
defaults to "now" and drifts every time the WFO is re-run without an
explicit end date, and the original run that produced "1.12" was never
pinned to a specific, recorded end date. **The exact window that produced
1.12 cannot be recovered.** This is a pre-existing provenance gap, not
something the anchor fix introduced.

**Finding 2: on the best-available matched window, performance is
materially lower than cited, on both Sharpe and gate-pass rate.** OOS Sharpe
0.97 vs. 1.12 (a 0.15 shortfall — inside the range the roadmap's own
"~0.15 over a 2.5-month window shift" drift estimate would predict, which is
itself telling: the magnitude is consistent with ordinary window drift, not
proof the anchor fix specifically broke something). CAGR is the sharper
divergence: 7.8% vs. 16.3%, roughly half. Gate pass rate is 12/17 vs. the
claimed 16/17 — more than four times the anchor-fallback rate. Whether this
gap is a artifact of window drift, the anchor fix, evolving OHLCV data
(dividend/split adjustments, delisted-symbol handling — note the extensive
"possibly delisted" warnings during data load), or some mix, **cannot be
disentangled from a single re-run** — there is no surviving artifact of the
original run to diff against directly.

**Finding 3: this specific matched window (2021-01-31→2026-04-30) is
independently corroborated as internally consistent with a prior,
pre-anchor-fix measurement.** The July-13 blend study's own audit
(`docs/roadmap.md` line 117) re-ran the standalone SP500 WFO on this exact
window *before* the anchor fix and got Sharpe **0.97**, CAGR 13.4%, MaxDD
-11.0% — the Sharpe figure is identical to this post-fix run's 0.97 to two
decimal places. CAGR (7.8% vs 13.4%) and MaxDD (-7.6% vs -11.0%) differ,
consistent with the anchor fix changing which combo gets deployed in the
5 gate-fail folds (fold-level evidence in the 2026-08-18 report already
showed the pre-fix and post-fix anchor combos genuinely differ pre-2023).
That the Sharpe held essentially constant while CAGR/MaxDD moved is a real,
if modest, signal that the anchor fix has a bounded but non-zero effect on
this window — consistent with, not contradicting, the 2026-08-18 report's
own conclusion that the fix is "unlikely to explain a large share of the
previously reported 1.12."

**Net: the anchor fix is not the primary explanation for the 1.12-vs-0.97
gap. Eval-window drift is.** The Sharpe-preserving, CAGR/MaxDD-shifting
pattern across the anchor fix (Finding 3) is a small, bounded effect. The
much larger gap between 0.97 (this window, either side of the fix) and 1.12
(the original, unrecoverable window) is a provenance failure, not a
correctness bug in the harness.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

This report does not reopen or reject any strategy-level arc. `ensemble` SP500
core is not un-deployed by this finding (nothing runs live off the standalone
core number anyway — the live trader runs the 3-sleeve blend, see §5). All
prior verdicts (leveraged rotation NO-GO, leveraged trend NO-GO, ML gate
falsified, pairs-stat-arb NO-GO, IC-weighted voting NO-GO, exit-rule sweep
NO-GO, Kelly sizing NO-GO, and the nine rejected diversification-sleeve
candidates PEAD/insider-cluster-buy/congress-trades/short-interest/
short-volume-ratio/max-effect/index-deletion-fade/idio-vol/VIX-throttling)
stand unchanged in direction — see `docs/research/RESEARCH_SNAPSHOT.md`.
**What changes is the confidence in the specific numeric bar (1.12 /
16.3% / -11.0%) those nine candidates were measured against — see §5.**

## 5. Operational Roadmap: Recommended First Action

**Pin `--eval-end` explicitly in every future WFO/blend/sweep run, and
record it in whatever doc cites the resulting number.** This is the direct,
mechanical fix for the provenance gap Finding 1 exposes: `--eval-end`
defaulting to "now" (`ggTrader.lab.cli`) makes every un-pinned run
silently non-reproducible, and this report is the second time that gap has
caused real confusion (the first was the July-13 blend study's own
"1.12 turned out to be from a different window" side-finding). Every
research report, roadmap entry, and snapshot table going forward should
state the exact `--eval-start`/`--eval-end` pair alongside any cited Sharpe/
CAGR/MaxDD number — a bare "17-fold WFO, Sharpe X.XX" is not a reproducible
claim without it.

**No live-trading change is required or recommended.** The live `PaperTrader`
runs the 3-sleeve leverage-realistic blend, not the standalone SP500 core —
this report's core-only number does not directly gate that configuration.
The 3-sleeve blend result from this same run (§2: Sharpe 0.68, CAGR 4.76%,
MaxDD -6.70%, vs. the cited 1.14/-5.39% headline) is the directly relevant
comparator, and it is materially worse on Sharpe (though drawdown, while
worse than cited, is still far shallower than SPY's own -22.09% over the
same window). **Flynn's standing decision is that the live paper trader
keeps running regardless of this report's outcome** — this is a
research-provenance finding, not an incident, and does not itself trigger a
rollback. The magnitude of the blend's shortfall (a full 0.46 Sharpe below
the number that justified going live 2026-07-14) is large enough that it
should inform the next scheduled review of the live config, even though no
action is being taken here and now.

**Second-order implication for the nine rejected diversification sleeves:**
every one of them (PEAD, insider-cluster-buy, congress-trades, short-
interest, short-volume-ratio, max-effect, index-deletion-fade, idio_vol,
VIX-throttling) was rejected by comparison against "the 1.12 baseline" as
a fixed reference point. If the true, reproducible SP500-core number on the
best-available matched window is closer to 0.97, some of those NO-GO margins
were larger than the actual bar warranted — none of them scored close to
0.97 either (the closest, PEAD, matched-window standalone was 0.58; idio_vol
was 0.57), so **no rejection verdict is reversed by this finding**, but the
margin each one was rejected by should be read as somewhat smaller than the
original write-ups imply. This does not warrant re-running any of them
absent a new mechanism-level reason to revisit.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: if the anchor fix isn't the primary cause of the gap
(Finding 3), was fixing it and running this report worth the multi-hour
compute cost?** Yes, for two independent reasons. First, the anchor fix
itself is a correctness bug fix regardless of whether it moved this
particular number (as the 2026-08-18 report already argued) — a future run
on any strategy/universe/window that hits a gate-failing or halted
pre-convergence fold would otherwise silently leak future information into
a number labeled "out-of-sample." Second, and specific to this report: the
attempt to isolate the fix's effect is what *surfaced* the much larger,
previously-undetected eval-window-drift problem via the SPY control
(Finding 1) — a genuinely new finding, not a restatement of the roadmap's
prior indirect suspicion. That finding would not have been produced without
running this decisive re-run end to end.

**Contrarian question: is 0.97 actually the "true" SP500-core number, or is
there yet another confound (data vintage, delisted-symbol handling) this
report hasn't isolated?** Left genuinely open (§3, Finding 2). The OHLCV
loader emitted many "possibly delisted" warnings during this run for symbols
that have since left the index or been renamed/acquired (e.g. ABC, ATVI,
KSU, TWTR) — normal for a multi-year point-in-time universe, but a reminder
that the underlying data itself is not static between runs taken months
apart, layering a second, harder-to-isolate source of drift on top of the
eval-window issue. Not pursued further here; would require diffing the
actual OHLCV panel used for the original 1.12 run against this one, and no
artifact of that original panel survives to diff against.

### Parked Direction: eval-window pinning enforcement

Rather than relying on every future report's author to remember to pass
`--eval-end` explicitly (the discipline recommended in §5), consider making
`ggt.py lab ... --wfo/--blend` refuse to run without an explicit
`--eval-end`, or print a loud warning + record the resolved date directly
into the persisted run's metadata (`ggTrader.lab.persist`) so at minimum
future readers can look up exactly what window produced a given run_id even
if the CLI invocation itself wasn't recorded verbatim. Not implemented here
— flagged as a low-effort, high-value follow-up, not urgent enough to block
this report.
