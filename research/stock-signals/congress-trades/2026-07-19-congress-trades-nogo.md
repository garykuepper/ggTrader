# Congressional Trade Mirroring (House STOCK Act): NO-GO — Third Consecutive Eval-Window-Drift Rejection

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-19
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/congress_trades.py`
(`CongressTradeMirrorStrategy`): candidate #9 from `WEB_RESEARCH_CANDIDATES.md`.
Mechanism: mirror House members' disclosed open-market purchases
(Periodic Transaction Report transaction type "P"), holding for
`hold_days` (126/189/252 ≈ 6/9/12 months). House-only (v1) — the Senate's
`efdsearch.senate.gov` requires a stateful CSRF-protected session, a
meaningfully different engineering lift not undertaken here.

**Data infrastructure built:** `src/ggTrader/lab/house_ptr_data.py` — the
House Clerk's public bulk annual index (`financial-pdfs/{year}FD.zip`) for
filing discovery, plus per-filing PTR PDF fetch and text parsing. Verified
PTRs are genuinely digitally-generated (not scanned images) even for
2015-era filings — no OCR needed, though the extracted text is messy
(irregular whitespace, owner-code prefixes glued onto asset names, dates
sometimes glued together with no separator); a regex anchored on the
reliably-structured part of each line (ticker in parens, transaction code,
two dates, dollar range) handles this robustly. Backfilled 2015-2026:
**41,443 transaction rows across 7,578 PTR filings**, essentially clean
(3 errors total).

**This is the strongest long-window standalone result of the entire
research push this session** — OOS Sharpe **0.89 vs SPY 0.77** (a real
beat, not a near-tie), gate pass **34/40 (85%)** and fold-stability
**13/40 (33%)**, both the highest of any candidate tested. Per the
now-established discipline (first applied to PEAD, confirmed a second time
by insider cluster-buying), this triggered an immediate matched-window
retest and 4-sleeve blend test before any of those numbers were trusted.

**Result: NO-GO, both standalone and as a blend sleeve — the third
consecutive candidate to show this exact failure mode**, and the worst
blend degradation of the three. On the deployed blend's actual 2021-2026
validation window, the standalone edge evaporates entirely (Sharpe 0.36
vs SPY 0.58), and adding it as a 4th blend sleeve is the worst outcome of
any diversification-sleeve candidate tested this session (Sharpe
1.14→1.04, MaxDD -5.39%→-5.43%).

## 2. Quantitative Performance Context

| System Configuration | Window | OOS Sharpe | CAGR | MaxDD | Gate Pass | Stability |
|---|---|---|---|---|---|---|
| **congress_trades@sp500 (long window)** | 2015-06–2026-07 (40 folds) | **0.89** | 12.7% | -23.6% | 34/40 (85%) | 13/40 (33%) |
| **congress_trades@sp500 (matched window)** | 2021-01–2026-04 (17 folds) | **0.36** | 6.1% | -20.0% | n/a | 1/17 |
| SPY (matched window) | 2021-01–2026-04 | 0.58 | 13.0% | -22.1% | N/A | N/A |

| Blend Configuration (matched window, `--max-leverage 1.0`) | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| **3-sleeve baseline** (ensemble @ sp500+midcap400+nasdaq100) — deployed | **1.14** | 9.92% | **-5.39%** |
| + pead@sp500 (July 17) | 1.06 | 10.18% | -6.51% |
| + insider_cluster_buy@sp500 (July 19) | 1.12 | 9.83% | -5.45% |
| **+ congress_trades@sp500 (this report)** | **1.04** | 9.03% | -5.43% |

**Diversification check** (`scripts/congress_trades_correlation_check.py`,
same methodology as the other three): OOS return correlation to the
deployed core = **0.461**, beta = 0.437 — moderate, in the same range as
`idio_vol` (0.447) and `pead` (0.422). As with the prior two cases, the
correlation number itself isn't wrong, but a return stream's low
correlation to the core is only useful when the stream's own edge
survives to the window that matters — here it didn't.

Note the matched-window fold-stability (1/17) is dramatically lower than
the long-window's 13/40 (33%) — the same instability signature the
long-window run's healthy diagnostics masked. The long-window WFE was
`nan` (uncomputable); the matched-window WFE (0.93) looks healthy in
isolation, but that's measuring train-vs-test consistency *within* a
window where the strategy already underperforms SPY, not evidence of
genuine edge.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Congressional (House STOCK Act) trade mirroring, standalone, SP500 —
REJECTED (matched window).** OOS Sharpe 0.36 vs SPY 0.58 on the deployed
blend's own validation window — despite being the strongest long-window
result of the session (Sharpe 0.89, 85% gate pass, 33% stability), none of
that survived to the window that actually matters for a deployment
decision.

**B. Congressional trade mirroring as a 4th blend sleeve
(`congress_trades@sp500` alongside the deployed
`ensemble@sp500,midcap400,nasdaq100`) — REJECTED.** Sharpe 1.14→1.04,
MaxDD -5.39%→-5.43% — the worst blend degradation of any
diversification-sleeve candidate tested this session (worse than
`pead`'s 1.06 and `insider_cluster_buy`'s 1.12).

**C. Pattern now confirmed three times this session** (`pead` July 17,
`insider_cluster_buy` July 19, `congress_trades` July 19): a long,
favorable-looking standalone eval window reliably fails to predict
performance on the deployed blend's actual matched validation window, and
none of the three diversification-sleeve candidates with moderate
(0.38-0.46) core-correlation improved the blend. This is no longer a
one-off caveat — treat any future standalone "beats SPY" claim as
unverified until matched-window-tested, by default.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `congress_trades` in any configuration.** No live-config
changes.

**Infrastructure kept regardless of this NO-GO:** `house_ptr_data.py` and
the 41,443-row backfill (7,578 PTR filings, 2015-2026, House only) remain
reusable — the bulk annual-index discovery mechanism and text-extraction
regex are the correct general approach for any future House-disclosure
research, and could be extended to the Senate side (via the stateful
`efdsearch.senate.gov` session, not attempted here) if a future decision
specifically re-opens this line.

Move to the next lowest-effort remaining candidate per `next_steps.md`'s
effort-ordered backlog: **#8 retail-attention factors**, **#5 stealthy
shorts**, then **#4 crypto funding-rate carry** and **#14 options IV
skew** last.

**Process recommendation:** given three consecutive failures of the exact
same shape, consider whether the deployed 3-sleeve blend (Sharpe 1.14,
MaxDD -5.39%) is already close to a local optimum for *this specific
construction* (inverse-vol/target-vol blending of long-only equity
sleeves) — every new-signal-category candidate tested against it this
session, regardless of mechanism (earnings drift, insider intent,
political-access), has failed to improve it on the matched window. That
doesn't mean no future candidate could work, but it's worth flagging as a
standing observation rather than treating each new rejection as
independent evidence about only that one candidate's mechanism.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** the original candidate's own caveat flagged that
any real edge is plausibly concentrated in committee-leadership-tied
trades, not the full disclosure feed this v1 mirrors — would restricting
to committee-relevant purchases recover an edge the blanket-mirror
construction dilutes?

**Resolution:** Not worth pursuing given the pattern established across
three candidates this session. Even if a committee-restricted version
showed a better long-window number, the same matched-window/blend-test
discipline would need to be applied before trusting it — and building the
committee-assignment mapping (a new data source: historical committee
membership by Congress session, cross-referenced against each filer) is
real additional engineering for a refinement that has no reason to expect
a different outcome than what's now been observed three times in a row.
Closing this line rather than parking a refinement.
