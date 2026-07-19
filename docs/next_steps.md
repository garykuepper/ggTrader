# Next Steps

This file is the **only-look-1-2-steps-ahead** worklist — literally the next
thing(s) to do, nothing further out. It exists so work can be handed to a
cheaper model without them needing to re-derive context from `roadmap.md`'s
full history. When a step here is done, delete it and add the next one (if
any) — don't let this file accumulate a backlog.

## Where new strategy ideas come from (2-stage pipeline)

1. **Discovery (external, no repo access)** — run
   `docs/research/prompts/web-strategy-research-prompt.md` in a web-research
   tool (Google Gemini or Claude's web-UI research feature). Paste the
   results back into a local session and ask to merge them into
   `docs/research/WEB_RESEARCH_CANDIDATES.md` (an accumulating backlog, not
   overwritten). `docs/research/RESEARCH_SNAPSHOT.md` §6 also holds a
   smaller set of internally-derived candidate ideas (reasoned from what's
   failed, not external research) — both are valid sources.
2. **Implementation (local, full repo access)** — pick ONE candidate from
   either source, copy it into
   `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`'s
   "Candidate Strategy" section, and queue a concrete step below. That
   template's prompt is what actually drives the build-and-WFO-test work in
   this repo.

Do not skip straight to implementation-brainstorming here — this file is
for a single, already-scoped next step, not a list of ideas to pick from.

---

(Empty — as of July 17, the leverage-realistic 3-sleeve blend (Sharpe 1.14 /
MaxDD −5.39%, July 13 GO) remains wired into the live `PaperTrader` —
deployed, not just researched, and reconfirmed exactly (Sharpe 1.14, MaxDD
-5.39%) via fresh blend runs this session — no tooling drift. Since the
July 16 leveraged-ETF closures, nine research arcs have closed NO-GO:
market-neutral pairs/stat-arb (July 17, first of `RESEARCH_SNAPSHOT.md`
§6's 4 internal candidates), the MAX-effect quintile filter (July 17,
candidate #11), the free-data-only short-interest cut (July 17, candidate
#3), PEAD (July 17, candidate #12), the S&P 500 index-deletion overshoot
fade (July 17, candidate #13 — the fastest build of the session, but also
the worst outcome: MaxDD -68.7%), insider cluster-buying (July 19,
candidate #1 — the highest-effort build, ~24-hour SEC Form 4 backfill),
and Congressional trade mirroring (July 19, candidate #9 — see note below,
the third and clearest confirmation of the eval-window-drift pattern). See
`roadmap.md` §3 and
`docs/research/2026-07-16-leveraged-index-rotation-nogo.md` /
`docs/research/2026-07-16-leveraged-trend-following-nogo.md` /
`docs/research/2026-07-17-pairs-stat-arb-nogo.md` /
`docs/research/2026-07-17-max-effect-nogo.md` /
`docs/research/2026-07-17-short-interest-nogo.md` /
`docs/research/2026-07-17-pead-nogo.md` /
`docs/research/2026-07-17-index-deletion-fade-nogo.md` /
`docs/research/2026-07-19-insider-cluster-buy-nogo.md` /
`docs/research/2026-07-19-congress-trades-nogo.md`.

**The eval-window-drift pattern is now confirmed three times and should be
treated as a standing expectation, not a caveat.** PEAD (July 17) first
showed it: a long-window "beats SPY" result that erased on the deployed
blend's matched 2021-2026 window and hurt the blend (1.14→1.06).
Insider cluster-buying (July 19) confirmed it independently (long-window
near-tie + lowest core-correlation seen, matched-window Sharpe 0.39 vs SPY
0.58, blend 1.14→1.12). **Congressional trade mirroring (July 19) then
produced the STRONGEST long-window result of the entire session** (Sharpe
0.89 vs SPY 0.77, gate pass 85%, stability 33% — all highest seen) **and
still failed identically**: matched-window Sharpe 0.36 vs SPY 0.58, and
the worst blend degradation of the three (1.14→1.04). Three consecutive
diversification-sleeve candidates, three different mechanisms (earnings
drift, insider intent, political access), three identical failures.
**Any future candidate showing a standalone "beats SPY" or "low
correlation" result must be re-verified on the deployed blend's exact eval
window (and, if still promising, an actual blend test) before being
reported as promising, no exceptions** — and per the closure report's own
recommendation, consider whether the deployed 3-sleeve blend construction
itself may be near a local optimum that new equity-signal sleeves aren't
going to move, rather than treating each rejection as isolated evidence
about only that one candidate.

**Candidate #2 (analyst estimate-revision momentum) was checked and found
infeasible for an honest historical WFO** — every free source checked
(`yfinance`'s `eps_trend`/`eps_revisions`/`earnings_estimate`, this
project's other integrated sources) is current-snapshot-only, no queryable
point-in-time history; building it would need a paid I/B/E/S-style feed or
would introduce look-ahead bias. Deprioritized pending a paid-data decision
— skip past it in the effort ordering below.

**Candidate #7 (Anomaly-Driven Demand) was also checked and found
infeasible (2026-07-18)** — same class of blocker as #2. Verified against
the actual Chen & Zimmermann dataset source code
(`openassetpricing` package): the firm-level characteristics data is keyed
purely by CRSP `permno`, no ticker column, and the package's own pipeline
calls a WRDS connection directly for some signals. No free permno-to-ticker
crosswalk exists (confirmed via search — CRSP ticker-history identity data
is a WRDS subscription product). A name-matching heuristic crosswalk was
considered and rejected as a silent-data-corruption risk given ticker
reuse over the dataset's multi-decade span. Deprioritized pending a
WRDS-access decision — skip past it too.

**Candidate #8 (retail-attention factors) is built and tested but PAUSED,
not resolved (2026-07-19)** — `retail_attention` strategy + Google Trends
pipeline (`google_trends_data.py`) are complete, 19 tests passing. The
live backfill (`scripts/google_trends_backfill.py`) hit a Google rate-limit
lockout (429s that didn't clear on retry) partway through — a quick
feasibility spot-check beforehand under-sampled the real constraint. Retry
the backfill in a later session with more conservative pacing (current
script already uses 2s/request; consider longer, or spread across multiple
sessions) once the block has likely cleared. Do not attempt IP rotation.
Skip past it for now — it's not blocking the effort-ordered queue below,
just not runnable this session.

Effort-ordered remaining backlog: next up is **#5 stealthy shorts** (same
scraping/classification tier as the now-closed #1 and #9), then **#4
crypto funding-rate carry** and **#14 options IV skew** (different asset
class/execution model, or paid-data risk) last. Once #8's Google Trends
block clears, revisit it before or alongside those. No Step 4 queued yet:
pick the next one, copy it into
`docs/research/prompts/local-implementation-prompt-TEMPLATE.md`, and queue
it here. See the pipeline above for the mechanics.)
