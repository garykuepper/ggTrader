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
July 16 leveraged-ETF closures, ten research arcs have closed NO-GO:
market-neutral pairs/stat-arb (July 17, first of `RESEARCH_SNAPSHOT.md`
§6's 4 internal candidates), the MAX-effect quintile filter (July 17,
candidate #11), the free-data-only short-interest cut (July 17, candidate
#3), PEAD (July 17, candidate #12), the S&P 500 index-deletion overshoot
fade (July 17, candidate #13 — the fastest build of the session, but also
the worst outcome: MaxDD -68.7%), insider cluster-buying (July 19,
candidate #1 — the highest-effort build, ~24-hour SEC Form 4 backfill),
Congressional trade mirroring (July 19, candidate #9 — see note below, the
third and clearest confirmation of the eval-window-drift pattern), and
short-volume-ratio/"stealthy shorts" free-data cut (July 19, candidate #5
— a clean "no signal at this fidelity" rejection, not an eval-window or
overfitting one). See `roadmap.md` §3 and
`docs/research/2026-07-16-leveraged-index-rotation-nogo.md` /
`docs/research/2026-07-16-leveraged-trend-following-nogo.md` /
`docs/research/2026-07-17-pairs-stat-arb-nogo.md` /
`docs/research/2026-07-17-max-effect-nogo.md` /
`docs/research/2026-07-17-short-interest-nogo.md` /
`docs/research/2026-07-17-pead-nogo.md` /
`docs/research/2026-07-17-index-deletion-fade-nogo.md` /
`docs/research/2026-07-19-insider-cluster-buy-nogo.md` /
`docs/research/2026-07-19-congress-trades-nogo.md` /
`docs/research/2026-07-19-short-volume-ratio-nogo.md`.

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

**Candidate #4 (crypto funding-rate carry) was checked and found
infeasible for an honest WFO (2026-07-20)** — same class of blocker as #2
and #7, closing the long-parked internal crypto-carry line
(`RESEARCH_SNAPSHOT.md` §6 Rank 4) on the same evidentiary basis. Kraken
Futures is the only real venue (Binance.US has no perpetual futures at
all, spot-only). Kraken's historical-funding-rates data — verified via
both `ccxt` and Kraken's own native API directly — only retains a rolling
~1 year of hourly records, nowhere near the 12mo-train + 3mo-test-per-fold,
many-folds depth this project's WFO requires. Free third-party aggregators
don't offer bulk historical download; paid ones would still fall short of
this project's usual eval-window convention. Deprioritized pending either
a much longer free-data source or a paid-data decision.

**Candidate #14 (options IV skew) was checked and found infeasible
(2026-07-19)** — same class of blocker as #2/#7/#4. Verified concretely:
`yfinance`'s `option_chain()` takes only a future-expiry selector, no
historical/as-of parameter (current-snapshot-only, like #2's
`eps_trend`/`eps_revisions`); Alpaca's option-bars API (already integrated
here) returned zero data for a real contract in both 2022 and 2024 windows
— Alpaca's options market data only starts in 2024, and even then there's
no historical listed-contracts feed to reconstruct past chains from. Deep
historical options-chain data remains a paid-vendor problem (OptionMetrics
IvyDB), exactly as originally flagged, and the signal may just be
re-deriving #3's already-NO-GO'd borrow-cost signal through a noisier
instrument anyway. No code built.

**First 14-candidate batch is fully resolved** (10 NO-GO, 4 infeasible —
#2/#7/#4/#14, 1 paused — #8). See prior paragraphs above for detail.

**2026-07-19 batch: 25 candidates, all non-US-equity asset classes** —
merged into `WEB_RESEARCH_CANDIDATES.md` from a discovery pass explicitly
scoped away from equities per the pivot recommendation in
`RESEARCH_SNAPSHOT.md` §4/§6 (nine consecutive equity diversification
sleeves had failed). Covers FX, commodities, Treasuries/rates, and crypto.

**Corrected and reorganized a second time (2026-07-19)** following an
independent review of the reformatted draft — the review caught real
citation errors (one paper wrongly called "fabricated," a venue
misattribution, a magnitude error, a reversed-then-re-reversed finding
now flagged contested) and replaced the single "Confidence" score with
four independent ratings (Evidence status / Rule correspondence /
Implementation class / Validation stage — see `WEB_RESEARCH_CANDIDATES.md`
for the full explanation). Candidates are now organized by strategy type
— **A. Active strategy replication queue** (A1-A9), **B. Risk/exposure
overlays** (B1-B4), **C. Portfolio-construction methods** (C1-C2), **D.
Parked hypotheses** — rather than by source-prestige tier. Every
candidate is still at Validation stage: Literature only.

**Most consequential change: the previous #1 pick (direct CIP/basis
harvesting) is explicitly demoted for this project's home-lab/ETF
workflow** — the literature validates the CIP deviation's existence and
cause, not "tilt an FX-ETF carry portfolio using the basis as a signal,"
which is a separate, untested extension. FX ETFs can't implement the
actual documented arbitrage (needs forwards/swaps/funding access this
project doesn't have). **The dynamic FX hedge overlay (A1) is now the
clear top pick** instead.

**Effort-ordered next-up, per the register's own "home-lab / mostly-ETF
workflow" ranking** (this project's actual setup):
1. **A1 — Dynamic FX hedge overlay (carry+value+trend).** Free/cheap spot
   FX, rate differentials, CPI data. Cleanest source-to-rule
   correspondence in the whole register — published, high-pedigree paper
   (Castro/Hamill/Harber/Harvey/Van Hemert 2025).
2. **A7 — Pre-FOMC long-Treasury drift.** Free (TLT/IEF/EDV + FOMC
   calendar). Promoted from parked status after its citation was replaced
   with a verified, current (June 2026), directly-on-point paper (Pan &
   Peng) — the prior version's citation was an unverified draft.
3. **A3 — Commodity medium-term trend.** Free/cheap ETF proxies. Note this
   is now a standalone candidate — the register split what was one
   combined "carry+trend+basis reversal" idea into three independently-
   evidenced sub-signals (A2/A3/A4); don't assume they combine as
   originally proposed without testing each separately.
4. **A5 — Treasury term-structure factors, static/ETF-approximation
   version only.** Free (TLT/IEF/SHY). Must be explicitly labeled an
   approximation, not a replication — the underlying paper (now upgraded
   to "accepted at Review of Finance") specifies 4 factors, a 3-ETF
   version only captures 3.
5. **A8 — Headline/LLM sentiment on small/mid-cap equities.** Needs
   credible point-in-time headline data — do not proceed on a look-ahead-
   biased feed. Authors themselves warn of a crowding/decay effect as LLM
   use spreads.
6. **B1 — Stablecoin stress signal.** Scope as a risk/leverage-reduction
   overlay, not a standalone short — the underlying papers show stress
   raises *two-sided* jump risk, not a reliably negative one.

Pick ONE (A1 is the natural first pick), copy its entry from
`WEB_RESEARCH_CANDIDATES.md` into
`docs/research/prompts/local-implementation-prompt-TEMPLATE.md` — **also
apply the register's "Minimum validation standard" 7-point checklist**
(frozen rule, point-in-time data, full costs, nested OOS, multiple-testing
correction, stress-period behavior, shadow portfolio) on top of this
project's usual WFO/NDH/DSR gate — and queue it here as the concrete next
step. Also still open, lower priority: retry #8's Google Trends backfill
once the rate-limit has likely cleared.)
