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
deployed, not just researched. Since then, five research arcs have closed
NO-GO: the two leveraged/inverse-ETF timing arcs (July 16), market-neutral
pairs/stat-arb mean reversion (July 17, the first of `RESEARCH_SNAPSHOT.md`
§6's 4 internal candidates), the MAX-effect/lottery-demand quintile filter
(July 17, candidate #11 — rejected both standalone and as a diversification
sleeve), and the free-data-only cut of #3 short-interest/cost-to-borrow
(July 17 — new FINRA data pipeline built, `short_interest_data.py`, 150
settlement dates backfilled; rejected as a noise/overfitting result,
consistent with the literature's own large-cap-insignificance caveat). See
`roadmap.md` §3 and
`docs/research/2026-07-16-leveraged-index-rotation-nogo.md` /
`docs/research/2026-07-16-leveraged-trend-following-nogo.md` /
`docs/research/2026-07-17-pairs-stat-arb-nogo.md` /
`docs/research/2026-07-17-max-effect-nogo.md` /
`docs/research/2026-07-17-short-interest-nogo.md`.

**Candidate #2 (analyst estimate-revision momentum) was checked and found
infeasible for an honest historical WFO** — every free source checked
(`yfinance`'s `eps_trend`/`eps_revisions`/`earnings_estimate`, this
project's other integrated sources) is current-snapshot-only, no queryable
point-in-time history; building it would need a paid I/B/E/S-style feed or
would introduce look-ahead bias. Deprioritized pending a paid-data decision
— skip past it in the effort ordering below.

Effort-ordered remaining backlog: next up are **#12 PEAD**, **#13
index-deletion overshoot fade**, and **#7 Anomaly-Driven Demand** (new feed
+ real engineering), then **#1 Form 4 insider cluster buying**, **#9
Congressional STOCK Act trades**, **#8 retail-attention factors**, **#5
stealthy shorts** (scraping/classification effort), then **#4 crypto
funding-rate carry** and **#14 options IV skew** (different asset
class/execution model, or paid-data risk) last. No Step 4 queued yet: pick
the next one, copy it into
`docs/research/prompts/local-implementation-prompt-TEMPLATE.md`, and queue
it here. See the pipeline above for the mechanics.)
