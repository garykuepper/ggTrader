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
deployed, not just researched. Since then, three research arcs have closed
NO-GO: the two leveraged/inverse-ETF timing arcs (July 16) and market-neutral
pairs/stat-arb mean reversion (July 17, the first of `RESEARCH_SNAPSHOT.md`
§6's 4 candidates). See `roadmap.md` §3 and
`docs/research/2026-07-16-leveraged-index-rotation-nogo.md` /
`docs/research/2026-07-16-leveraged-trend-following-nogo.md` /
`docs/research/2026-07-17-pairs-stat-arb-nogo.md`.

`docs/research/WEB_RESEARCH_CANDIDATES.md` is no longer empty — its first
web-research run (July 16) landed 14 externally-sourced candidates,
triaged into first/second/third-wave priority. No Step 4 queued yet: pick
one of the first-wave candidates (#1 Form 4 insider cluster buying, #2
analyst estimate-revision momentum, #7 Anomaly-Driven Demand, or the
free-data-only cut of #3 short-interest/cost-to-borrow) or one of
`RESEARCH_SNAPSHOT.md` §6's remaining 3 internal candidates (PEAD,
options-derived signal, revisit parked crypto-carry — note the first two of
those now have more-detailed, citation-backed counterparts in
`WEB_RESEARCH_CANDIDATES.md` #12/#14, cross-referenced there), copy it into
`docs/research/prompts/local-implementation-prompt-TEMPLATE.md`, and queue
it here. See the pipeline above for the mechanics.)
