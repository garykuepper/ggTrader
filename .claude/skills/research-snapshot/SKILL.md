---
name: research-snapshot
description: Regenerate the ggTrader strategy-research snapshot (docs/research/RESEARCH_SNAPSHOT.md) and the ggTrader-context sections of the two research prompt templates (docs/research/prompts/web-strategy-research-prompt.md, docs/research/prompts/local-implementation-prompt-TEMPLATE.md) from source of truth. Use when the user asks to "update the research snapshot", "regenerate the research prompts", "what have we tried", "summarize research history", or wants these briefs refreshed after new research closes.
---

# Research Snapshot Generator

Regenerates living documents from source of truth — never hand-edit them
incrementally, and never trust a prior version's claims without re-verifying.
`docs/roadmap.md`/`docs/next_steps.md` are hand-maintained and have already
gone stale (roadmap's timeline section stopped updating while its own prose
kept moving) — this skill exists specifically to avoid that failure mode by
re-deriving everything fresh each run.

This project separates two distinct research jobs — keep them separate in
what you generate:
- **Discovery** (external, no repo access) — `web-strategy-research-prompt.md`
  is pasted into a web-research tool (Gemini/Claude web UI) to find
  genuinely new ideas from outside literature/practice. Its findings land in
  `docs/research/WEB_RESEARCH_CANDIDATES.md`, an **accumulating backlog**
  fed by the user pasting results back — **this skill does not regenerate
  that file**, only the ggTrader-context section of the prompt that
  produces its inputs.
- **Implementation** (local, full repo access) — `local-implementation-prompt-TEMPLATE.md`
  is a template for building/WFO-testing one specific, already-chosen
  candidate in this repo.

Regenerated outputs (this skill's actual job) are **overwritten in place**
(not dated/appended) — git history is the record of how they evolved. After
regenerating, report a short diff summary to the user (what changed since
the last version).

## 1. Survey source-of-truth files

Read fresh every run, in this order:

1. `docs/roadmap.md` and `docs/next_steps.md` — current stated state. Note
   explicitly if either looks stale relative to what you find in steps 2-4
   (e.g. a closed arc in `docs/research/` or a recent commit not reflected
   here).
2. Every file in `docs/research/*.md` (skip `TEMPLATE-research-report.md`,
   `WEB_RESEARCH_CANDIDATES.md`, and anything under `docs/research/prompts/`)
   — each is an authoritative, already-verdicted arc. Summarize each in 1-3
   sentences with its real numbers.
3. `src/ggTrader/lab/strategies/__init__.py`'s `STRATEGY_REGISTRY` — full
   entry list. For each entry, classify GO / NO-GO / ambiguous using evidence
   from step 2, `docs/roadmap.md`, or git log (step 4) — never assert a
   verdict you can't cite. If a verdict traces only to deleted/superseded
   code (check for a "lab-first rewrite" or similar architectural rewrite in
   git log), flag it as a *soft* rejection, not hard evidence against the
   current implementation.
4. `git log --oneline --all --grep="NO-GO\|NOGO\|reject\|validated" -i` —
   catch anything closed via commit message only, not restated in docs.
5. The current live-baseline's **numbers only** (not deployment/cron
   mechanics — that's out of scope) — check whatever config currently drives
   the deployed strategy (position sizing, universe blend weights, leverage
   cap) against the OOS metrics cited in the arc that validated it.
6. `docs/research/WEB_RESEARCH_CANDIDATES.md` — read-only context (do not
   regenerate it), just to check if any `resolved` entries there should be
   folded into the roster in step 2's output, and to avoid the "don't
   re-propose" list contradicting something already queued/testing there.

For anything ambiguous or requiring cross-referencing many files, delegate to
an Explore subagent rather than grepping everything inline yourself — keeps
this skill's own context budget small and lets you run the actual survey in
parallel across a few focused agents (e.g. one for docs, one for the registry
+ git log).

## 2. Regenerate `docs/research/RESEARCH_SNAPSHOT.md`

Fixed structure, every section populated from step 1's findings (cite
sources — doc path or commit hash — for every claim):

1. **Current validated baseline** — mechanism + OOS numbers (Sharpe/CAGR/
   MaxDD vs. SPY) for whatever is actually validated and deployed today.
   State it as "the bar every new idea has to clear or complement." Strategy
   framing only, not ops mechanics.
2. **Full roster table** — one row per tried lever (every `STRATEGY_REGISTRY`
   entry plus any non-registry levers like sizing/gating/regime experiments):
   verdict (GO / NO-GO / ambiguous), key metric, one-line mechanism, source
   citation.
3. **What worked, structurally** — methodology wins, not just strategy wins
   (e.g. WFO/gate framework changes, sizing conventions, validated
   diversification techniques) — these are often more durable than any one
   strategy's verdict.
4. **What failed, grouped by root cause** — cluster rejected levers by *why*
   they failed (e.g. "signal-strength-as-timing doesn't survive real
   cost/decay", "added complexity underperforms the simpler default",
   "diversification is correlation-capped without leverage realism") rather
   than listing them flatly. This is the most valuable section for a fresh
   agent — it teaches the failure *pattern*, not just the instance. This is
   also the canonical source for the "don't re-propose" list embedded in
   both prompt files — keep those three in sync.
5. **Known documentation gaps** — anything found stale/ambiguous in step 1,
   so the reader doesn't cite it as settled.
6. **Internally-derived candidate ideas** (>= 3, feasibility-ordered) — see
   §4 below for how to refresh these each run. Explicitly labeled as
   internal (reasoned from what's failed), distinct from
   `WEB_RESEARCH_CANDIDATES.md`'s externally-sourced backlog — point to that
   file rather than duplicating its content here.

## 3. Regenerate the ggTrader-context sections of both prompt templates

**`docs/research/prompts/local-implementation-prompt-TEMPLATE.md`** — keep
self-contained (embed context directly; it's for a Claude Code session with
full repo access, free to explore further but shouldn't need to for the
brief to make sense). Regenerate everything EXCEPT the "Candidate Strategy"
placeholder section at the top (that's filled in per-use, not by this
skill — leave it as the placeholder block):
1. Role/mission framing (senior quant researcher, honest-OOS-or-nothing,
   mirrors `agents.md`'s "Role" section) + instruction to benchmark against
   both SPY and the current baseline, and against buy-and-hold for any
   timing/rotation idea.
2. Condensed "don't re-propose" list, pulled from the snapshot's §4.
3. Current baseline, numbers matching the snapshot exactly.
4. Constraints (data available, `target_kind` weights-vs-signals overlay
   gotcha, point-in-time universe membership).
5. Deliverable expectations (dated report per the template, GO or NO-GO
   either way, no live-config changes without a separate ask).

**`docs/research/prompts/web-strategy-research-prompt.md`** — keep portable
(plain text, no file paths framed as instructions, no tool/MCP references,
explicit "don't write code / don't assume repo access"). Regenerate:
1. The "currently deployed and working" summary (plain-English version of
   the current baseline, no code/file references).
2. The condensed "don't re-propose" list, same root-cause grouping as the
   snapshot's §4 but described in plain language (no ggTrader-internal
   names like `ensemble_ic` unless useful as a label) — external readers
   don't have this repo's context.
3. Constraints (retail/home-lab feasible, free-or-cheap data, execution
   realism).
Do NOT add fixed strategy recommendations to this file — its whole point is
that the *external* agent generates recommendations; seeding it with ideas
defeats the purpose. Its "required output format" section is stable and
shouldn't need to change run over run.

## 4. Refresh the internally-derived candidate ideas (snapshot §6)

Current set (adjust wording/ranking as the codebase evolves, but preserve
the reasoning for each):

- **Rank 1 — Market-neutral pairs / stat-arb mean reversion.** Long/short
  dollar-neutral pairs within correlated clusters, reverting on spread
  z-score. First market-neutral construction attempted if everything on
  record so far is long-only directional — the diversification value is the
  point even if standalone Sharpe is modest. Cheapest to execute if it only
  needs OHLCV already in the DB.
- **Rank 2 — Post-earnings-announcement drift (PEAD).** A well-documented
  equity anomaly — check whether it's been tried; if not, it's the first
  fundamental/event-driven signal in a history of otherwise pure
  price-action strategies. Needs earnings-date + surprise data — first task
  is verifying data source coverage/quality before committing research time.
- **Rank 3 — Options-derived conditioning signal or covered-call income
  overlay.** Different instrument class entirely. Explicitly data-constrained
  (check current options-data access before ranking this higher) — flag as
  needing a feasibility check first, not a promise it's buildable today.
- **Rank 4 (lower priority, if still parked) — Revisit any parked
  capital-gated or data-gated ideas** (e.g. crypto-carry) now that time has
  passed — check whether the original gating condition (capital threshold,
  data backfill) has been resolved.

On subsequent runs: re-derive this list by checking (a) which of the above
have since been tried (move to the roster, drop from recommendations, add a
genuinely new idea in its place) and (b) whether anything closed this cycle
suggests a new direction (e.g. a NO-GO's root cause implies an adjacent idea
worth trying). Keep at least 3 ranked candidates at all times. Do not merge
this list with `WEB_RESEARCH_CANDIDATES.md`'s entries — they stay in
separate files with separate maintenance models (this one regenerated,
that one accumulated).

## 5. Report to the user

After writing the files, run `git diff --stat` on them (or note "new file"
if first run) and summarize in a few lines: what changed since the last
version — new arcs closed, baseline numbers moved, recommendations
added/dropped/re-ranked. Don't dump full file contents into chat; the user
can read the files. Explicitly do not mention `WEB_RESEARCH_CANDIDATES.md`
as "regenerated" — if the user wants that updated, that's a separate,
manual merge-from-pasted-results action, not part of this skill's run.
