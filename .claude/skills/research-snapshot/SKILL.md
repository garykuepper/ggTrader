---
name: research-snapshot
description: Regenerate the ggTrader strategy-research snapshot (docs/research/RESEARCH_SNAPSHOT.md) and the ggTrader-context sections of the two research prompt templates (docs/research/prompts/web-strategy-research-prompt.md, docs/research/prompts/local-implementation-prompt-TEMPLATE.md) from source of truth, or answer "what have we tried" questions from that snapshot. Use whenever the user says "create a snapshot", "snapshot", "update/refresh the research snapshot", "regenerate the research prompts", "sync the research docs", or has just closed a research arc (GO or NO-GO report committed) and the roster needs to catch up. Also use for "what have we tried", "has X been tested", "summarize research history" — read the snapshot and answer, regenerating only if it's stale or the user asks.
---

# Research Snapshot Generator

Keeps three living documents in sync with the repo's actual research record:

- `docs/research/RESEARCH_SNAPSHOT.md` — the verdict roster and current bar.
- `docs/research/prompts/local-implementation-prompt-TEMPLATE.md` — its
  ggTrader-context sections.
- `docs/research/prompts/web-strategy-research-prompt.md` — its
  ggTrader-context sections.

They exist because hand-maintained docs rot: `docs/roadmap.md` stopped
tracking reality months ago while its prose kept sounding current. This
skill re-derives claims from source rather than trusting the prior version.
Outputs are overwritten in place; git history is the record.

This project separates two research jobs, and so does this skill:
- **Discovery** (external, no repo access): the web prompt is pasted into a
  web-research tool. Its findings land in
  `docs/research/WEB_RESEARCH_CANDIDATES.md`, an accumulating backlog that
  the `research-prompts` skill merges. **This skill never regenerates that
  file**; it only reads it.
- **Implementation** (local, full repo): the local template drives one
  already-chosen candidate through build + WFO.

## 0. Pick the mode

**Question mode** — the user asked what's been tried, whether X was tested,
or for a history summary. Read the snapshot, check its "Last full
regeneration" date against `git log` for newer research reports, and
answer with citations. If newer closed arcs exist, say the snapshot is
behind and offer to update it. Don't rewrite three files to answer a
question.

**Update mode** — decide full vs incremental:

```bash
ANCHOR=$(git log -1 --format=%h -- docs/research/RESEARCH_SNAPSHOT.md)
git log --oneline $ANCHOR..HEAD
git status --short docs/research src/ggTrader/lab/strategies docs/next_steps.md
```

Also check whether the snapshot itself has uncommitted edits. If it does, an
update already happened after the anchor; read the file's own header to see
what it covers.

- **Incremental** is fine when the last *full* regeneration is ≤ 7 days old
  and the delta above is small enough to enumerate completely: every new
  report, registry entry, and baseline-moving commit. Seven days is a
  heuristic. The point is that a short window means you can name every
  change instead of hoping you found them.
- **Full** when the user asks for it, the last full run is older than that,
  or the delta includes a baseline-moving change (a bias or data fix, a
  re-baseline, a live-strategy swap). Those invalidate claims all over the
  file, not just in new rows.

Either way, read the **entire** current snapshot, not just the sections you
expect to touch. §1 (baseline) and §5 (known gaps) go stale fastest, because
they describe live state, and a "not yet verified" note from last week is
often resolved or replaced by now.

## 1. Survey source of truth

In full mode, cover all of this. In incremental mode, cover the delta plus
anything in §1/§5 whose claim could have moved.

1. `docs/next_steps.md` and `docs/roadmap.md`: the stated state. Note
   explicitly if either lags what you find below.
2. Every `docs/research/*.md` (skip `TEMPLATE-research-report.md`,
   `WEB_RESEARCH_CANDIDATES.md`, `prompts/`, `briefs/`). Each is an
   authoritative, verdicted arc; summarize with its real numbers.
3. `STRATEGY_REGISTRY` in `src/ggTrader/lab/strategies/__init__.py`. Classify
   every entry GO / NO-GO / ambiguous with a citation. If a verdict traces
   only to deleted/pre-rewrite code, call it a *soft* rejection.
4. `git log --oneline --all -i --grep="NO-GO\|NOGO\|reject\|validated"`, to
   catch anything closed only in a commit message.
5. **The deployed construction, not just the strategy.** Numbers only, not
   cron/ops mechanics. Check:
   - what live actually trades (signal source, sizing, cash handling such
     as an idle-cash sweep, leverage);
   - whether that matches what the validating arc measured (parameters,
     position count, exposure). Read the sizing code on both paths
     (`lab/simulate.py` vs `paper/risk.py`) rather than trusting prose: the
     snapshot once called both "flat 3.3%" when the backtest used 3% of
     remaining cash;
   - any mismatch, which goes in §5.

   A strategy's standalone Sharpe answers the wrong question if the book
   also holds an index ETF. This check is what surfaced, on 2026-09-24, that
   live ran ~4× the validated stock exposure.
6. `WEB_RESEARCH_CANDIDATES.md` (read-only): which entries are resolved
   (fold them into the roster) and what's queued, so the don't-re-propose
   list never contradicts something queued there.

For a full run, delegate the broad reading to a couple of parallel Explore
subagents (e.g. one for the reports, one for the registry and git log). That
keeps your own context for the synthesis.

## 2. Regenerate `RESEARCH_SNAPSHOT.md`

Header: set `Last full regeneration` (full mode) or add `Last incremental
update` (incremental mode), honestly. The intro paragraph says in 2–4
sentences what this run changed.

Fixed sections, every claim cited (report path or commit hash):

1. **Current validated baseline**: the bar every idea must clear or
   complement. Include the deployed construction's numbers (§1 step 5) next
   to the standalone strategy's, plus SPY on the same pinned window. Also
   state how a candidate is judged (standalone vs sleeve).
2. **Full roster table**: one row per tried lever (registry entries plus
   non-registry levers such as sizing, gating, or construction studies). Columns: verdict,
   key metric, one-line mechanism, source.
3. **What worked, structurally**: methodology wins (harness fixes, bias
   corrections, validation conventions). These outlive any single verdict.
4. **What failed, grouped by root cause**: cluster by *why*, not by name.
   This is the most useful section for a fresh agent because it teaches the
   pattern. It is also the canonical don't-re-propose list for both prompts.
5. **Known documentation gaps**: stale docs, unverified live state,
   live-vs-validated mismatches, and harness flaws found but not yet fixed.
   Drop items that are now resolved; don't let this become a graveyard.
6. **Internally-derived candidate ideas**: ≥ 3, ranked (see §4 below).

## 3. Regenerate the prompt context sections

**Local template**: self-contained, for a Claude Code session with repo
access. Regenerate everything except the "Candidate Strategy" placeholder
block at the top:
- the role framing;
- the don't-re-propose list, condensed from snapshot §4;
- the baseline, with numbers identical to snapshot §1;
- the constraints (data, the `target_kind` overlay gotcha, PIT membership);
- the deliverables (dated report, GO or NO-GO, no live-config changes
  without a separate ask).

**Web prompt**: portable plain text. No file paths, no tool names, no
repo-internal strategy names unless useful as labels. Regenerate:
- the plain-English "currently deployed" summary;
- the don't-re-propose list in plain language;
- the retail constraints: broker instruments, no futures/FX, 1.0x, one daily
  decision before the close.

Never seed it with strategy recommendations; the external agent's job is to
generate those. Its output-format section is stable, so leave it alone.

Bump each file's "last regenerated/synced" date.

## 4. Refresh the internal candidate ideas (snapshot §6)

The current ranked set lives in the snapshot's own §6. Start there, not
from memory. Each run:
- **Remove** ideas that were tried this cycle (they move to the roster)
  and ideas a new result makes moot. Record them under "Dropped this run".
- **Add** ideas a new result implies. A NO-GO's root cause often points to
  an adjacent lever, and a live-vs-validated mismatch can itself be the top
  item: validating the book as it runs comes before adding to it.
- **Re-rank** by expected payoff × survival odds ÷ effort, restricted to
  what's executable now (data in hand, broker instruments, no leverage).
- Keep ≥ 3. Point to `WEB_RESEARCH_CANDIDATES.md` for external ideas rather
  than copying them, but you may reference specific entries (by ID) when
  one fits the internal reasoning.

Open §6 with "What changed this run (date)". Condense the previous run's
note to one "Carried from (date)" paragraph, and drop anything older.

Historical note: earlier versions of this skill hardcoded pairs/stat-arb,
PEAD, options overlays, and crypto carry as the standing list. All four
were tried or ruled infeasible by 2026-07, so don't resurrect them without
a new mechanism.

## 5. Consistency check, then report

Before reporting, check that the three files agree. Grep the headline
numbers (SPY Sharpe, core Sharpe, deployed-construction Sharpe) across all
three, and confirm each §4 root-cause cluster appears in both prompts.
Mismatched numbers between the snapshot and a prompt are the most common
defect.

Then report to the user in a few lines:
- `git diff --stat` on the three files;
- new arcs closed, baseline numbers moved, gaps added or resolved, and
  candidates added, dropped, or re-ranked;
- full vs incremental, and why.

Don't paste file contents. Don't describe `WEB_RESEARCH_CANDIDATES.md` as
regenerated.

**Don't commit.** This project commits only when the user asks, so offer
it. If they say yes, the message style is
`docs(research): <date> snapshot regen — <what moved>`.
