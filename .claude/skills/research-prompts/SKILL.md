---
name: research-prompts
description: Run ggTrader's LLM strategy-research loop end to end — hand over the web-research discovery prompt ready to paste (refreshing it first if stale), merge pasted research results into docs/research/WEB_RESEARCH_CANDIDATES.md with dedupe and citation checks, and turn a chosen candidate into a filled implementation brief. Use whenever the user wants to "do another research run", "generate the research prompt", "get the Gemini/web prompt", pastes back a strategy-research report or list of trading ideas, asks to "merge these into the candidates", or says "let's build/test candidate X" and needs the implementation prompt filled in — even if they don't name the files.
---

# ggTrader research-prompt loop

New strategy ideas reach this repo through a two-stage pipeline
(`docs/next_steps.md` "Where new strategy ideas come from"):

1. **Discovery** — an external web-research LLM (Gemini / Claude web UI) runs
   `docs/research/prompts/web-strategy-research-prompt.md`. It has no repo
   access; it only knows what that prompt tells it.
2. **Ingest** — its report is pasted back here and merged into
   `docs/research/WEB_RESEARCH_CANDIDATES.md`, an **accumulating** backlog.
3. **Implementation** — one candidate is copied into
   `docs/research/prompts/local-implementation-prompt-TEMPLATE.md` and handed
   to a Claude Code session that builds and WFO-tests it.

This skill runs stages 1–3. It does **not** regenerate the ggTrader-context
sections of the two prompt files — the `research-snapshot` skill owns those,
because they must stay in sync with `RESEARCH_SNAPSHOT.md`. Figure out which
mode the user wants from what they said or pasted, then do only that mode.

## Mode A — Hand over the discovery prompt

**1. Check freshness.** The prompt is only as good as its "already tried"
context. A stale one makes the external LLM re-propose closed ideas or
benchmark against a dead baseline — the 2026-09-23 prompt still said the
core beat SPY until it was regenerated.

```bash
cd /home/flynn/ggTrader
grep -m1 'Last full regeneration' docs/research/RESEARCH_SNAPSHOT.md
grep -m1 'last synced' docs/research/prompts/web-strategy-research-prompt.md
ls docs/research/20*.md | sed 's#.*/##' | sort | tail -3      # newest closed arcs
git log -5 --format='%h %ad %s' --date=short -- docs/research src/ggTrader/lab/strategies
```

Stale if any research report or strategy-registry change is dated after the
prompt's "last synced" date, or the snapshot and prompt dates disagree. If
stale, run the `research-snapshot` skill first, then continue. If current,
say so and move on — don't regenerate for nothing.

**2. Optional focus.** If the user wants this run steered ("focus on things
that complement SPY", "crypto only", "nothing needing options data"), add a
short `## Focus for this run` section just above `## Constraints` in a
**temp copy** — never in the tracked file, whose content is shared by every
future run. Keep a focus to an area or constraint; don't name specific
strategies, because the prompt's whole value is that the external agent
generates the ideas. When the focus is about complementing a holding (e.g.
SPY), also ask for a per-candidate **"Fit with the holding"** field:
correlation to it in its drawdowns (2022, when stocks and bonds fell
together, is the test), a suggested allocation, and what it gives up in
bull markets — "diversifies" is cheap to claim, and this makes the
external agent show it.

```bash
T=$(mktemp /tmp/ggt-research-prompt-XXXX.md)
cp docs/research/prompts/web-strategy-research-prompt.md "$T"   # then edit "$T"
```

**3. Hand it over.** Give the path (the temp copy if you made one) and tell
the user to paste the *whole file* into the web tool's deep-research mode,
then paste the report back here for Mode B. Don't dump 250 lines into chat
unless asked — the file is the deliverable.

## Mode B — Ingest pasted research results

The pasted report is the least trustworthy input in this pipeline. Past
passes produced fabricated citations, a real paper wrongly called
fabricated, a misattributed venue, an overstated effect size, and a
finding with its direction flipped (`WEB_RESEARCH_CANDIDATES.md` batch
headers record each). Treat every claim as unverified until checked.

**1. Read the context you dedupe against:**
- `RESEARCH_SNAPSHOT.md` §2 (roster) and §4 (failure root causes)
- every `###` heading in `WEB_RESEARCH_CANDIDATES.md` and its status
- the "don't re-propose" list in the web prompt

Headings are not enough. Older batches park and split ideas inside
bullets — e.g. a lettered **18A** (dollar + carry + business-cycle) lives in
a bullet of the 2026-07-19 batch's deprioritized list, with no heading. On
2026-09-28 a merge added "dollar carry" as a new A21 and missed it. Before
adding any entry, **full-text search the whole backlog** (and the snapshot)
for the mechanism's key terms, cited authors and instruments, e.g.
`grep -niE "dollar|carry|UUP|Lustig" docs/research/WEB_RESEARCH_CANDIDATES.md`,
and read every hit's context.

**2. Classify each pasted candidate by mechanism, not name.** "Earnings
surprise momentum in small caps" is PEAD; "buy when insiders buy" is the
closed insider-cluster arc. For each, one of:

| Class | Action |
|---|---|
| **Duplicate of a closed arc** | Don't add. List it in the batch's "rejected on intake" note with the arc it duplicates and its report path. |
| **Same mechanism as an existing backlog entry** | Don't add a new entry. If the paste brings a genuinely new, stronger source, append it to the existing entry under a dated "Additional source (batch YYYY-MM-DD)" line. |
| **Hits a known data wall** | Add, but mark the feasibility risk up front — point-in-time analyst estimates, options-chain history, CRSP/permno panels, and long crypto funding history have each come back paid-only (snapshot §4). |
| **Genuinely new** | Add as a full entry. |

**3. Spot-check citations — and data claims.** Use web search on each new
entry's primary source: does it exist, is the venue/year right, and does the
headline number match? Before calling a citation fabricated, search the
title, the authors' pages and SSRN/NBER/arXiv — a first-pass miss has been
wrong before. Record what you checked; don't claim a check you didn't run. If
there are many candidates, check the ones heading for Group A first and say
which ones went unchecked.

Check **data-availability claims** as hard as citations — in the 2026-09-28
batch every citation was real but three data claims were false: "free"
futures settlement history (CME sells it), a FRED series FRED had truncated
to 3 years (ALFRED vintages too), and a macro index revised back to 1971
every week (point-in-time vintages only from 2011). For any series a rule
depends on, ask: is the *history* free, how deep, and is it point-in-time
(ALFRED vintages) or revised? Check `fred_series` / `ohlcv` for what is
already cached before calling something unavailable.

With more than ~8 sources, split them across 2–3 parallel background agents
(each: verify existence, metadata, the exact quoted number; quote the
supporting text; list URLs checked; write no files). Continue with the
dedupe and sanity checks while they run, and fold their results in as they
land.

**4. Write the batch.** Append one new section to the end of
`WEB_RESEARCH_CANDIDATES.md` — never rewrite or reorder earlier batches;
this file accumulates, and git history is its record.

```markdown
## YYYY-MM-DD batch — <one-line theme>

Source: <tool/model>, prompt synced <date>[, focus: <focus>]. <N> candidates
pasted, <M> added, <K> rejected on intake. Citation checks: <what was
checked, what was corrected, what was left unchecked>.

**Rejected on intake:** <name> — duplicates <arc> (`docs/research/<report>.md`); ...

### <ID>. <Name>

**Mechanism.** ... (say so if the source supports the mechanism but not this rule)
**Source(s).** ... (verification result inline: verified / corrected: ... / not found after: ...)
**Why it's plausible.** ...
**Data requirements.** ... (free/cheap, or the feasibility risk; for event signals, whether event dates are known in advance or must be forecast — using the actual date is lookahead)
**How it differs from what's already been tried.** ... (name the nearest closed arc)
**Evidence status:** ... **Rule correspondence:** ... **Implementation class:** ... **Validation stage:** Literature only.

**Status: untriaged.**
```

IDs continue the file's existing scheme: check the last batch's IDs and
continue with the next letter-group/number instead of restarting at 1.

**5. Triage for the user.** End with a short ranked shortlist (top 2–3) and
why, judged against today's bar in `RESEARCH_SNAPSHOT.md` §1 — as of
2026-09-23 that is *beat SPY, or improve SPY + sleeve*, not beat the core.
Weigh: rule correspondence, retail implementability, free point-in-time
data, and distance from the §4 failure patterns. Note any overlap with the
snapshot's §6 internal candidates.

**Quick sanity checks.** For a simple timing or sleeve rule, a few lines of
pandas on the pinned window can rule it out cheaply — label it a sanity
check, not a WFO result. Save the script and its output to
`docs/research/artifacts-<date>-web-batch/` (argparse + `main()`; yfinance
direct is fine for symbols not in `ohlcv`). Two comparisons matter most:
- **Fund any sleeve from SPY** and compare the book, not the sleeve alone —
  the live account has no idle cash.
- **For any timing rule, also run the same instruments held statically.**
  On 2026-09-28 a trend-timed TLT/GLD/DBC sleeve looked like a win until the
  static version matched it: the gain was diversification, not timing. Any
  brief that follows must then use the static version as the timing rule's
  bar.

**Alpaca tradability.** The owner is heading for real money through the
existing Alpaca path, which buys by dollar notional. For every instrument in
the shortlist, check `mcp__alpaca__get_asset`: `tradable` and `fractionable`
must be true; the attribute `ptp_with_exception` marks a publicly traded
partnership that issues a K-1 (DBC, UUP, UDN, most commodity pools).
Prefer no-K-1 equivalents — PDBC for DBC, USDU for long-dollar, SGOV/BIL for
T-bills — and note tax traps for a taxable account (gold ETFs are taxed as
collectibles). Record the result in the batch header.

## Mode C — Fill the implementation brief for a chosen candidate

**1. Locate the candidate** in `WEB_RESEARCH_CANDIDATES.md` or
`RESEARCH_SNAPSHOT.md` §6. If it's already `queued`/`testing`/`resolved`,
say so before doing anything — it may already have a report.

**2. Write the filled brief** to `docs/research/briefs/YYYY-MM-DD-<slug>.md`:
a copy of `local-implementation-prompt-TEMPLATE.md` with the "Candidate
Strategy" block replaced. Copy a backlog entry **verbatim** (citations,
ratings, verification notes) so the implementer has the full trail; for an
internal §6 idea, carry over its reasoning and success bar. Leave the
template itself untouched — `research-snapshot` expects its placeholder.

**3. Pre-fill what the implementer would otherwise rediscover**, as bullets
under "Known feasibility notes":
- **which windows are already seen**: list every window an earlier sanity
  check or report ran this rule (or a near-variant) on. Those can't serve as
  a holdout; name the truly unseen span, even if it is short
- instruments that pass the Alpaca tradability check (no K-1s), and a
  T-bill ETF (BIL) as the "off"/cash leg — lab cash earns 0%, which
  understates any sleeve that sits in cash
- for timing rules, the **static** version of the same instruments as a
  benchmark the rule must beat (not just SPY), plus a leave-one-leg-out
  diagnostic so one leg's run can't carry the verdict
- for any SPY-funded sleeve, **SPY de-risked to the same equity share with
  the rest in T-bills** as a pre-registered benchmark, and a Sharpe computed
  on returns in excess of T-bills alongside the lab's 0%-cash Sharpe. Without
  them, "lower drawdown" can't be told apart from "held less stock"
- `target_kind`: weights or signals, and the matching simulate path
- universe: an index (needs the PIT `universe_fn`) or an ETF list (no
  membership issue)
- which data already exists — `ohlcv` symbols in the DB, FRED series, etc.
  Check the DB rather than guessing (`docker exec ggtrader_db psql -U
  ggtrader -d ggtrader`). Filter `ohlcv` on `venue = 'yfinance'`: some
  tickers (SPY among them) also have `kraken_spot` tokenized-stock rows,
  and an unfiltered query double-counts days.
- the benchmarks: SPY; the instruments' own buy-and-hold for any
  timing/rotation idea; and any **existing ETF that already packages the
  strategy** (e.g. USCI for commodity carry) — if the rule can't beat
  simply buying that fund, there's no case for building it
- a pre-registered pass bar, written before any run

**4. Update the bookkeeping** the backlog's own process calls for: set the
entry's status to `queued` with a link to the brief, and add one concrete
research step to `docs/next_steps.md` pointing at it. That file is kept
1–2 steps deep and also holds ops steps — replace a research step this one
supersedes, don't touch the ops items, and don't pile on a backlog.

**5. Stop there.** Building and running the WFO is the brief's job, in a
session the user starts deliberately. Offer to start it here if they want.

## Across all modes

- Never modify live trading code, `.env`, or deployed config.
- Don't commit unless asked. Report which files changed.
- Use plain terms with the user; expand abbreviations (PIT, WFO, OOS) the
  first time they appear.
