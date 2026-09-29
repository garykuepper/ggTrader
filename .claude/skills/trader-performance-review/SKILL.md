---
name: trader-performance-review
description: Senior-quant performance review of ggTrader's live Alpaca paper book — return and drawdown vs SPY since inception and per config era, attribution across the SPY sweep sleeve, stock picks and cash, live-vs-backtest parity (missed exits, holdings outside the universe, stale holds, position count vs the cap), the 80/20 shadow book, and a ranked keep / adjust / switch recommendation with kill criteria. Use when the user asks "how is the strategy doing", "are we beating SPY", "review performance", "should we keep this strategy", "weekly review", "act as a senior quant and review", "is live matching the backtest", or is deciding whether to put real money in — even if they only ask "how did it do". For a quick is-it-running health check use daily-trader-check instead.
---

# Trader performance review

`daily-trader-check` answers "is it running?". This answers **"is it working,
and is live the strategy we tested?"** — the second question went unasked for
a week in September 2026 while live silently held ~30 positions against the
backtest's ~3 (missed crossover exits; see `docs/changelog.md` 2026-09-28).

Act as the project's senior quant (CLAUDE.md "Role"): skeptical, blunt about
NO-GO, never shading a result positive. The owner's goal (as of 2026-09-28)
is real money within ~3 months, so rank recommendations by deployability.

## 1. Context before numbers

Read, briefly: `docs/next_steps.md` (active items), the top of
`docs/changelog.md`, and `docs/research/RESEARCH_SNAPSHOT.md` §1 (the
backtested bar: SPY's pinned-window Sharpe and the deployed construction's).
`next_steps.md` also lists actions **already queued** — manual liquidations,
pending config flips, fixes awaiting their first run. Don't recommend
something already in flight; say it's queued and what to verify instead.

Confirm what code is actually live before attributing behavior to it: the
container runs its image, not the repo.
```bash
docker image inspect ghcr.io/garykuepper/ggtrader:latest --format '{{.Created}}'
git log -1 --format='%h %ci' -- src/
```
A fix committed after the image was built hasn't run yet.
Note the **config eras** from the changelog — e.g. 3-universe blend →
SP500 core revert (2026-09-22), cash sweep on (2026-08-24), exit catch-up and
`MAX_POSITIONS` cap (2026-09-29). Returns across eras measure different
strategies; split the attribution at those dates.

## 2. Book vs SPY

Query `paper_snapshots` joined to SPY closes (postgres MCP, or
`docker exec ggtrader_db psql -U ggtrader -d ggtrader`):

```sql
WITH s AS (SELECT run_date::date d, portfolio_value pv, cash, positions FROM paper_snapshots),
spy AS (SELECT timestamp::date d, max(close) c FROM ohlcv
        WHERE symbol='SPY' AND interval='1d' AND venue='yfinance' GROUP BY 1)
SELECT s.d, s.pv, s.cash, spy.c,
       coalesce((s.positions->'SPY'->>'market_value')::numeric, 0) spy_mv
FROM s LEFT JOIN spy USING (d) ORDER BY s.d;
```
Filter `venue='yfinance'` (SPY also has `kraken_spot` rows). `positions` is a
JSONB object keyed by symbol. SPY is price-only here (~1.3%/yr dividends
missing) — say so when the gap is small.

Report since inception and per era: book return, SPY return, gap, and max
drawdown. Split each era's P&L into **SPY sleeve / stocks / cash**: stocks =
pv − cash − spy_mv. The question that matters is whether the stock sleeve
adds or subtracts versus holding SPY with that money.

Cross-check the latest snapshot against the broker (Alpaca MCP
`get_account_info`). The DB value is split-corrected (e.g. MNST's unapplied
2:1), so a gap of about the corrected positions' value is expected; explain
any other gap.

## 3. Parity: is live the backtested strategy?

```bash
.venv/bin/python .claude/skills/trader-performance-review/scripts/parity_probe.py
```
Read-only. It lists every strategy position with entry date, days held, the
most recent exit bar, and flags: **missed exits** (should already be empty
since the 2026-09-29 catch-up — if not, that's a bug), **outside universe**
(no exit can ever fire — propose liquidation unless already queued),
**held > 30 days**, **no BUY record**, and **dead data**: a broker price far
from our latest close, or no tape bar for 5+ days. That last flag is how a
merger or delisting shows up (AVB merged into EQR in 2026; the broker price
froze and the position became un-exitable). Names outside the universe also
go stale on the tape, since the keepalive doesn't fetch them. Also compare the position count with `MAX_POSITIONS` in `.env`
and with the backtest's typical count (~3), and note any symbol that is both a
buy signal and an exit today (same-day sell-and-rebuy risk once slots free).

Known structural differences to mention if relevant, not rediscover: live
evaluates on a 12:45 PT partial bar, and loads ~82 bars so indicator warm-up
differs from the backtest.

## 4. Shadow book

```bash
.venv/bin/python scripts/shadow_cross_asset.py
```
The static 80% SPY + 20% TLT/GLD/PDBC paper shadow since 2026-09-29, rebased
to live NAV. Report it beside live and SPY. It is read-only apart from its
daily CSV, which goes to `results/` (git-ignored), so it is safe to run. Skip if the 2007–2011 holdout
withdrew the GO (check `docs/research/` for the holdout report).

## 5. Recommendations

Weigh live evidence against the backtest honestly: a few months of paper is
statistically close to nothing; the pinned PIT backtest is stronger evidence.
Give the **top 3** recommendations, each with:
- the call: keep (to gather information — only if live is the tested
  strategy), adjust (what, and the pre-registered test), or switch (to what,
  with its research status),
- a **kill criterion** with a number and a date,
- whether it is deployable on Alpaca (tradable, fractionable, no K-1 — see
  the research-prompts skill's tradability note) and any tax consequence for
  a real-money taxable account.

Where research is needed, point at the `research-prompts` skill (the owner
runs three external LLMs for discovery) and name the backlog entries
(`docs/research/WEB_RESEARCH_CANDIDATES.md`) or snapshot §6 items that fit.

## Output

```
## Bottom line
<one paragraph: is it working, is live the tested strategy, the headline call>

## Performance
| Period | Book | SPY | Gap | Book MaxDD |
<since inception, per era; attribution line: SPY sleeve / stocks / cash>

## Parity
<findings from the probe; each with its consequence>

## Shadow
<80/20 vs live vs SPY, or "skipped: holdout NO-GO">

## Top 3 recommendations
1. ... (kill criterion; deployability)
```
Keep the numbers traceable: say which query or script each came from. If a
step fails (DB down, broker API error), report it and continue — a partial
review with the gap stated beats none.
