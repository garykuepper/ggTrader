# Headline/LLM Sentiment on Small/Mid-Cap Equities (Candidate A8): NO-GO (PROVISIONAL)

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-24
**Amended:** 2026-07-25 — verdict downgraded to **provisional**
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

> **⚠️ AMENDMENT (2026-07-25) — this verdict is provisional, not settled.**
> The implementation audit (`docs/research/2026-07-25-strategy-implementation-audit.md`)
> found three independent reasons this test is underpowered, none of them a
> code defect:
>
> 1. **3 folds.** Every other verdict in the research record rests on 20–54
>    folds. One of A8's three gate-failed, leaving two usable observations.
> 2. **Unrepresentative sample.** The alphabetically-first-50 midcap400 pilot
>    is ~2× overweight Industrials (30% vs 17% in a random baseline from the
>    rest of the universe) and underweight Consumer Cyclical (10% vs 15%).
>    This is the check §6 below called "the one cheap thing worth verifying" —
>    now done, and it does show a skew.
> 3. **7–8 positions.** Only 50 of ~400 midcap400 names had sentiment
>    coverage, so the top-quintile bucket resolved to ~7 stocks (vs 92-99 for
>    comparable candidates). That concentration mechanically explains both the
>    fold-to-fold swing (0.27 / 1.44 / -0.38) and the NDH variance-gate
>    failure, without needing a signal-quality explanation.
>
> **What still holds:** the direction. Benchmark choice does not rescue it —
> over its own OOS window midcaps performed comparably to SPY (MDY Sharpe
> 1.39 / CAGR 22.9%; IJH 1.41 / 23.3%; SPY 1.61 / 21.6%), so A8's 0.25 / 4.8%
> underperforms its own universe just as badly.
>
> **Also note:** the SPY benchmark figure of 1.14 quoted below is understated
> by ~29% by the data bug in §2.0 of the audit (true value ≈ 1.61). Both
> strategy and benchmark are deflated by the same factor, so the comparison
> direction is unaffected.
>
> **Treat as: unfavourable direction, low confidence.** Not a structural
> refutation of the mechanism.

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/headline_sentiment.py`
(`HeadlineSentimentStrategy`): candidate A8 from `WEB_RESEARCH_CANDIDATES.md`'s
cross-asset register — the last item in that register's home-lab priority queue,
picked up after A1/A7/A3/A5 (FX hedge overlay, pre-FOMC drift, commodity trend,
Treasury term-structure) were all rejected. Source: Lopez-Lira & Tang, *"Can
ChatGPT Forecast Stock Price Movements?,"* SSRN/arXiv, latest public revision
October 28, 2025 (working paper; publication status not independently
confirmed) — an LLM classifier on news headlines generates a daily long/flat/
short signal, trading the delayed post-publication drift, strongest in smaller
names and after negative news.

**Mechanism implemented.** `HeadlineSentimentStrategy` scores every unique
headline for a symbol over a trailing `lookback_days` window (swept: 10/20/40)
via an LLM call (0/±1 score), averages per symbol, ranks descending, and takes
the top `1/quintile` bucket (swept: quintile 4 or 5) equal-weighted. Sentiment
scores are joined at load time from `headline_sentiment_scores` back out to
every (symbol, created_at) row a headline was tagged with, so one LLM call
covers every symbol a headline mentions.

**New infrastructure built.** `src/ggTrader/lab/headline_sentiment_data.py` —
Alpaca's `/news` API (already integrated in this project for paper trading;
no new subscription), with manual timestamp-cursor pagination working around
a confirmed-broken `next_page_token` (silently truncates results past the
first page on this account/tier). Sentiment scoring routes through the local
LiteLLM proxy (`deepseek-flash`, temperature 0.0), deduplicated by unique
`news_id` before scoring so a headline tagging five symbols costs one LLM
call, not five. `PUBLISH_LAG_DAYS = 0` by design — a headline's `created_at`
timestamp already is the moment of public availability, unlike the
SEC/FINRA-filing-lag pattern used by other candidates in this project.

**Real bugs found and fixed under load** (all via TDD — regression test
written first, confirmed failing for the right reason, then fixed):
- A `deepseek-flash` reasoning-model token-truncation bug (`max_tokens=5`
  left zero budget for the actual answer after hidden chain-of-thought;
  raised to 300) — caught via smoke-testing before committing to the
  backfill, not a live failure.
- A `None`-response crash: even at `max_tokens=300`, the LLM occasionally
  still exhausts its budget on a particular headline and returns HTTP 200
  with `content: null`. `score_headline`'s try/except only guarded the call
  itself, not a `None` return, so this escaped the fail-safe and crashed the
  pilot backfill mid-run (at 5,600/7,939 headlines). Fixed by treating a
  falsy response as neutral, same as the documented fail-safe philosophy.
- A DST timezone bug: `news_headlines.created_at` rows carry different UTC
  offsets across a DST boundary (the DB session timezone is
  America/Los_Angeles), and `pd.to_datetime()` rejects a Series of
  mixed-offset `datetime.datetime` objects unless `utc=True` is passed
  explicitly. Crashed the WFO immediately on a 2024-08→now eval window
  (which spans two DST transitions). Fixed in both `load_news` and
  `load_sentiment_scores`.

**Scope of this test — a bounded pilot, not the full register.** Per an
explicit scoping decision, this is a **50-symbol, 2-year pilot** (midcap400
universe, alphabetically first 50 tickers, 2024-07-21 to 2026-07-21), not the
full ~400-symbol midcap400 universe or a multi-year history. 7,975 unique
headlines were fetched and scored (661 bearish, 5,508 neutral, 1,806
bullish — real variance, not a degenerate all-neutral failure mode). The WFO
below only has sentiment coverage for these 50 symbols; the other ~350
midcap400 symbols are excluded from every fold by the strategy's own design
(`select()` excludes, rather than zero-fills, symbols with no sentiment data
in the lookback window).

**Result: clean NO-GO.** The strategy fails walk-forward validation badly —
not a marginal miss, but a near-total loss of the in-sample edge out of
sample.

## 2. Quantitative Performance Context

WFO (midcap400 universe, `--wfo`, 3 folds, rolling 12mo train / 3mo test,
2024-08-01 through 2026-07-21, sweeping `lookback_days` ∈ {10, 20, 40} ×
`quintile` ∈ {4, 5}, 6 combos):

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate |
|---|---|---|---|---|
| **headline_sentiment (pilot, 50 symbols)** | **0.25** | **4.8%** | **-18.7%** | 2/3 folds |
| SPY baseline | 1.14 | 22.9% | -9.1% | N/A |

Aggregate WFE (walk-forward efficiency — OOS Sharpe / train Sharpe) is
**0.02**, far below the 0.50 floor: the strategy's train-set Sharpe of
~0.80-1.01 essentially evaporates out of sample, the classic overfitting/
no-real-signal signature (distinct from the FX-hedge/Treasury-curve
"dynamic loses to static" pattern — this is closer to the exit-rule-sweep
and PEAD failure shape, where in-sample optimization found noise, not
edge).

Per-fold detail:

| Fold | Train Window | Test Window | Winner | Train Sharpe | OOS Sharpe | Gate | WFE |
|---|---|---|---|---|---|---|---|
| 1 | 2024-08 → 2025-08 | 2025-08 → 2025-11 | lookback20/quintile4 | 0.80 | 0.27 | PASS | 0.51 |
| 2 | 2024-11 → 2025-11 | 2025-11 → 2026-02 | lookback20/quintile4 | 0.80 | 1.44 | **FAIL** (NDH var 0.53 vs 0.20 cap) | n/a |
| 3 | 2025-02 → 2026-02 | 2026-02 → 2026-05 | lookback40/quintile5 | — | -0.38 | PASS | -0.48 |

Fold 2's headline Sharpe (1.44) looks like a win in isolation, but it
**fails the NDH (non-degenerate-hyperparameter) gate** — the selected
combo's variance across the parameter grid (0.53) blows through the 0.20
cap, meaning the "winner" isn't a stable signal, it's the parameter sweep
finding one lucky combo. Fold 3 goes outright negative (WFE -0.48, same
sign-flip pattern seen in the exit-rule-sweep NO-GO). Only fold 1 is a
clean pass, and even there OOS Sharpe (0.27) is a fraction of train Sharpe
(0.80).

## 3. Actionable Research Directions

None ranked — this is a closure report. The data infrastructure
(`headline_sentiment_data.py`, the Alpaca `/news` pagination fix, the
LiteLLM sentiment-scoring pipeline, `news_headlines` /
`headline_sentiment_scores` schema) is reusable for any future headline- or
LLM-classification-based candidate, independent of this rejection.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Headline/LLM sentiment (candidate A8, 50-symbol/2-year midcap400
pilot) — REJECTED.** OOS Sharpe 0.25 vs SPY 1.14, CAGR 4.8% vs 22.9%,
aggregate WFE 0.02 (target ≥ 0.50) — the in-sample edge does not survive
out-of-sample testing. Fold 2's apparently-strong OOS Sharpe (1.44) fails
the NDH gate on parameter instability, and fold 3 goes negative. This is a
"the signal doesn't generalize" rejection, not an infrastructure or
data-quality failure — the sentiment pipeline itself works correctly
(verified score distribution has real variance: 8.3% bearish / 69.1%
neutral / 22.6% bullish, not degenerate).

**Known scope limitation, for the record (do not treat as a reason to
re-attempt without addressing it):** this pilot only covers 50 of ~400
midcap400 symbols and 2 years of history, per an explicit small-pilot
scoping decision made before the backfill. A full-universe, multi-year
test was never run. If this candidate is ever revisited, that gap — not
the mechanism — is the first thing to close, since a 50-symbol subset with
only 3 WFO folds is a genuinely thin sample to draw a structural conclusion
from. This report treats the result as sufficient to close the *pilot*, not
as definitive proof the mechanism can never work at full scale.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `headline_sentiment` in any configuration.** No live-config
changes. This closes out the cross-asset register's home-lab priority
queue (A1/A7/A3/A5/A8 — five candidates tested, five rejected). Per the
register's priority ordering, next up is **B1 (stablecoin stress signal)**
— note its recommended use is as a **risk-overlay/leverage-reduction
trigger**, not a standalone alpha sleeve, a different shape of candidate
than everything tested so far in this arc.

**Infrastructure kept regardless of this NO-GO:**
`headline_sentiment_data.py` (Alpaca news pagination fix, LiteLLM sentiment
pipeline, point-in-time-correct schema) remains reusable for any future
headline-based or LLM-classification candidate.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** is this a genuine falsification of the
Lopez-Lira & Tang mechanism, or an artifact of the deliberately narrow
pilot scope (50 symbols, 2 years, 3 WFO folds) — small enough that a real
but modest signal could be lost in noise, especially given the paper's own
author-acknowledged decay warning (strategy returns declining as LLM use
becomes more widespread, i.e. the edge may already be smaller today than
when the original study window was measured)?

**Resolution:** Plausible, and worth being honest about — 3 folds is a
thin sample, and this pilot never tested the full ~400-symbol midcap400
universe the strategy was designed for. But two things argue against
re-running at full scale: (1) the failure isn't a narrow miss, it's WFE
0.02 with one fold outright gate-failing on parameter instability — a
signal with real, generalizable structure over even a 50-symbol subsample
should show *some* stability across folds, and it doesn't; (2) the
paper's own crowding/decay caveat cuts against, not for, spending more
backfill time and LLM cost scaling this up — if the effect was real but
weakening as of the paper's 2025 revision, a 2026 backfill is testing an
even-further-decayed version of it. Not worth extending to full scale
without a specific reason to believe the 50-symbol subsample was
unrepresentative (e.g., a sector or size-tilt bias in the alphabetically-
first-50 selection) — which was not checked here and would be the one
cheap thing worth verifying before fully closing the book on this
candidate, rather than a full-universe re-backfill.

### Parked Direction: Full-universe headline sentiment retest

If revisited, the explicit gate to open this: a quick, cheap check of
whether the alphabetically-first-50 midcap400 subsample is meaningfully
different (sector concentration, average market cap, headline volume per
symbol) from the full universe — before spending the LLM-call budget and
wall-clock time on a full ~400-symbol, multi-year backfill. Absent that
check turning up a real selection bias, this candidate stays closed.
