# Live Indicator Warm-Up: 120-Day Window Drifts from the Backtest; 365 Days Removes It

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-28
**Audience:** ggTrader owner & research collaborators

## 1. Executive Summary & Core Engine Audit

- **The problem.** `paper/signal_runner.generate_signals` loads `lookback_days=120` calendar
  days (~83 bars) and computes the 5-voter ensemble on just that window. EMA(20/50),
  MACD(12/26/9) and Wilder RSI(14) are recursive (`ewm(adjust=False)`), so each is seeded
  at the window's first bar. The backtest seeds them years earlier, so the two can
  disagree about whether a crossover happened.
- **The measurement.** Over the last 125 sessions (2026-03-30 → 2026-09-25), across the
  point-in-time S&P 500, the ensemble was recomputed as live does for 6 candidate windows
  and compared with a reference computed on history back to 2016.

**Verdict:**

- **The 120-day window mismatches 4.0% of entries and 3.4% of exits.** That is 21 missed
  and 21 extra entries out of 1,026, and 106 missed and 97 extra exits out of 5,881.
- **It corrupts the missed-exit catch-up input.** The most-recent exit date the trader
  uses (`last_exit`) is wrong for **7.0%** of symbol-days.
- **The EMA-50 cross is almost the whole cause.** On its own, that voter disagrees on 408
  of 640 entry events at 120 days.
- **A 365-day window (~252 bars) gives zero mismatches** on every measure. 250 days still
  misses 4 exits and 13 EMA-voter entries; 365 is the first value that is exact.
- **The extra cost is negligible.** The DB read goes from 10.1 s to 10.5 s per run, plus a
  one-time yfinance backfill of 2 young listings. But see §4.B: a separate cache bug
  makes that "one-time" backfill repeat on every run.

**The AVB anecdote is not a warm-up effect, and it points to a worse problem (§4.A).**
- On 2026-08-14 both the 120-day window and full history fire AVB's RSI exit (RSI
  16.4 → 76.7). It fires because that day's bar is the only true AVB price in a
  corrupted stretch of tape.
- From 2026-07-20 the `ohlcv` "AVB" series tracks EQR's price level (AVB merged into
  EQR 1:2.793 on 2026-08-17). That shows up as a phantom −64% crash.
- The live account bought AVB on 2026-08-10 ($745) and **still holds 4.05 merged AVB
  shares**, marked at a frozen $184.06.

## 2. Quantitative Performance Context

- **Raw results:** `docs/research/_live_indicator_warmup_results.json`.
- **Driver:** `scripts/live_indicator_warmup.py`.
- **Ensemble:** `EnsembleSignal` defaults, the same object live builds: 5 voters,
  `min_agree` 2, RSI exit independent.
- **Eligibility:** live's floor of ≥60 bars in the window.
- **Universe:** S&P 500 members as of each day, 507 symbols loaded.
- **Mismatch** = (missed + extra) / union of events, counted on day d only.

| `lookback_days` | Median bars | Entry mismatch | Exit mismatch | `last_exit` date wrong |
|---|---|---|---|---|
| **120 (live today)** | 83 | **4.0%** (21 missed / 21 extra of 1,026) | **3.4%** (106 / 97 of 5,881) | **7.0%** (4,392 / 62,773) |
| 180 | 124 | 0.2% (0 / 2) | 0.6% (17 / 19) | 0.8% |
| 250 | 172 | 0.0% | 0.1% (4 / 1) | 0.1% |
| **365** | **252** | **0.0%** | **0.0%** | **0.0%** |
| 500 | 343 | 0.0% | 0.0% | 0.0% |
| 730 | 502 | 0.0% | 0.0% | 0.0% |

**Per voter at 120 days (missed / extra / reference events):**

| Voter | Entries | Exits | Why |
|---|---|---|---|
| EMA 20/50 cross | 408 / 424 / 640 | 373 / 402 / 633 | span-50 seed still has ~4% weight after 83 bars |
| RSI 14 | 30 / 31 / 596 | 111 / 109 / 3,846 | Wilder smoothing, values near 30/50 flip |
| MACD 12/26/9 | 7 / 13 / 3,772 | 67 / 61 / 2,592 | EMA seeds |
| BB 20, volume-BB | 0 / 0 | 0 / 0 | fixed rolling windows, no memory |

The EMA voter's disagreement is large on its own, but it moves the ensemble only when it
is the deciding second vote, which caps the ensemble-level damage at 4%.

**Load, measured read-only against 503 current members (2026-09-28):**

| `lookback_days` | Rows | DB read, median of 3 | Symbols triggering a yfinance full fetch |
|---|---|---|---|
| 120 | 82 | 10.09 s | 0 |
| 250 | 172 | 10.30 s | 1 (FDXF) |
| **365** | **250** | **10.50 s** | **2 (FDXF, Q)** |
| 730 | 499 | 11.02 s | 3 (FDXF, Q, SNDK) |

- **The DB read is dominated by per-query overhead, not rows:** +0.4 s (+4%) at 365 days.
- **In-memory size is trivial:** 503 × 250 × 5 fields.
- **yfinance cost** is the full-range fetch for listings younger than the window.
  Intended once; currently every run (§4.B).

## 3. Actionable Research Directions

### Rank 1: Set `lookback_days=365` in live signal generation (proposed, not made)

```python
# src/ggTrader/paper/signal_runner.py
def generate_signals(universe: str = "sp500", lookback_days: int = 365) -> dict:
    # ~252 bars: EMA-50/MACD/RSI seeds wash out, so day-d signals match the
    # backtest's long-history indicators exactly (0 mismatches over 125 sessions,
    # docs/research/2026-09-28-live-indicator-warmup.md). 120 days (~83 bars)
    # mismatched 4.0% of entries, 3.4% of exits and 7.0% of last_exit dates.
```

- **Why 365 over 250:** 250 still misses exits (4 in 6 months). The cost difference
  between the two is 0.2 s and one extra young symbol.
- **Why not more:** 500 and 730 buy nothing further.
- **Effort:** S. One default, plus a test asserting the default ≥ 365.
- **Deployment:** this is live code, so it ships through `ggtrader-deploy`
  (image rebuild + pull).
- **Expected effect:** live entries and exits match the lab rule. This is unvalidated as
  P&L; it is a parity fix, not an edge.
- **Primary risk:** the first run after deploy recomputes `last_exit` on correct
  indicators. Expect a one-off catch-up sell of any held name whose true last exit
  post-dates its BUY. Run a dry run first and review the sell list.

### Rank 2: Fix the inception-cache read so young listings are fetched once

```python
# src/ggTrader/data/live/cached_yfinance_loader.py, _get_known_inceptions
return {sym: pd.Timestamp(dt).tz_convert("UTC") for sym, dt in rows}
```

`first_date` is `timestamptz`, so psycopg returns tz-aware datetimes and
`pd.Timestamp(dt, tz="UTC")` raises (§4.B). This matters regardless of Rank 1, because the
lab hits it too. Effort: S, plus a test with a tz-aware row.

### Rank 3: Residual parity gap, eligibility floor (not measured here)

Live uses `min_history_bars=60`; the lab default is 400. Young listings (FDXF, Q, SNDK) are
eligible live but excluded in the backtest. This is not fixable by the window alone, since
365 days is only ~252 bars. Decide whether live should require ≥252 bars, or the lab
should drop to 60.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. AVB 08-14 exit as a warm-up symptom — CORRECTED: it's a data-integrity failure.**

| Date (ohlcv) | "AVB" close | EQR close | Real AVB |
|---|---|---|---|
| 2026-07-17 | 192.53 | 69.00 | ~$192 |
| **2026-07-20** | **68.69** | 68.80 | ~$184–192 (merger arb) |
| 2026-08-13 | 65.85 | 65.97 | |
| **2026-08-14** | **184.06** | 65.97 | $184.06 (Alpaca mark) |
| 2026-08-17 | 63.66 | 63.66 | merged into EQR 1:2.793 |
| after 2026-08-24 | no bars | | |

- **Source of the corruption:** it starts on 07-20, the same session the benchmark tape
  stalled (`signal_runner.BENCHMARK_SYMBOLS` comment). The mechanism was not traced.
- **Price-level check:** the pre-07-20 AVB/EQR price ratio was ~2.78–2.98, and 0.357 ≈
  1/2.793, the merger ratio. So after 07-20 the source is serving merger-adjusted or EQR
  prices under the AVB ticker.
- **Trades it probably caused (not verified):**
  - The 08-10 live BUY: the phantom −64% day puts BB/RSI deep in "oversold".
  - The 08-14 RSI "exit".
- **Current state:**
  - Alpaca still reports the AVB position: 4.050706521 sh, cost $745.33, current and
    last-day price both $184.06.
  - The constituent file still lists AVB as an S&P 500 member on 2026-09-28.
- **Other level breaks (daily ratio > 1.6 or < 0.6, 06-01 → 09-28), each needing
  verification:**
  - APH 0.497 and MNST 0.489 on 07-20. These look like 2:1 splits whose earlier cached
    history was not back-adjusted.
  - MRNA ×2.77 on 08-19.
  - MLI 0.505 on 06-25.
  - STI and ADCT on 06-04.

**B. Inception cache — CORRECTED: written, never read.**
- `symbol_inception` holds 88 rows, re-recorded as recently as 2026-09-29 03:42 UTC.
- Every read logs `Inception lookup failed (Cannot pass a datetime or Timestamp with
  tzinfo with the tz parameter...)` and returns `{}`.
- So any symbol whose cache starts after the requested start is full-fetched from
  yfinance **on every call**, in live and in the lab.

## 5. Operational Roadmap: Recommended First Action

**Owner decision on AVB comes first: the live book holds a merged, unpriced AVB position
that the trader cannot exit.** It has no tape since 08-24 and computes no exit.
Options are to confirm with Alpaca whether the paper account will convert it to EQR, or to
close it manually. This is an order against the live account, so it is not done here.

Then ship Rank 1 (365 days) and Rank 2 (inception tz) together through `ggtrader-deploy`,
with a dry run first to review any catch-up sells. Then audit the §4.A level breaks
before the next lab run cites any number touching those names.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: 4% is small, so does this matter?**

- **Per signal, it is modest.** But the catch-up input is wrong 7% of the time, and that
  path exists precisely to make live track the backtest (the 2026-09-28 exit-parity fix).
- **The fix is one default, at +0.4 s per run.** There is no argument for keeping 120.

**What this study does not show:**

- The P&L impact of the mismatches. That needs a live-vs-backtest trade reconciliation.
- Whether the measured windows hold for other periods. The span is only 125 sessions.
- The effect of the 12:45 PT partial bar. Both sides were compared on completed bars.

### Parked Direction: Price-bar sanity filter

The AVB, APH, MNST and MRNA breaks all pass silently into signals. A per-symbol
daily-ratio guard is the natural fix. Its prerequisite is the §4.A audit, so the guard
isn't tuned on corrupted data. **Gate:** audit done.
