# Dynamic FX Hedge Overlay (Candidate A1): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-20
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/fx_hedge_overlay.py`
(`FxHedgeOverlayStrategy`): candidate A1 from `WEB_RESEARCH_CANDIDATES.md`'s
2026-07-19 cross-asset register — the top-ranked pick for this project's
home-lab/ETF workflow after an independent review demoted the previous #1
pick (direct CIP basis harvesting) as not implementable with retail ETFs.
Source: Castro, Hamill, Harber, Harvey & Van Hemert, *"The Best Strategies
for FX Hedging,"* *Journal of Portfolio Management* 51(9) (2025) — vary the
hedge ratio on international equity exposure using carry, PPP-value, and
trend signals instead of a static hedged/unhedged policy.

**Retail-proxy implementation:** two currently-active hedged/unhedged ETF
pairs — EWJ/DXJ (Japan, JPY, since 2006) and EZU/HEZU (Eurozone, EUR, since
2014). Several other single-country hedged ETFs (HEWG, HEWU) were checked
and found delisted 2023-2024, deliberately excluded. Per pair, monthly
rebalance allocates between the unhedged and hedged share class based on a
combined score: carry (foreign short rate − US short rate, FRED
`TB3MS`/`IRSTCI01{cc}M156N`), PPP-value (real exchange rate z-score vs its
own trailing history, from FRED spot FX + CPI series), and trend (12-1
month momentum of the unhedged/hedged price ratio, which isolates the
currency return net of shared equity beta).

**New infrastructure built:** `src/ggTrader/lab/fred_data.py` — FRED's
public `fredgraph.csv` endpoint (free, no API key, distinct from FRED's
key-gated JSON API). Point-in-time `available_as_of` with a conservative
45-day publication lag for monthly macro series. Found and fixed a real
bug: `DataFrame.iterrows()` over a frame mixing a `datetime64` date column
and a `float64` value column silently coerced one row's value to `NaT` on
a ~950-row series (reproduced and regression-tested; fixed via
`to_dict("records")`). Also found and fixed a data-selection mistake of my
own: the originally-chosen Eurozone CPI series ID (`CPALTT01EZM659N`) does
not exist on FRED — an earlier `curl | wc -l` check miscounted an HTML 404
error page's line count as if it were 545 real CSV rows, a same-shaped
methodology error to the "grep false positive" lesson from earlier in this
project's research log. Replaced with the verified `CP0000EZ19M086NEST`
(Eurozone HICP). New `fx_hedge` universe (static ETF-ticker snapshot,
registered alongside sp500/midcap400/nasdaq100/russell2000).

**Result: clean NO-GO, and a clear falsification of the specific claim
being tested.** SPY is not the right benchmark for this candidate — the
paper's actual claim is that *dynamic* hedging beats a *static* hedge
policy on the same instruments, not that FX hedging beats the US equity
market. The decisive test (`scripts/fx_hedge_overlay_static_baseline_check.py`)
compares the dynamic strategy's stitched OOS equity curve against three
static baselines (100% unhedged, 100% hedged, static 50/50) on the exact
same instruments over the exact same OOS window (2017-01-02 to
2026-06-30). **The dynamic strategy underperforms every static baseline
tested, including the naive 50/50 split.**

## 2. Quantitative Performance Context

**Standalone WFO** (SP500-style config, `--wfo`, 38 folds, 2016-01 through
2026-07, sweeping `k` ∈ {1.5, 2.25, 3.0} × `trend_lookback` ∈ {126, 252}):

| System Configuration | OOS Sharpe | CAGR | Max Drawdown |
|---|---|---|---|
| fx_hedge_overlay (dynamic) | 0.41 | 6.0% | -36.7% |
| SPY (reference only — not the real benchmark for this candidate) | 0.74 | 15.2% | -33.7% |

Aggregate WFE 1.10 (comfortably above the 0.50 floor — not an overfitting
signature; the strategy genuinely trades the way the sweep intended, it's
just not a good trade).

**Decisive test: dynamic vs static hedge policy**, same instruments, same
OOS window (2017-01-02 to 2026-06-30):

| Configuration | Sharpe | CAGR | Max Drawdown |
|---|---|---|---|
| **dynamic (carry+value+trend)** | **0.41** | **6.0%** | **-36.7%** |
| static 100% unhedged | 0.53 | 9.7% | -35.2% |
| static 50/50 | 0.66 | 12.4% | -34.8% |
| static 100% hedged | 0.73 | 14.7% | -35.9% |
| SPY (reference only) | 0.74 | 15.2% | -33.7% |

The dynamic strategy is worst on every metric among all four FX
configurations tested. Correlation between the dynamic strategy's daily
returns and the static 50/50 baseline's is 0.835 — the dynamic signal
isn't meaningfully differentiating itself from a naive fixed split, and
what differentiation it does add subtracts value rather than improving
it. Notably, **simply staying 100% currency-hedged was the best of the
four FX configurations** (Sharpe 0.73, essentially matching SPY) — over
this specific window (2017-2026, a broadly dollar-strong decade against
both JPY and EUR), removing currency risk entirely paid off, and the
dynamic overlay's occasional tilts toward being unhedged detracted from
that.

## 3. Actionable Research Directions

None ranked — this is a closure report. The underlying data
infrastructure (`fred_data.py`, the `fx_hedge` universe, both hedged/
unhedged ETF pairs backfilled) is reusable for any future FRED-based
candidate (e.g. A5's Treasury term-structure factors, A7's pre-FOMC drift)
independent of this rejection.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Dynamic FX hedge overlay (carry+value+trend on hedged/unhedged ETF
pairs) — REJECTED.** The specific claim under test — dynamic hedging beats
static — is falsified for this retail-proxy implementation: OOS Sharpe
0.41 vs static 50/50's 0.66 and static 100%-hedged's 0.73, worse Sharpe
and worse-or-comparable drawdown than every static alternative. This is
not an eval-window-drift or overfitting rejection (WFE 1.10 is healthy) —
it's a clean "the signal construction doesn't add value over doing
nothing clever" result, closer to the `short_volume_ratio`/
`short_interest` failure signature than the PEAD/insider-cluster/
congress-trades pattern.

**Known implementation limitations, for the record (do not treat as
reasons to re-attempt without addressing them):**
- Only 2 currency pairs (JPY, EUR) — most single-country hedged ETFs
  outside these were checked and found delisted (HEWG, HEWU), so this
  implementation cannot easily be broadened to more currencies with
  currently-tradable retail instruments.
- ETF-switching is an approximation of the paper's forward-based hedge
  ratio construction, not a replication, per the register's own caveat —
  transaction costs of repeatedly shifting weight between two ETFs
  tracking the same underlying index were modeled via this project's
  standard slippage/fee assumptions, not the paper's actual forward-
  market cost structure.
- Signal combination (tanh-normalized carry/trend, z-scored value) is a
  reasonable but not paper-derived construction; the paper's own hedge-
  ratio formula was not directly implemented.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `fx_hedge_overlay` in any configuration.** No
live-config changes. Move to the next candidate in the register's
home-lab priority order: **A7 (pre-FOMC long-Treasury drift)** — promoted
out of parked status after its citation was replaced with a verified,
current (June 2026) paper directly on this exact rule (Pan & Peng), free
data (TLT/IEF/EDV + FOMC calendar), simple event-driven mechanism.

**Infrastructure kept regardless of this NO-GO:** `fred_data.py` (FRED CSV
loader, DB-cached, point-in-time correct) and the `fx_hedge` universe
snapshot remain reusable for A5 (Treasury term-structure, needs FRED yield
data) and A7 (needs only OHLCV + a FOMC calendar, not FRED, but the module
pattern still applies to any future macro-series candidate).

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** is the near-uniform "static hedged wins" result
period-specific (this exact 2017-2026 dollar-strong decade) rather than a
structural refutation of the dynamic-hedging thesis — i.e., would the
paper's own longer sample (back to 1973 per its abstract) show dynamic
hedging adding value in periods this test window doesn't cover?

**Resolution:** Plausible, and worth flagging honestly rather than
over-claiming a structural refutation — this test only covers ~9.5 years
(2017-2026) against instruments with meaningfully shorter live history
than the paper's multi-decade academic sample. But the practical
constraint is real regardless: the retail-accessible hedged/unhedged ETF
pairs used here only have 12-20 years of live trading history each (and
several close analogues were delisted mid-sample), so this project cannot
extend the test window further without a data source this project doesn't
have. Not worth pursuing further at the current data-access tier; would
need either (a) a genuinely longer live-instrument history to accumulate,
or (b) synthetic/backtested currency-hedge construction from raw FX
forwards data, which reopens the same institutional-access gap already
flagged for candidate A9 (direct CIP harvesting).
