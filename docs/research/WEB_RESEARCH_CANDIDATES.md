# Web-Sourced Strategy Candidates (Backlog)

This file **accumulates** — unlike `RESEARCH_SNAPSHOT.md` and the prompt
files (which are fully regenerated/overwritten each run), this is a running
backlog of externally-sourced ideas that haven't been triaged or tested yet.
Nothing here gets deleted on a "regenerate" pass; entries only leave this
file when they're resolved (see Status below).

## How this file gets updated

1. Run `docs/research/prompts/web-strategy-research-prompt.md` in a
   web-research tool (Google Gemini or Claude's web-UI research feature).
2. Paste the results back into this session.
3. Ask to "merge these into the web research candidates" — I'll dedupe
   against existing entries (by mechanism, not just name) and append
   genuinely new ones below, each with a status.
4. When a candidate is picked to actually build, copy its entry into
   `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`'s
   "Candidate Strategy" section, queue a concrete step in
   `docs/next_steps.md`, and update this entry's status to `queued`.
5. Once implemented and WFO-tested (GO or NO-GO), mark the entry `resolved`
   with a link to its `docs/research/<date>-<slug>.md` report — the next
   `research-snapshot` skill run will pick it up into `RESEARCH_SNAPSHOT.md`'s
   roster, and it can be removed from this backlog at that point.

## Status legend

`untriaged` (found, not yet evaluated) · `queued` (picked, implementation
scheduled) · `testing` (WFO run in progress) · `resolved` (GO/NO-GO landed,
see linked report — ready to remove from this backlog)

---

## 2026-07-16 batch — Novel Strategy Candidates (non-technical signal categories)

Source: web-research pass scoped to signals/market-structures genuinely
different from price-action/technical timing of large-cap U.S. equities
(the technical-signal category was already exhausted — see project memory
`project_edge_search_2026-06-08`). Full report: 14 candidates, ranked
roughly by promise × retail feasibility. Went through two verification
rounds (an initial primary-source check, then an independent LLM fact-check
that was itself spot-checked against primary sources). Net effect of the
verification chain: two stats in the original submission were corrected
(a momentum-alpha figure and a crypto-carry yield figure, both inflated),
one paper's scope was narrowed (Asquith-Pathak-Ritter does not directly
study borrow fees/utilization), and one candidate (#6) was downgraded after
pulling its actual results table showed a statistically-significant but
economically-zero effect (mean 0.00%). One candidate (#10) has **no located
source at all** and is a placeholder only. Take verification status per
entry as current best-effort, not a guarantee — the report itself flags
that fact-check chains can compound errors as easily as catch them.

**Status: `untriaged` for all 14.** Recommended triage order per the
report's own priority tiers (see table below): first wave #1, #2, #7 (and
the free-data-only version of #3); second wave #4, #5, #11; third wave #8,
#9, #12, #13, #14; do-not-build-yet #10 (no source located).

**Cross-reference note:** 3 of these 14 overlap in mechanism with
`RESEARCH_SNAPSHOT.md` §6's internally-derived candidates — #4 (crypto
carry) ↔ internal Rank 4, #12 (PEAD) ↔ internal Rank 2, #14 (options IV
skew) ↔ internal Rank 3. Each overlapping entry below is annotated inline;
treat this file's version as the more detailed one (real citations,
verified/corrected effect sizes, decay warnings) when starting work — don't
restart from the thinner internal write-up. §6's Rank 1 (market-neutral
pairs/stat-arb) has no counterpart in this batch and is already
mid-investigation outside either list (untracked `pairs_stat_arb.py` +
`pairs_stat_arb_research.py`, per project memory: signals long-only, shorts
unverified, pairwise-correlation missing).

### 1. Insider cluster-buying signal (SEC Form 4) — RESOLVED, NO-GO (2026-07-19)
`docs/research/2026-07-19-insider-cluster-buy-nogo.md`. The highest-effort
candidate built this session — full SEC EDGAR Form 4 pipeline
(`form4_data.py`), 833,158 transaction rows across 764 SP500 symbols since
2015, ~24-hour rate-limited backfill. Initial long-window SP500 test
looked distinctive (Sharpe 0.76 near-tied SPY, shallowest drawdown and
lowest core-correlation of any candidate this session at 0.382) — but
following the process lesson from PEAD's closure, a matched-window retest
against the deployed blend's own 2021-2026 validation window showed real
underperformance (Sharpe 0.39 vs SPY 0.58), and a 4-sleeve blend test
confirmed no improvement (Sharpe 1.14→1.12, MaxDD -5.39%→-5.45%). Third
diversification-sleeve candidate rejected this way this session (after
`idio_vol` and `pead`) — moderate-to-low core-correlation alone hasn't
been sufficient in any case tested so far. `form4_data.py` and the
833,158-row backfill remain reusable for any future insider-transaction
work (e.g. parked candidate #6).

3+ distinct insiders (officers/directors/10%+ holders) making open-market
purchases (code "P") within ~2 weeks, excluding 10b5-1/option-exercise/grant
transactions; long the basket, 6–12mo hold. Key modern reference: Cohen,
Malloy & Pomorski, "Decoding Inside Information" (JF 2012) — split
"routine" vs. "opportunistic" trades, alpha concentrated in opportunistic.
Data: free (EDGAR Form 4 XML since ~2003; OpenInsider/secform4.com for
prototyping). Differs from prior work: corporate-insider-intent signal, not
price/volume — independent sleeve for the existing ensemble framework.

### 2. Analyst estimate-revision momentum (SUE / earnings-revision breadth)
Rank by direction/breadth of trailing 1–3mo forward-EPS revisions; long top
decile, short/avoid bottom decile, monthly rebalance. Sources: Givoly &
Lakonishok (1979), Stickel (1991), Guerard's CTEF/BARRA earnings-momentum
work. Data: free-tier feasible for a forward-testing version (Zacks Rank,
Seeking Alpha, Financial Modeling Prep forecast endpoints); deep historical
backtest would want paid I/B/E/S-style consensus data. Differs from prior
work: fundamentals-expectations signal, orthogonal to the
Bollinger/RSI/EMA/MACD ensemble and to the already-rejected price-momentum
test.

### 3. Hard-to-borrow / short-interest signal — RESOLVED, NO-GO (2026-07-17, free-data cut)
`docs/research/2026-07-17-short-interest-nogo.md`. Built as `short_interest`
strategy on a new FINRA consolidated-short-interest data pipeline
(`src/ggTrader/lab/short_interest_data.py`, 86,793-row backfill, 150
settlement dates 2020-04 through present). WFO (SP500, 20 folds): OOS
Sharpe 0.27 vs SPY 0.61, WFE 0.11 (below the 0.50 floor), regime-halt 16/20
folds — a noise/overfitting rejection, consistent with the
value-weighted-insignificance caveat already flagged below. The paid
cost-to-borrow/utilization version remains untested (feasibility risk, not
attempted). Notable reusable infrastructure: `discover_settlement_dates()`
corrects a real bug where guessing FINRA's settlement-date calendar (15th/
month-end) silently missed 47% of real cycles (dates shift to the nearest
business day around weekends/holidays) — use it for any future FINRA
short-interest work rather than re-deriving dates from a calendar rule.

Underweight/avoid/short names transitioning from easy- to hard-to-borrow
with rising fees. Verdad Capital "Costly Shorts" (2024) is the primary
source; Asquith-Pathak-Ritter (JFE 2005) is real but **narrower than often
summarized** — uses short interest + institutional ownership, not
fees/utilization directly, and its headline underperformance (215bps/mo) is
equal-weighted only — the value-weighted result (39bps/mo) is
insignificant, meaning the effect concentrates in small/illiquid names, a
real caveat for a large/mid-cap application. Data: FINRA short-interest is
free but bi-monthly/low-frequency; real-time borrow-fee/utilization data
(Ortex, IHS Markit) is paid — flag as feasibility risk before committing
engineering time; a free-data-only (bi-monthly short interest) version is
the first-wave-feasible fallback.

### 4. Crypto perpetual-futures funding-rate carry (delta-neutral) — INFEASIBLE for honest WFO (checked 2026-07-20)
Checked before building anything. Two corrections to the original framing:
(1) Binance.US does not offer perpetual futures at all (`ccxt.binanceus()`
reports `has['future']=False`, `has['swap']=False` — spot only, consistent
with US regulatory restrictions on retail crypto derivatives), so "Kraken
Futures/Binance US" was never really a two-venue choice — Kraken Futures is
the only option. (2) Kraken's historical-funding-rates data — verified via
both `ccxt`'s `fetchFundingRateHistory` (pagination stalls, no further
progress) and Kraken's own native API
(`futures.kraken.com/derivatives/api/v3/historical-funding-rates`
directly) — only retains a **rolling ~1 year** of hourly records (earliest
available: 2025-07-13, as of this check on 2026-07-20; 8,912 hourly rows ≈
372 days). This project's WFO methodology needs 12-month train + 3-month
test *per fold*, with many folds for a credible result (every other
candidate this session got double-digit folds) — 1 year of total history
supports essentially zero valid folds. Free third-party aggregators
(Coinalyze, CF Benchmarks' KFRI) don't offer bulk historical download;
paid ones (CoinAPI, back to ~2021) would still fall short of this
project's usual 2015-2020-era eval-window convention and require a paid
subscription regardless. Deprioritized pending either a much longer
free-data source or a paid-data decision — same class of blocker as
candidates #2 and #7 (infeasible for the required methodology, not an
engineering-effort problem).

*Overlaps with `RESEARCH_SNAPSHOT.md` §6 internal Rank 4 (revisit parked
crypto-carry) — this entry is the more detailed, citation-backed version;
check here before restarting that internal item.*

Spot long + perp short (or inverse) to collect funding while staying
price-neutral; enter on persistently elevated funding (multi-period, not a
single reading) on Kraken Futures/Binance US (already-integrated venues).
**Correction applied:** the academic reference (Schmeling, Schrimpf &
Todorov, "Crypto Carry", BIS WP 1087/Management Science) documents carry
reaching 40–60% annualized and highly volatile — not a stable "8–10% mean"
as originally submitted. A 2025 follow-up (arXiv 2510.14435) found the
strategy's annualized Sharpe fell from 6.45 (2020–2025 full sample) to 4.06
from 2024, **turning negative in 2025** — the easy/crowded version of this
trade is being arbitraged away; test with the decay in mind, don't assume
the historical average carry holds. Data: free/cheap via exchange APIs
already integrated. Differs from prior work: market-neutral carry, not
directional timing — first candidate with no price-direction bet at all.

### 5. Informed liquidity-supplying short sellers ("stealthy shorts") — RESOLVED, NO-GO (2026-07-19)
`docs/research/2026-07-19-short-volume-ratio-nogo.md`. Built as
`short_volume_ratio`, the free-data-only cut (plain trailing short-volume
ratio, no liquidity-demand/supply decomposition — that needs
transaction-level tick data this project doesn't have). New FINRA daily
short-volume pipeline (`short_volume_data.py`, different dataset than
short_interest_data.py's bi-monthly one; verified CDN retention starts
2018-08-01). 1,189,652 rows backfilled. WFO (SP500, 27 folds): OOS Sharpe
0.21 vs SPY 0.72, gate pass 16/27 (59%), regime halt 20/27 (74%,
persistent) — a market-neutral book with shallow drawdown but near-zero
return, consistent with the construction working mechanically but not
capturing a differentiating signal at this fidelity. No matched-window
follow-up needed (never showed standalone promise). `short_volume_data.py`
and the backfill remain reusable for future daily-short-volume research.

Decompose daily short-sale volume into liquidity-demanding vs.
liquidity-supplying components (standard trade-classification on FINRA
short-sale volume); short highest-quintile passive-short-volume names, long
lowest quintile, ~1mo hold. Source: Goyal, Reed, Smajlbegovic & Soebhag,
"Stealthy Shorts: Informed Liquidity Supply" (JFE Vol. 172, Oct 2025) —
verified real/recent; central finding (informed shorts are liquidity
*suppliers*, not demanders) confirmed against the publisher abstract, but
the specific "38bps over 21 days" figure from the original submission
**could not be independently verified** — treat as unconfirmed. Data:
FINRA daily aggregate short volume is free; the paper's clean
liquidity-demand/supply decomposition used institutional-grade
transaction-level data (Cboe/Nasdaq/NYSE) a retail setup won't have — a
noisier proxy (short volume vs. intraday range/VWAP deviation) is buildable
on free data but flag the fidelity gap. Differs from #3: stock-loan market
data (cost/scarcity to borrow) vs. this one's order-flow execution style
within the short-sale itself.

### 6. Form 144 vs. Form 4 "reporting inversion" (insider non-execution) — low priority
Track Form 144 sale notices (90-day execution window) against matching
Form 4 executions; when a 144 expires unexecuted, treat as a bullish
"overhang removed" signal. Source: Neupane (arXiv 2602.17890, Feb 2026) —
real, currently-posted, but a **single-author, non-peer-reviewed preprint
with no independent replication**. **Downgraded on direct verification of
the paper's own results table:** the reported post-non-execution price
reaction is statistically significant but at mean magnitude **0.00%** —
real in a statistical sense, economically worthless, almost certainly
smaller than realistic trading costs. The mechanism/rationale is sound but
currently unsupported by a tradeable effect size. Data: free (EDGAR Form
144 + Form 4). Worth revisiting only if a larger/different-sample
replication shows an economically meaningful effect — not worth research
time as currently evidenced.

### 7. Anomaly-Driven Demand (ADD) — factor-crowding rebalancing pressure — INFEASIBLE (checked 2026-07-18)
Verified against the actual Chen & Zimmermann dataset (via the
`openassetpricing` Python package's source, `_dl_signal`/`_dl_individual_signal`
in `openap_download.py`): the firm-level characteristics data is keyed
purely by CRSP `permno` + `yyyymm` — **no ticker column at all**, and the
package's own pipeline calls `wrds.Connection()` directly for some signals
(Price/Size/STreversal). No free permno-to-ticker crosswalk exists;
complete ticker-history identity mapping is part of CRSP's proprietary
NYSE/AMEX/NASDAQ monthly stock file (a WRDS subscription product) —
confirmed via search, no institution-free alternative found. Same class of
blocker as candidate #2 (analyst estimate-revision momentum): the
"free" framing in the original write-up describes the predictor CSVs
themselves, not what's needed to actually join them to a ticker-based
universe. Building a name-matching heuristic crosswalk was considered and
rejected — ticker reuse over the dataset's multi-decade span makes this a
real silent-data-corruption risk, not proportionate to a "moderate effort"
candidate. Deprioritized pending a WRDS-access decision; skip past it in
the effort ordering.

Replicate a broad cross-section of published return anomalies (value,
asset growth, accruals, share issuance, etc.); score each stock monthly by
net change in long-leg vs. short-leg anomaly membership; long biggest
net-increase (anticipated synchronized anomaly-following buy pressure),
short/avoid the mirror case. Source: Posselt & Kjær, "Anomaly-Driven
Demand" (2026) — confirmed real (conference version on conftool.org,
discussed by AlphaArchitect/IBKR Quant in 2026 summaries); builds on Chen &
Zimmermann's "Open Source Cross-Sectional Asset Pricing" (Critical Finance
Review, 2022), a real open anomaly-definitions dataset. Conceptually
similar to the S&P 500 index-reconstitution effect (#13) but generalized
across the whole universe of published factor anomalies rather than one
index — newer, less publicly-documented decay evidence than #13's
well-studied effect. Data: free/cheap (Chen & Zimmermann's open dataset +
standard monthly fundamentals/returns). Differs from prior work: not a bet
on any one anomaly (unlike the already-rejected cross-sectional momentum
test) — a meta-signal about mechanical flows *around* anomalies.

### 8. Retail-attention-conditioned factor anomalies — BUILT, BLOCKED ON LIVE DATA (2026-07-19)
Implemented as `retail_attention` strategy testing Da/Engelberg/Gao's
actual validated core finding (search-volume spikes → short-term buying
pressure) rather than the vaguer, unconfirmed "condition an unspecified
factor" framing below — `src/ggTrader/lab/google_trends_data.py`
(pytrends), full test coverage (19 tests, all passing against injected
fakes — the parsing/strategy logic is proven correct). **Not resolved
yet**: a small feasibility spot-check (39/40 rapid queries succeeded)
under-sampled Google's actual rate limiting — starting the real ~750-symbol
backfill immediately after triggered a 429 lockout that didn't clear on
retry (unlike a simple per-request rate limit, this looks like an IP-level
cooldown of unknown duration). Paused, not closed — retry the backfill
(`scripts/google_trends_backfill.py`) in a later session once the block has
likely cleared, with more conservative pacing than the 2s/request already
used. Do not attempt IP rotation or other circumvention.

Apply standard factor signals (value, momentum, etc.) only to the subset of
stocks with unusually high retail attention (search-volume/clickstream
spikes vs. volume-matched peers), on the theory anomalies are less
efficiently arbitraged where retail behavioral bias dominates price
formation. Underlying attention-proxy concept is solid (Da, Engelberg &
Gao, "In Search of Attention," JF 2011 — validated Google search volume as
a retail-attention proxy), but the original submission's specific
"Alpha Architect/ExtractAlpha" attribution and "9.8%→19.2%" value-spread
figures **could not be independently verified** — treat as unconfirmed.
Data: free (Google Trends); commercial clickstream data would be a
nice-to-have, not required to test the core idea. Differs from prior work:
a conditioning overlay on top of a factor signal, not a standalone
technical rule — independent of the rejected ML-gating experiment (that
gate used price/volume features; this one is an alternative-data signal
about who's trading).

### 9. Congressional (STOCK Act) trade disclosures — RESOLVED, NO-GO (2026-07-19)
`docs/research/2026-07-19-congress-trades-nogo.md`. Built House-only
(Senate's efdsearch.senate.gov needs a stateful CSRF session, not
attempted) — `house_ptr_data.py`, 41,443 rows across 7,578 PTR filings
2015-2026. **This was the strongest long-window standalone result of the
entire session** (OOS Sharpe 0.89 vs SPY 0.77, gate pass 34/40 = 85%,
stability 13/40 = 33%, all highest seen) — but per the discipline
established by `pead` and confirmed by `insider_cluster_buy`, a
matched-window retest and blend test both overturned it: standalone
Sharpe fell to 0.36 vs SPY 0.58, and the 4-sleeve blend showed the worst
degradation of any diversification candidate this session (Sharpe
1.14→1.04). Third consecutive candidate to fail this exact way — the
eval-window-drift pattern is now a standing expectation, not a caveat.
`house_ptr_data.py` and the backfill remain reusable for any future
congressional-disclosure research (including a Senate extension).

Track members-of-Congress stock trade disclosures (45-day disclosure
window under the 2012 STOCK Act) and mirror buy-side trades, optionally
weighted toward committee assignments relevant to the traded sector.
**Post-Act evidence is genuinely mixed**, weaker than a "blanket mirror
all trades" framing suggests: pre-Act studies (Ziobrowski et al. 2004/2011)
found large outperformance, but Eggers & Hainmueller (2013, portfolio-level
returns) found no significant effect, Huang & Xuan found the informational
edge essentially disappeared post-Act, and a ScienceDirect study found
members slightly *underperforming* a comparable index before costs. Some
post-Act studies (Hanousek et al. 2023, a 2025 CEPR-summarized study) still
find ~5% abnormal returns concentrated among committee leaders trading in
sectors tied to their power — i.e., any edge is plausibly concentrated in
specific informed subgroups, not present across the full disclosure feed.
Data: free (Quiver Quantitative, Capitol Trades). If tested, restrict to
committee-relevant/leadership-tied trades rather than the full feed.

### 10. DeFi/tokenized-equity 24/7 arbitrage — unverified, most speculative, no source located
Trade tracking-error gaps between tokenized/synthetic equities/commodities
(traded 24/7 on DeFi rails) and their underlying reference asset's actual
price, particularly on weekends/holidays when the underlying market is
closed but the synthetic keeps trading. **No academic paper or credible
practitioner source could be located or verified** for this exact
strategy — carried over from the original submission unconfirmed. The
general phenomenon (wider tracking error/off-hours volatility for
synthetic-asset platforms) is directionally plausible mechanically, but
this is the least-verified candidate on the list. Data: likely feasibility
risk — would need feeds from DeFi/tokenization platforms, a newer and less
standardized data landscape. **Do not build — watch-and-revisit only**
until a dedicated source-check is done.

### 11. MAX effect / lottery-demand anomaly (avoid or fade high-MAX stocks) — RESOLVED, NO-GO (2026-07-17)
`docs/research/2026-07-17-max-effect-nogo.md`. Built as `max_effect`
strategy (`src/ggTrader/lab/strategies/max_effect.py`), 6-combo WFO, SP500,
2015-present: OOS Sharpe 0.39-0.45 vs SPY 0.76-0.88 (healthy WFE 0.84-0.97,
gate pass 31/42 — not an overfitting rejection, a genuine "too weak to
beat SPY" result). Diversification follow-up (same methodology as
`idio_vol`'s 2026-07-07 check): OOS correlation to the deployed core =
0.692, higher than `idio_vol`'s already-insufficient 0.447 — rejected as a
blend candidate too. Closed both standalone and diversification angles;
remove from active backlog.

Compute MAX (highest single daily return over the trailing month) per
stock; underweight/avoid the highest-MAX decile (optionally
overweight/long the lowest-MAX decile), monthly rebalance, as a
portfolio-construction filter layered on top of the existing ensemble
signal (not a replacement). Source: Bali, Cakici & Whitelaw, "Maxing Out:
Stocks as Lotteries..." (JFE 2011) — foundational, finds high-MAX stocks
significantly underperform going forward. Follow-up work is more contested:
a 2026 "MAXβ" beta-neutralized refinement argues part of the original
effect is equity-issuance-driven; other work (Gorman et al.) reframes it as
an overreaction/reversal effect tied to idiosyncratic skewness rather than
pure lottery demand. Data: free (computable directly from existing daily
OHLCV, no new vendor). Differs from prior work: cross-sectional
skewness/behavioral-demand filter, mechanically unrelated to momentum
ranking or the VIX-regime filter already tested — but note the effect has
degraded/been re-explained in newer literature; treat as a portfolio tilt
to test cautiously, not a standalone strategy.

### 12. Post-earnings-announcement drift (PEAD), lower-coverage names only — RESOLVED, NO-GO (2026-07-17)
`docs/research/2026-07-17-pead-nogo.md`. Built as `pead` strategy on a new
yfinance earnings-surprise pipeline (`earnings_surprise_data.py`, 62,552
rows, 949 symbols). **Important methodological result, not just a
rejection**: an initial long-window (2015-2026) SP500 test looked
genuinely promising — beat SPY (Sharpe 0.93 vs 0.76-0.90), healthy WFE
(1.01), low regime-halt (20%), moderate 0.422 correlation to the deployed
core. A matched-window retest (2021-2026, the deployed blend's own
validation window) and 4-sleeve blend test overturned it: standalone edge
evaporates (Sharpe 0.58, tied with SPY) and adding it to the deployed
blend makes it worse (Sharpe 1.14→1.06, MaxDD -5.39%→-6.51%). Russell 2000
(the "lower-coverage" test the candidate itself calls for) also only tied
SPY with a much higher regime-halt rate (60%), contrary to the
literature's expectation. Confirms: never trust a standalone "beats SPY"
result without a matched-window retest against the deployed blend's exact
eval window.

*Overlaps with `RESEARCH_SNAPSHOT.md` §6 internal Rank 2 (PEAD) — this
entry is the more detailed, citation-backed version, including the 2022/2025
large-cap-decay findings the internal entry doesn't have; check here first.*

After an earnings release, rank surprise magnitude (actual vs. consensus
EPS); long strong positive surprises, avoid/short strong negative
surprises, hold several weeks — restricted to lower-coverage,
smaller/mid-cap names rather than the liquid large caps already traded.
Recent research disagrees sharply on whether PEAD still exists in large,
liquid U.S. stocks: Martineau's "Rest in Peace PEAD" (2022) argues the
effect vanished in non-microcap U.S. stocks by 2006 (decimalization +
faster electronic arbitrage); a 2025 replication (Subrahmanyam,
2001–2024 data, via UCLA Anderson Review) found the effect is highly
sensitive to microcap inclusion — much weaker/absent once excluded. Some
counter-evidence exists (a 2024 NBER paper ties PEAD strength to retail
contrarian-trading intensity; Lan/Xie/Mi/Zhang 2023–24 propose refined
surprise measures with continued profitability, though China-market-
focused). Data: free/cheap (Financial Modeling Prep, Zacks free tier).
Ranked mid-list specifically because the strongest evidence says PEAD may
no longer work in the exact universe (liquid large/mid-cap U.S. equities)
already traded here — worth testing with real skepticism, probably only
worthwhile extended into smaller-cap/lower-coverage names.

### 13. S&P 500 index-reconstitution deletion overshoot (fade the deletion) — RESOLVED, NO-GO (2026-07-17)
`docs/research/2026-07-17-index-deletion-fade-nogo.md`. Zero new data
infrastructure needed — built directly on the already-maintained
point-in-time SP500 membership history
(`ggTrader.data.core.index_constituents`). WFO (SP500, 42 folds): OOS
Sharpe 0.30 vs SPY 0.76, MaxDD **-68.7%** (worst of any candidate closed
this session, nearly double SPY's own drawdown), gate pass 17/42 (40%),
regime halt 32/42 folds (76%). Likely explanation: many real S&P 500
deletions reflect genuine fundamental deterioration (bankruptcy risk,
earnings collapse), not just mechanical index-committee timing — "buy the
deletion and wait for reversion" buys falling knives as often as
oversold-but-fine names. Also found and fixed a real infrastructure bug
along the way: `simulate_weights` crashed on an all-empty-combo fold
(a scenario only a sparse-event strategy like this one triggers) —
fixed with a flat-equity short-circuit, benefits any future sparse-event
strategy in this lab.

On announcement of an S&P 500/Russell deletion, go long the deleted stock
shortly after the effective reconstitution date, betting on mean-reversion
of mechanical forced-selling pressure from index/benchmark-tracking funds.
A 2025 arXiv paper ("On the Hidden Costs of Passive Investing") finds
1990–2002 additions saw ~8.8% abnormal returns around reconstitution
(~4% reversing over the following month) with deletions showing the
mirror-image pattern that also partially reverses. Important caution:
Greenwood & Sammon's "The Disappearing Index Effect" (HBS working paper)
documents this price effect has been shrinking over decades and is less
reliable in the S&P 500 specifically as more additions arrive via
"migration" from related indices (telegraphed further in advance, arbitraged
away faster). Data: free (public reconstitution announcements + standard
OHLCV). Flagged as more speculative than the other "structural" candidates
given credible evidence the effect has been decaying toward zero in the
exact index already traded here.

### 14. Option-implied volatility skew as a return predictor — RESOLVED, INFEASIBLE (2026-07-19)
**Checked and confirmed infeasible for an honest historical WFO** — same
class of blocker as #2 (analyst revisions) and #7 (Anomaly-Driven Demand):
no free or already-integrated source has the historical depth this
project's WFO methodology needs (many folds, each 12mo train + 3mo test,
spanning back to ~2015-2018 across other strategies in this lab).
Concretely checked two sources: (1) `yfinance`'s `option_chain()` — its
signature (`date=None, tz=None`) takes only a *future* expiry selector, no
historical/as-of parameter; it is current-snapshot-only, identical to the
`eps_trend`/`eps_revisions` problem found for #2. (2) Alpaca's option-bars
API (already integrated in this project for equities/paper trading) —
queried a real contract (`AAPL260116C00200000`) for both 2022-01 and
2024-01 date ranges and got zero bars both times; Alpaca's options market
data only started in 2024, and even then a historical panel needs to know
*which contracts were listed* on each past date, which Alpaca's contract
endpoints don't expose historically either. Deep historical options-chain
data with clean bid/ask across strikes remains a paid-vendor problem
(OptionMetrics IvyDB is the academic standard) exactly as the original
candidate write-up flagged. Combined with the pre-existing note that this
signal may just be re-deriving #3's already-NO-GO'd borrow-cost signal
through a noisier instrument, this is not worth pursuing further without a
paid-data decision. Deprioritized alongside #2/#7/#4 — no code built.

*Overlaps with `RESEARCH_SNAPSHOT.md` §6 internal Rank 3 (options-derived
conditioning/covered-call overlay) — this entry is the more detailed,
citation-backed version, including the likely redundancy with #3's
borrow-cost signal that the internal entry doesn't flag; check here first.*

Compute IV skew (OTM put IV minus ATM call IV) or vol spread (call IV minus
put IV) per name; use cross-sectionally — steep negative skew/high
put-implied IV historically precedes weaker subsequent returns. Sources:
Xing, Zhang & Zhao (2010) and Cremers & Weinbaum (2010) are the foundational
papers. Important caveat from a 2025 ScienceDirect paper: once the
option-implied *borrowing fee* is properly accounted for, most of the
skew's/spread's predictive power disappears — i.e., this signal may largely
just be re-deriving the stock-borrow-cost signal (#3) through a noisier
instrument, not adding independent information. Data: **clearest feasibility
risk on the list** — deep historical options-chain data with clean
bid/ask across strikes typically requires a paid vendor (OptionMetrics
IvyDB is the academic standard); free/cheap current-day chains exist but
historical depth for a proper backtest is genuinely costly. Do not commit
engineering time until the data-cost question is resolved; lower priority
than #3 given the likely redundancy.

### Summary triage table (14 candidates)

| # | Candidate | Signal category | Data cost | Verification status | Distinctness |
|---|---|---|---|---|---|
| 1 | Insider cluster buying (Form 4) | Corporate-insider intent | Free | Well-established | High |
| 2 | Analyst revision momentum | Fundamentals/expectations | Free–low | Well-established | High |
| 3 | Short interest / cost-to-borrow | Securities-lending microstructure | Mixed (real-time paid) | Well-established, but core cite weaker than summarized (insignificant value-weighted result) | High |
| 4 | Crypto funding-rate carry | Derivatives market-neutral carry | Free | Verified, but recent decay confirmed — Sharpe negative in 2025 | High (non-directional) |
| 5 | Stealthy shorts | Order-flow microstructure | Mixed (retail proxy noisier) | Paper verified real; exact bps figure unconfirmed | High |
| 6 | Form 144/Form 4 non-execution | Corporate-insider intent | Free | Real preprint; directly-verified result is economically negligible (μ=0.00%) | Lowest priority |
| 7 | Anomaly-Driven Demand | Mechanical rebalancing flow | Free–low | Paper verified real (2026) | High |
| 8 | Retail-attention-conditioned factors | Behavioral/alt-data | Free (Google Trends) | Concept verified; cited stats unconfirmed | High |
| 9 | Congressional (STOCK Act) trades | Regulatory/informational access | Free | Verified, but post-Act evidence genuinely mixed/contested | Medium — restrict to committee-tied trades |
| 10 | DeFi synthetic-equity tracking error | Cross-market structural gap | Feasibility risk | Unverified — no source located | High, but most speculative overall |
| 11 | MAX effect | Behavioral/skewness | Free | Well-established original; newer work contests mechanism | High, but contested in recent lit |
| 12 | PEAD (lower-coverage names) | Earnings-surprise drift | Free–low | Well-established, but debated for large-caps specifically | Medium (large-cap decay evidence) |
| 13 | Index-deletion overshoot fade | Forced-flow mean reversion | Free | Well-established, but effect documented as decaying | Medium (decaying effect) |
| 14 | Options IV skew | Derivatives-market information | Paid for real backtest | Well-established, but may be redundant with #3 | Low-medium (possible duplicate of #3) |

**First wave (free/cheap data, distinct mechanism, least-contested
evidence):** #1, #2, #7, plus the free-data-only (bi-monthly short
interest) version of #3.
**Second wave (worth testing, real caveats already built in):** #4 (test
with the 2025 decay finding in mind), #5 (retail proxy noisier than the
paper's institutional data), #11 (contested mechanism in recent lit).
**Third wave (lower confidence / smaller allocation / pending
verification):** #9 (contested post-Act evidence), #8 (concept solid,
stats unconfirmed), #12–#13 (documented decay in the exact universe
already traded), #14 (paid-data feasibility risk, possible redundancy
with #3).
**Do not build without further source-checking:** #10 (no source located
at all — placeholder only).

---

## 2026-07-19 batch — Master Consolidated Cross-Asset Strategy Report (24 candidates, A–X)

Source: web-research pass explicitly scoped to non-US-equity asset classes
(FX, commodities, Treasuries/rates, crypto), per the discovery-pivot
recommendation added to `web-strategy-research-prompt.md` after the
2026-07-16 batch's 9 equity-diversification-sleeve candidates all failed
(see `RESEARCH_SNAPSHOT.md` §4/§6). This report merges three separate
research passes (8 + 8 + 12 ideas = 28 before dedup) into 24 distinct
candidates, each already run through a citation-verification pass (primary
sources checked, one fabricated citation caught, one misattribution
caught, one scope-narrowing caught). No overlap found with any of the
2026-07-16 batch's 14 candidates or with this lab's tried/rejected roster —
every idea here is a genuinely different asset class from everything
tested so far, exactly the gap the pivot was meant to fill.

**Two soft relations to flag (not duplicates, different mechanism):**
- **Idea P (crypto spot-perp calendar basis)** and **idea M (cross-asset
  carry rotation)** are adjacent to, but distinct from, this lab's already-
  infeasible candidate #4 (perpetual-futures *funding-rate* carry, rejected
  2026-07-20 — Kraken retains only ~1yr of funding history). P/M trade the
  *futures-to-spot basis* (quarterly-expiry convergence) or *rate
  differentials across assets*, not the funding rate itself — worth
  re-checking data depth independently rather than assuming the same
  infeasibility carries over.
- **Idea L (headline/LLM sentiment)** is adjacent to, but distinct from, the
  paused candidate #8 (Google Trends retail-attention). L uses NLP sentiment
  scoring of news text; #8 uses raw search-volume spikes. Different data
  source, different mechanism — no redundancy, just the same broad
  "alternative/behavioral data" category.

**Status: `untriaged` for all 24.** Recommended triage order below follows
the report's own tiers (verification confidence + how well it diversifies
from the existing all-US-equity ensemble), not necessarily backtest
promise — nothing here has been backtested yet.

### Tier 1 — Best supported (verified, high-pedigree, low equity correlation)

**A. Cross-currency basis / CIP deviation harvesting (FX).** Tilt G10 carry
positions using the cross-currency basis swap spread as a dealer
balance-sheet-stress signal (post-Basel III capital constraints prevent the
basis arbitraging to zero). Sources: Du, Tepper & Verdelhan, *"Deviations
from Covered Interest Rate Parity,"* *Journal of Finance* 73 (2018);
Dao, Gourinchas & Itskhoki, *"Breaking Parity,"* NBER WP 34443 / IMF WP
2025/153 — both verified real. ⚠ A third citation ("arXiv:2605.20137"
attributed to Du/Tepper/Verdelhan) is misattributed — that paper is
solo-authored by Useong Shin; cite separately if used. Data: free (FRED
cross-currency basis series, FX ETFs). **Confidence: High** — the
best-supported idea across all three source reports.

**B. Commodity carry + trend + short-term basis reversal.** Combine
term-structure carry/roll yield, cross-sectional trend with a
volatility-regime filter, and weekly mean-reversion in the front-minus-
second-month futures basis. Sources: Koijen, Moskowitz, Pedersen & Vrugt,
*"Carry,"* *JFE* 127 (2018); Bloomberg, *"Capturing curve, carry and trend
premia in commodity markets"* (Feb 2026); Rossi, Zhang & Zhu, *"Short-Term
Basis Reversal,"* SSRN 5250499 (May 2025, Sharpe >1 pre-cost, independently
replicated by multiple practitioner blogs). Data: free/cheap (PDBC/GCC/DBC
ETF proxies for carry/trend; continuous futures data improves the
basis-reversal leg). **Confidence: High** — three independently verified
sources, the strongest thematic overlap of any idea across the source
reports.

**C. Treasury duration / term-premium factors.** Two variants: (i) static
investable term-structure factors on Treasury ETFs (steepener/flattener,
roll-down) — verified via Filipović, Pelger & Ye, *"Shrinking the Term
Structure,"* NBER WP 32472 (2024); (ii) macro-regime-conditional duration
rotation (HMM classifying growth/inflation regimes) — sources (GitHub repo,
unverified arXiv/SSRN IDs) **not independently verified**. Data: free
(TLT/IEF/SHY, FRED macro series). **Confidence: High** for (i), **Low** for
(ii) — prototype the static factor version first. **Status: C-i untriaged;
C-ii deprioritized — low confidence, do not resurface without a
verified source.**

**D. FOMC-window spillover trading in country ETFs.** Trade U.S.-listed
country ETFs representing markets closed during FOMC announcements —
price discovery migrates to the ETF while the local market is shut.
Sources: Kansoy (Oxford), *"The Immediate Global Impact of US Monetary
Policy"* (SSRN 5871422, Dec 2025) — ETF returns predict overnight local-
index gaps only for closed markets, ~$280B spillover destruction within 30
min across 37 countries; Neuhierl & Weber, *"Monetary Momentum,"* NBER WP
24748 (2018) — also the correct anchor for idea K below. Data: free/cheap
(FOMC calendar, intraday country-ETF bars). **Confidence: High** — both
sources verified, Kansoy's paper is strong and directly on-point.

**E. Dynamic FX hedge overlay (carry + value + trend).** Vary the hedge
ratio on international equity/bond exposure using carry, PPP-value, and
trend signals instead of a static hedged/unhedged policy. Source: Castro,
Hamill, Harber, Harvey & Van Hemert, *"The Best Strategies for FX
Hedging,"* *Journal of Portfolio Management* 51(9) (2025) — published,
Man Group + Campbell Harvey authorship. Data: free/cheap (spot FX, rate
differentials, CPI). **Confidence: High.**

**F. Cross-asset base-pair selection (portfolio construction).** Decompose
cross-sectional signals (value/momentum/carry) into asset-pair portfolios,
keep only pairs with strong own- and cross-asset explanatory power across
a multi-asset ETF universe — a portfolio-construction technique, not a
single-signal anomaly. Source: Goulding & Harvey, *"Investment Base
Pairs,"* SSRN 5193565 (March 2025) — 1,710 futures pair portfolios, top
pairs roughly triple average annualized returns at fixed leverage over 20
years (3.4%→10.4% aggregate). Data: full academic version is futures-heavy;
an ETF-proxy version is a feasible home-lab approximation. **Confidence:
High.**

### Tier 2 — Real, verified citations with a specific caveat attached

**G. Bond ETF discount-to-NAV reversal (credit/high-yield).** Buy bond
ETFs at unusually wide NAV discounts (illiquid underlying market, slow to
reprice) — the ETF leads, the NAV lags. Sources: Fulkerson, Jordan &
Riley, *"Predictability in Bond ETF Returns,"* *Journal of Fixed Income*
23(3) (2013, ~11.5%/yr long-short alpha, dated in-sample); Karmaziene &
Terrada, *"Fast ETFs, Slow Bonds,"* *Finance Research Letters* 90 (2026) —
confirms the mechanism was active during 2022-2023 tightening specifically
in high-yield ETFs. **Confidence: High** on both citations; treat the 2013
alpha figure as dated, lean on the 2026 paper as the decision-relevant
confirmation.

**H. Stablecoin-stress signal for BTC/ETH (jump-risk hedge).** Use
stablecoin depeg/stress events as a crypto plumbing signal — reduce crypto
beta or run a short-horizon downside trade on stress. Sources: Perez Riaza
& Gnabo, *"From depegs to jumps,"* *JIMF* 155 (2025) — a Tether depeg
raises BTC/USD jump probability nearly 5x within 5 minutes; Ma, Zeng &
Zhang, *"Stablecoin Runs and the Centralization of Arbitrage,"* NBER WP
33882 (2025, now *Journal of Finance*) — Tether allows only ~6 redeeming
agents/month, the structural reason pegs become stress-transmission
channels. **Confidence: High** on these two; a third citation
(Adams/Ibert/Liao) remains unverified.

**I. CIP-adjacent: dynamic hedge/factor rotation for international
equities.** Country-momentum rotation across developed ex-US equity ETFs,
layered with factor rotation + idea E's dynamic currency-hedge overlay.
⚠ Its currency-hedge-signal citation (SSRN 6447259, Bräuer) may have its
finding reversed — the paper appears to show ETF hedge-ratio choices are
explained by *survey* FX expectations, not that portfolio-implied
expectations outperform surveys as claimed. **Confidence: Medium** on the
country/factor-rotation framing generally, **Low** on the specific
hedge-signal citation until re-confirmed directly.

### Tier 3 — Plausible mechanisms, unverified or thin sourcing

**J. Session-aware crypto intraday mean reversion/trend.** Source: Wątorek,
Skupień, Kwapień & Drożdż, *"Decomposing cryptocurrency high-frequency
price dynamics into recurring and noisy components,"* *Chaos* 33 (2023) —
verified, three session-aligned activity phases, bursts around US macro
releases. **Confidence: High** on the citation; smaller in scope than the
Tier 1 ideas (a single intraday signal, not a portfolio-level strategy).

**K. Pre-FOMC drift in long Treasuries (test-and-discard).** The
"has it decayed post-2023" claim traces to an unverified AEA 2026 draft —
use idea D's already-verified Neuhierl & Weber paper as the evidentiary
anchor instead, then test the post-2023 decay question empirically.
**Confidence: Medium** — real phenomenon via a verified citation, decay
claim itself unverified.

**L. Headline/LLM sentiment on small/mid-cap equities.** Sources:
Lopez-Lira & Tang, *"Can ChatGPT Forecast Stock Price Movements?,"*
arXiv:2304.07619 (2023, forthcoming *JFE*) — GPT scores predict returns,
stronger in small caps and after negative news; Saqur, Kato, Vinden &
Rudzicz, *"NIFTY Financial News Headlines Dataset,"* arXiv:2405.09747.
**Confidence: High** now that a citation-ID swap from the original report
is fixed. (See relation note above re: candidate #8.)

**M. Cross-asset carry rotation (risk-on/off via carry, not trend).** Same
AQR "Carry" paper as idea B, applied to a broader multi-asset universe
rather than commodities specifically. **Confidence: High** on the
citation (shares B's evidentiary base).

**N. Crypto options volatility risk premium (DVOL vs. realized vol).**
Deribit DVOL methodology + practitioner pieces, none independently
verified for crypto specifically; the underlying mechanism (implied >
realized on average, delta-hedged harvest) is standard, well-established
options theory generally. **Confidence: Medium** on mechanism, **Low** on
crypto-specific sourcing. **Status: deprioritized — low confidence, do
not resurface without a verified crypto-specific source.**

**O. Stablecoin yield arbitrage (CeFi/DeFi rate differential).** Industry
reports only (Galaxy Research, DeFiLlama, Messari), no academic
verification — but the rate fragmentation itself is observable market
structure, not a contested claim. **Confidence: Medium** on mechanism,
**Low** on cited figures. **Status: deprioritized — low confidence, do
not resurface without academic verification.**

**P. Crypto spot-perp calendar basis trade.** CME commentary + practitioner
pieces, no academic source; the mechanism (quarterly futures converge to
spot, basis reflects funding/term-structure expectations) is real,
documented institutional practice (post-spot-ETF CME basis trading).
**Confidence: Medium** on mechanism, **Low** on specific sources. (See
relation note above re: candidate #4.) **Status: deprioritized — low
confidence, do not resurface without a verified source.**

**Q. Token unlock event-driven short.** Data-vendor calendars (DefiLlama,
CoinGlass, Tokenomist), no empirical crypto-specific study found — the
equity IPO-lockup-expiry analogy is real literature. **Confidence:
Low-Medium** — reasonable mechanism, untested claim. **Status:
deprioritized — low confidence, do not resurface without an empirical
crypto-specific study.**

### Tier 4 — Weak sourcing or corrected/flagged claims — most caution

**R. G10 currency three-factor core (carry + momentum + value).** ⚠ The
original report's top-ranked idea rested on an apparently **fabricated**
citation ("Survival of the Fittest," claimed SSRN 6609879 — not found
under that title or ID anywhere). The three-factor concept itself is real
and broadly supported in FX literature (see idea E; Menkhoff, Sarno,
Schmeling & Schrimpf on currency momentum, or AQR's "Value and Momentum
Everywhere" dataset are real alternatives) — replace the citation before
treating this as strongly evidenced. **Confidence: Low** on the citation,
**Medium** on the underlying concept. **Status: deprioritized — low
confidence, do not resurface unless a real citation for the three-factor
core is found (idea E already covers the FX carry/value/trend concept with
a solid citation).**

**S. Treasury duration rotation via HMM regime-switching.** Same as idea
C's variant (ii) — leans entirely on a GitHub repo and unverified IDs.
**Confidence: Low** — see Tier 1C for the better-supported static-factor
alternative. **Status: deprioritized — duplicate of C-ii, do not
resurface.**

**T. Commodity seasonality (harvest/planting cycles).** ⚠ The cited paper
(PMC11305200) is real but studies economic-policy-uncertainty co-movement,
not planting/harvest seasonality — off-topic relative to the claim.
**Confidence: Low** pending a genuine seasonality citation. **Status:
deprioritized — low confidence, do not resurface without a genuine
on-topic citation.**

**U. ETH/BTC ratio mean-reversion with macro filter.** All-practitioner-
blog sourcing, no academic backing found. **Confidence: Low** — fast-test-
and-discard only. **Status: deprioritized — low confidence, do not
resurface without academic backing.**

**V. Energy pre-holiday seasonal sleeve.** Self-flagged by its own source
report as non-academic (Quantpedia write-up only). **Confidence: Low.**
**Status: deprioritized — low confidence, do not resurface without
academic backing.**

**W. DeFi delta-neutral concentrated LP + funding harvest.** Mechanism is
sound (observable market facts), but operationally the heaviest idea in
the whole pool (smart contracts, keeper bots, multi-venue connectivity).
**Confidence: Medium** on mechanism, flagged as a long-shot for
engineering-effort reasons, not citation reasons.

**X. Automated multi-asset strategy discovery (QuantEvolve-style).**
Source: Yun, Lee & Jeon, *"QuantEvolve,"* arXiv:2510.18569 (2025), Qraft
Technologies, ACM ICAIF-affiliated workshop — verified. **Confidence:
High** on the citation; this is a discovery *framework*, not a validated
strategy — the most speculative application in the pool even with a solid
citation.

### Summary triage table (24 candidates)

| # | Candidate | Asset class | Data cost | Confidence | Status | Notes |
|---|---|---|---|---|---|---|
| A | Cross-currency basis / CIP harvesting | FX | Free | High | untriaged | Best-supported idea overall |
| B | Commodity carry+trend+basis reversal | Commodities | Free/cheap | High | untriaged | 3 independent verified sources |
| C-i | Treasury term-structure factors (static) | Rates | Free | High | untriaged | Prototype this variant first |
| C-ii | Treasury duration rotation (HMM regime) | Rates | Free | Low | **deprioritized** | Unverified sourcing |
| D | FOMC country-ETF spillover | FX/equity (intl) | Free/cheap | High | untriaged | Strong recent primary source |
| E | Dynamic FX hedge overlay | FX | Free/cheap | High | untriaged | Published, high-pedigree |
| F | Cross-asset base-pair selection | Multi-asset | Futures-heavy (ETF-proxy feasible) | High | untriaged | Portfolio-construction insight |
| G | Bond ETF discount-to-NAV reversal | Credit/HY bonds | Free | High | untriaged | Lean on 2026 paper, not 2013 figure |
| H | Stablecoin-stress jump-risk hedge | Crypto | Free/cheap | High | untriaged | 2 verified, high-tier papers |
| I | Country/factor rotation + FX hedge signal | FX/equity (intl) | Free/cheap | Medium/Low | untriaged | Re-confirm Bräuer paper's actual claim |
| J | Session-aware crypto intraday | Crypto | Free/cheap | High | untriaged | Small scope, cleanest citation match |
| K | Pre-FOMC Treasury drift | Rates | Free | Medium | untriaged | Use idea D's citation, not the unverified one |
| L | Headline/LLM sentiment (small/mid caps) | Equity (alt-data) | Free/cheap (LLM cost) | High | untriaged | See relation note re: #8 |
| M | Cross-asset carry rotation | Multi-asset | Free/cheap | High | untriaged | Shares idea B's evidentiary base |
| N | Crypto options vol risk premium | Crypto derivatives | Feasibility risk | Medium/Low | **deprioritized** | Standard mechanism, crypto sourcing thin |
| O | Stablecoin yield arbitrage | Crypto/DeFi | Free (industry reports) | Medium/Low | **deprioritized** | No academic verification |
| P | Crypto spot-perp calendar basis | Crypto derivatives | Free/cheap | Medium/Low | **deprioritized** | See relation note re: #4 |
| Q | Token unlock event-driven short | Crypto | Free (calendars) | Low-Medium | **deprioritized** | Untested in crypto specifically |
| R | G10 currency 3-factor core | FX | Free/cheap | Low/Medium | **deprioritized** | ⚠ fabricated citation, real concept |
| S | Treasury HMM regime rotation | Rates | Free | Low | **deprioritized** | Duplicate of C-ii |
| T | Commodity seasonality | Commodities | Free | Low | **deprioritized** | ⚠ cited paper is off-topic |
| U | ETH/BTC ratio mean-reversion | Crypto | Free | Low | **deprioritized** | Blog-only sourcing |
| V | Energy pre-holiday seasonal | Commodities | Free | Low | **deprioritized** | Self-flagged thin by its own source |
| W | DeFi delta-neutral LP + funding harvest | Crypto/DeFi | Feasibility risk (infra-heavy) | Medium | untriaged | Engineering long-shot, not citation risk — kept active despite Tier-4 grouping |
| X | Automated multi-asset strategy discovery | Meta/framework | N/A | High (citation) | untriaged | Framework, not a strategy — most speculative application |

**Deprioritized (11 ideas: C-ii, N, O, P, Q, R, S, T, U, V).** None of
these were tried/tested — they're low-confidence on sourcing, not
NO-GO — so they stay in this file as the record of "already looked at,
weakly sourced." `web-strategy-research-prompt.md` now lists them by name
so a future discovery pass doesn't waste output resurfacing the same or
similar weakly-sourced ideas without a stronger citation than what's
already been found here.

**First wave (Tier 1, all independently verified, genuinely diversifying
from the all-US-equity ensemble):** A, B, D, E, F, C-i (in roughly that
priority order per the source report).
**Second wave (Tier 2, verified with a fixable caveat):** G, H, J.
**Needs a citation fix before serious investment:** R (replace the
fabricated citation), I (re-confirm the Bräuer paper's actual claim), A's
misattributed arXiv ID (easy fix — correct author is Useong Shin).
**Lowest priority given current sourcing (Tier 3/4 minus the above):**
C-ii/S, K, L, M, N, O, P, Q, T, U, V, W, X — mechanisms are often real
market facts, but treat stated Sharpe/alpha figures as unconfirmed until
checked directly; budget cheap, fast backtests to falsify quickly rather
than committing real research time.
