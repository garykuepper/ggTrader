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

## 2026-07-19 batch — Master Strategy Candidate Register (evidence-screened, corrected 2026-07-19)

**This replaces the earlier tiered draft of the same batch entirely.** An
independent review of that draft made a key methodological point: source
validation, mechanism validation, and tradable-strategy validation are
three different things, and a single "Confidence" score collapsed them
into one. This version separates them into four independent ratings (see
below), corrects several citation errors the review caught (including one
case where a real paper had wrongly been called "fabricated" — see item
18 below), and reorganizes by **strategy type** — active trades, risk
overlays, portfolio-construction methods, parked hypotheses — rather than
by paper prestige. **No return, alpha, Sharpe, drawdown, or correlation
figure in any source document has been independently reproduced; this
register validates literature mapping and mechanism logic, not
backtests.**

### How to read this batch

Each candidate carries four independent ratings instead of one:

| Field | Values | Answers |
|---|---|---|
| **Evidence status** | Peer-reviewed / Accepted / Working paper / Practitioner / Unsupported | How mature and credible is the source? |
| **Rule correspondence** | Direct / Close adaptation / Mechanism only / Unrelated | Did the source test *this exact* tradable rule, or just the underlying mechanism? |
| **Implementation class** | Retail / Professional / Institutional | Can the proposed instruments actually reproduce the studied effect? |
| **Validation stage** | Literature only / Replicated / Walk-forward OOS / Shadow live / Capital live | What has been independently established, as opposed to asserted? |

**Every candidate here is at Validation stage: Literature only** unless
noted — that's true across the board and is the main thing this register
is not yet claiming. This maps onto this lab's own methodology as the
first of many required gates: literature review is not a substitute for
this project's own honest walk-forward/NDH/DSR gate framework
(`src/ggTrader/lab/wfo.py`) — a candidate reaching "high confidence" here
still starts at zero WFO evidence.

**Numbering:** items keep the register's own letter+number IDs (A1-A9,
B1-B4, C1-C2, plus lettered sub-splits like 18A/18B in the parked-
hypotheses section) rather than being renumbered into this file's running
1-25 sequence, since several items were split or merged from the prior
draft and a flat renumbering would obscure the mapping. A crosswalk to
the prior draft's numbering is noted per item below.

**One item from the prior draft (old item 13, cross-asset carry rotation,
same AQR "Carry" citation as A2/old item 2) was not addressed by this
independent review** — carried forward unchanged from the prior draft
rather than dropped; treat it as `untriaged`, same evidentiary base as A2.

**Relation to this lab's existing roster (unchanged from before):** A9
(direct CIP basis harvesting) and the parked spot-perp/calendar-basis
hypothesis are adjacent to, but distinct from, this lab's already-
infeasible candidate #4 (perpetual-futures funding-rate carry). The A8
LLM/headline-sentiment candidate is adjacent to, but distinct from, the
paused candidate #8 (Google Trends retail-attention).

---

# A. Active strategy replication queue

Candidates with a plausible path to a specific tradable rule, roughly
ordered by how directly the cited evidence maps to the proposed trade.

### A1. Dynamic FX hedge overlay (carry + value + trend)

*(Prior draft: item 4, Tier 1, "High" confidence.)*

**Mechanism.** Vary the hedge ratio on international equity/bond exposure
using carry, PPP-value, and trend signals instead of a static
hedged/unhedged policy.

**Source(s).** Castro, Hamill, Harber, Harvey & Van Hemert, *"The Best
Strategies for FX Hedging,"* *Journal of Portfolio Management* 51(9)
(2025) — verified, published, high-pedigree.

**Why it's plausible.** Currency exposure isn't random noise: carry
monetizes rate differentials, value exploits PPP mean reversion, trend
captures macro repricing.

**Data requirements.** Free/cheap — spot FX, rate differentials, CPI data.
Specify home currency, exposed-asset set, continuous vs. binary hedge
ratio, rebalance frequency, forward-roll treatment, and costs — hedged/
unhedged ETF switching is only an approximation of the paper's forward-
based implementation.

**How it differs from what's already been tried.** A separate currency/
hedging sleeve, not another equity-entry refinement.

**Evidence status:** Peer-reviewed/published. **Rule correspondence:**
Direct to close. **Implementation class:** Professional natively; retail
proxy available via hedged/unhedged ETF pairs. **Validation stage:**
Literature only. **This remains the cleanest source-to-rule
correspondence in the whole register — the top build candidate,
including for a home-lab/ETF workflow.**

**Status: untriaged.**

---

### A2. Commodity carry (standalone)

*(Prior draft: part of item 2, Tier 1 — split here into three
independently-evidenced sub-signals; do not assume they combine as a
single strategy without testing each separately.)*

**Mechanism.** Long commodity futures/ETFs in backwardation, short/flat
those in contango.

**Source(s).** Koijen, Moskowitz, Pedersen & Vrugt, *"Carry,"* *Journal of
Financial Economics* 127 (2018) — verified, foundational, covers
commodities among other asset classes.

**Why it's plausible.** Compensation for hedging pressure and convenience-
yield effects.

**Data requirements.** ETF proxies (PDBC, GCC, DBC) are a feasible but
approximate substitute for true futures-curve carry.

**How it differs from what's already been tried.** Different asset class,
structural (not price-action) driver.

**Evidence status:** Peer-reviewed. **Rule correspondence:** Direct for
futures; close adaptation for ETF proxies. **Implementation class:**
Professional (futures) / Retail (ETF approximation). **Validation stage:**
Literature only.

**Status: untriaged.**

---

### A3. Commodity medium-term trend

*(Prior draft: part of item 2/3, Tier 1 — split, see A2.)*

**Mechanism.** Cross-sectional 12-1 month momentum across a liquid
commodity ETF/futures universe, with a volatility-regime filter to avoid
crash periods.

**Source(s).** Bloomberg Professional Services, *"Capturing curve, carry
and trend premia in commodity markets"* (Feb 2026) — verified real,
practitioner-grade (product commentary for Bloomberg's BERY index, not
independent research).

**Why it's plausible.** Slow price adjustment to supply/demand shocks.

**Data requirements.** Broad commodity ETFs can approximate directional
trend reasonably well.

**How it differs from what's already been tried.** Same as A2.

**Evidence status:** Practitioner. **Rule correspondence:** Close
adaptation. **Implementation class:** Retail-feasible via ETFs.
**Validation stage:** Literature only.

**Status: untriaged.**

---

### A4. Short-term futures-basis reversal

*(Prior draft: part of item 2, Tier 1 — split, see A2. Data-feasibility
downgraded — see below.)*

**Mechanism.** Weekly mean-reversion in the front-minus-second-month
futures basis, traded cross-sectionally or in time series.

**Source(s).** Alberto Rossi, Yingguang Zhang & Yandi Zhu, *"Short-Term
Basis Reversal,"* SSRN 5250499 (May 2025) — verified, and unusually
well-corroborated by independent practitioner replication (Quantitativo,
QuantSeeker, CXO Advisory all discuss/replicate it). Weekly negative
autocorrelation in the front/second-month basis, pre-cost Sharpe >1,
present in commodities, equity index futures, and bonds.

**Why it's plausible.** Differential price sensitivity to news across the
futures curve.

**Data requirements.** **Important limitation the prior draft glossed
over:** broad commodity ETFs generally cannot reproduce this signal — it
requires actual adjacent-contract futures data (continuous front/second-
month series), not an ETF proxy. This is the least retail-friendly of the
three commodity sub-signals.

**How it differs from what's already been tried.** A genuinely new
anomaly (2025), not a repackaging of carry or trend.

**Evidence status:** Working paper, but unusually well externally
corroborated for one this new. **Rule correspondence:** Direct.
**Implementation class:** Professional (needs futures access) — **not** a
clean retail ETF strategy despite being cheap in data terms. **Validation
stage:** Literature only.

**Status: untriaged.**

---

### A5. Treasury term-structure factors

*(Prior draft: item 6, Tier 1. Citation status upgraded; important
implementation caveat added.)*

**Mechanism.** Curve-factor positioning (steepener/flattener, relative
value across duration buckets, roll-down) using investable, non-
parametric term-structure factors.

**Source(s).** Filipović, Pelger & Ye, *"Shrinking the Term Structure,"*
NBER WP 32472 (2024) — verified, high-pedigree (EPFL, Stanford). ⚠
**Update:** per Markus Pelger's own faculty page, this paper is now listed
as **accepted at the Review of Finance** — stronger than "NBER working
paper" status. ⚠ **Important implementation caveat the prior draft
missed:** the paper identifies **four** investable factors from a rich
cross-section of Treasury cash flows, including a fourth "complexity"
factor tied to recession performance. A simple SHY/IEF/TLT three-ETF
implementation provides only three broad duration buckets and **cannot be
presumed to reproduce the fourth factor.** A closer replication needs cash
Treasuries, STRIPS, Treasury futures, or a substantially richer
instrument set.

**Why it's plausible.** The term structure is driven by monetary-policy
expectations, inflation risk, and duration-risk appetite.

**Data requirements.** Free for the three-ETF approximation; richer
instruments needed for a fuller replication.

**How it differs from what's already been tried.** Curve/term-structure
signals, not price-only timing.

**Evidence status:** Peer-reviewed, now accepted at a strong journal.
**Rule correspondence:** Direct for the full model; **mechanism only**
for a 3-ETF approximation. **Implementation class:** Professional for the
full model; Retail for the explicitly-labeled approximation. **Validation
stage:** Literature only. **Label any ETF-only version as an exploratory
approximation, not a replication.**

**Status: untriaged.**

---

### A6. ETF-implied FX expectations (hedge/unhedged flow signal)

*(Prior draft: item 9, Tier 2. Its own deprioritized-caution status is
now itself contested — see below; keep as untriaged pending a direct
re-read, do not deprioritize.)*

**Mechanism.** Extract currency-return expectations from investors'
revealed choices between matched hedged and unhedged share classes of the
same underlying international ETF; use the inferred expectations to trade
FX or set hedge ratios.

**Source(s).** Leonie Bräuer, *"Exchange Rate Expectations and Currency
Demand"* (working paper, 2025 drafts).

⚠ **Correction to a correction:** the prior draft warned that this
paper's finding might be reversed — that survey expectations explain ETF
allocations, rather than ETF-implied expectations outperforming surveys
at forecasting FX returns. An independent review disputes that reading,
stating the paper reports portfolio-implied expectations **do** forecast
subsequent exchange rates more accurately than survey expectations, macro
models, and conventional currency-pricing factors — and that there are
two distinct questions in the paper (whether hedge choices relate to
expected returns, and whether expectations inferred from those choices
predict subsequent FX rates), both of which the paper is said to support.
**This reversal has not been independently re-confirmed by directly
re-reading the paper's own claims section a third time** — given that
this citation has now been characterized two different ways by two
different passes, the responsible position is to flag it as **contested
pending a direct re-read**, not to simply flip the verdict a second time.
Read the paper's results section yourself before relying on either
characterization.

**Why it's plausible.** Extracting beliefs from capital-backed choices
(hedge ratio selection) is conceptually more grounded than survey-based
sentiment, since it reflects money actually at risk.

**Data requirements.** Feasible but tedious — matched ETF pairs,
consistent AUM history, clean same-exposure/different-hedge-status
mapping. The operational difficulty (assembling point-in-time AUM, flow,
and holdings data without survivorship or staleness problems) is real
regardless of which reading of the finding is correct.

**How it differs from what's already been tried.** A revealed-preference
FX signal from investor hedging choices, not another U.S. equity
characteristic sort.

**Evidence status:** Working paper (real, exists). **Rule
correspondence:** Contested — Direct per the review's reading, Mechanism-
only-with-reversed-sign per the prior reading. **Implementation class:**
Professional/complex data assembly. **Validation stage:** Literature
only, and specifically flagged as needing direct primary-source
confirmation before use either way.

**Status: untriaged — do not build without first directly re-reading the
paper's results section to resolve which characterization is correct.**

---

### A7. Pre-FOMC long-Treasury drift

*(Prior draft: item 11, Tier 3, "Medium" confidence, sourced to an
unverified "AEA 2026 draft." Citation now replaced with a verified,
current, directly-on-point paper — upgraded from parked/Tier-3 status.)*

**Mechanism.** Long TLT/IEF/EDV on the day before scheduled FOMC meetings,
exit at or around the announcement.

**Source(s).** ⚠ **Replaces the previously-cited unverified "AEA 2026
draft."** Jun Pan & Qing Peng, *"The Pre-FOMC Drift in Long-Term Treasury
Bonds"* (current draft June 2, 2026; also circulated as *"The Pre-FOMC
Drift and the Secular Decline in Long-Term Interest Rates,"* AEA 2026
conference program) — **verified real.** Documents positive,
statistically significant long-Treasury returns specifically on the day
*before* scheduled FOMC announcements (earlier than the equity pre-FOMC
drift), concentrated at longer maturities, driven primarily by the
term-premium component and uncertainty resolution (captured by a
pre-FOMC drop in the MOVE index). Explicitly contrasts with Lucca &
Moench (2015)'s original 24-hour-window equity framing. Neuhierl &
Weber's "Monetary Momentum" (NBER WP 24748, verified separately) remains a
valid *related* FOMC-drift reference but should not be the principal
citation for this specific Treasury-timing rule.

**Why it's plausible.** An uncertainty premium: elevated labor-market
uncertainty predicts and strengthens the drift; the pre-FOMC MOVE-index
decline captures uncertainty resolution.

**Data requirements.** Free — TLT/IEF/EDV daily prices, FOMC calendar.
Define the event window precisely (previous close → announcement open, or
another interval) before testing.

**How it differs from what's already been tried.** A pure event-driven
calendar anomaly with a direct, current (2026) academic citation.

**Evidence status:** Working paper (current, June 2026, from a credible
institution — SAIF/Shanghai Jiao Tong). **Rule correspondence:** Direct.
**Implementation class:** Retail/Professional. **Validation stage:**
Literature only — recent result, not yet broadly independently replicated
outside its own paper. Treat as a **replication candidate**, not an
established production anomaly.

**Status: untriaged — no longer test-and-discard given the stronger
citation; promoted into the active queue.**

---

### A8. Headline/LLM sentiment on small/mid-cap equities

*(Prior draft: item 12, Tier 3, "High" confidence. Publication claim
corrected; author-acknowledged decay risk added.)*

**Mechanism.** LLM/NLP classifier on news headlines generates a daily
long/flat/short signal; trade the delayed post-publication response in
smaller names.

**Source(s).** Lopez-Lira & Tang, *"Can ChatGPT Forecast Stock Price
Movements?,"* SSRN/arXiv, latest public revision **October 28, 2025**. ⚠
**Remove the "forthcoming Journal of Financial Economics" claim** — the
public SSRN/arXiv records confirm the paper and its latest working-paper
revision but do not independently confirm JFE acceptance; state it as
"working paper, publication status not independently confirmed." The
latest abstract adds two points worth including: the drift is stronger in
smaller stocks and after negative news (as previously noted), **and**
strategy returns decline as LLM use becomes more widespread — an explicit
crowding/decay warning from the authors themselves. Saqur, Kato, Vinden &
Rudzicz, *"NIFTY Financial News Headlines Dataset,"* arXiv:2405.09747 —
verified real; **use as a data resource, not as independent replication
of the alpha result.**

**Data requirements.** Point-in-time headline feed, immutable model/
version specification, latency assumptions, delisted-firm handling,
execution-cost threshold.

**How it differs from what's already been tried.** Not another equity
price-action/fundamental sort. See relation note above re: paused
candidate #8 (Google Trends) — different mechanism, no redundancy.

**Evidence status:** Working paper (real, actively revised). **Rule
correspondence:** Direct, with an author-acknowledged decay risk.
**Implementation class:** Professional (point-in-time data is the hard
part). **Validation stage:** Literature only.

**Status: untriaged — only with credible point-in-time data; do not
proceed on a look-ahead-biased headline feed.**

---

### A9. Direct CIP/cross-currency basis trading

*(Prior draft: item 1, Tier 1, "High" confidence, ranked #1 overall.
Materially reframed and downgraded for a retail/ETF workflow — read this
one carefully before treating it as the automatic top pick.)*

**Mechanism.** Harvest the covered-interest-parity deviation itself (the
cross-currency basis swap spread) directly, via forwards/swaps — **not**
as a tilt on an ETF-based carry portfolio.

**Source(s).** Wenxin Du, Alexander Tepper & Adrien Verdelhan, *"Deviations
from Covered Interest Rate Parity,"* *Journal of Finance* 73 (2018) —
verified, foundational. Mai Dao, Pierre-Olivier Gourinchas & Oleg
Itskhoki, *"Breaking Parity,"* NBER WP 34443 (2025) — verified,
high-pedigree, independently confirms the balance-sheet-constraint
mechanism.

⚠ **Important reframing:** the literature strongly validates the *CIP
deviation and its regulatory/balance-sheet cause* — it does **not**
validate "tilt a G10 carry ETF portfolio using the basis as a signal,"
which is a separate, untested predictive hypothesis layered on top.
**Direct** CIP arbitrage requires spot, forwards/swaps, funding,
collateral, and balance-sheet capacity that FX ETFs do not provide. Split
this into two distinct candidates: **(a) direct basis arbitrage**
(institutional-only), and **(b) a basis-conditioned carry tilt on retail
FX ETFs** (an unvalidated extension).

**Data requirements.** Free FRED basis series for monitoring; actual
harvesting requires institutional derivatives access.

**How it differs from what's already been tried.** Balance-sheet/
regulatory-constraint anomaly, not price prediction.

**Evidence status:** Peer-reviewed (both papers). **Rule correspondence:**
Direct mechanism for (a); mechanism-only for (b). **Implementation
class:** Institutional for (a); Retail proxy (unvalidated) for (b).
**Validation stage:** Literature only.

**Status: untriaged, but demoted from the previous draft's #1 ranking —
do not rank this first for a retail/ETF-based workflow, this project's
default. FX ETFs do not directly implement the documented arbitrage. It
belongs near the top only for a desk with actual forwards/swaps/funding
access, which this project does not have.**

---

# B. Risk and exposure overlays

Candidates better used as risk-management triggers than as standalone
directional trades.

### B1. Stablecoin stress signal

*(Prior draft: item 8, Tier 2, "High" confidence. Venue and directionality
corrected.)*

**Mechanism.** Use stablecoin depeg/stress events as a crypto plumbing
signal.

**Source(s).** Perez Riaza & Gnabo, *"From depegs to jumps,"* *Journal of
International Money and Finance* 155 (2025) — verified; a Tether depeg
raises BTC/USD jump probability nearly fivefold within 5 minutes. Ma, Zeng
& Zhang, *"Stablecoin Runs and the Centralization of Arbitrage,"* NBER WP
33882 (2025) — verified.

⚠ **Venue correction:** this paper is **accepted at the Review of
Financial Studies**, not the *Journal of Finance* as the prior draft
stated — confirmed directly via the lead author's own faculty page
(yimingma.com), which lists it explicitly as "Review of Financial
Studies, Accepted," also noting IMF Global Financial Stability Report and
Financial Times/Forbes/NPR coverage.

⚠ **Mechanism correction:** the depeg paper documents a sharp increase in
the *probability and magnitude* of BTC price jumps — it does **not**
establish that the expected jump is reliably negative. Depegs raise *jump
risk* (two-sided), not a guaranteed short opportunity. The runs/
arbitrage-concentration paper explains why pegs are structurally fragile;
it doesn't make every depeg an automatic short signal either.

**Recommended use:** a leverage-reduction trigger, a position-size/risk-
limit overlay, a convexity-purchase trigger where options are liquid, or
a directional short only when an independently validated price-direction
signal agrees — **not an unconditional short.**

**Data requirements.** Free — recent exchange-level minute bars via free
exchange APIs or CCXT-style pulls; cross-venue data quality (especially
around delistings) is the main practical risk.

**How it differs from what's already been tried.** A crypto-specific
collateral/settlement stress signal tied to market structure.

**Evidence status:** Peer-reviewed (both, one now at a top-tier venue).
**Rule correspondence:** Mechanism only for direction (jump risk is
two-sided). **Implementation class:** Professional. **Validation stage:**
Literature only.

**Status: untriaged — scope as a risk overlay, not a standalone
directional strategy, per the correction above.**

---

### B2. FOMC country-ETF price discovery

*(Prior draft: item 3, Tier 1, "High" confidence, framed as a standalone
spillover trade. Magnitude corrected; the trading-rule claim itself is
now unproven and needs its own test.)*

**Mechanism.** Originally proposed as a spillover *trade*: enter on the
ETF's early reaction, hold for the local market's opening gap.

**Source(s).** Fatih Kansoy (Oxford), *"The Immediate Global Impact of US
Monetary Policy"* (SSRN 5871422) — verified, real, current. ⚠ **Magnitude
correction:** the current draft reports approximately **$150–300
billion** in non-U.S. equity repricing for a one-standard-deviation shock
(a range), not a single fixed "$280 billion" figure as previously stated
— an earlier abstract/version may have used a different point estimate;
use the range from the current draft. The paper also notes announcement
effects do not fully reverse.

⚠ **Important reframing:** this is strong evidence of *price discovery* —
that U.S.-listed country ETFs rapidly and accurately incorporate FOMC
shocks while local markets are closed. It does **not** by itself
demonstrate that a trader can (1) observe the ETF's first 15–30 minute
reaction, (2) enter *after* that reaction, and (3) earn abnormal returns
waiting for the local market to reopen. By the time you observe and act
on the ETF's reaction, the ETF has already done the price discovery — the
later local-market gap is not automatically additional *ETF* profit; it's
evidence the ETF was right, not evidence there's a second bite of the
apple. Rename this candidate **"FOMC country-ETF price discovery and
residual continuation test,"** and separate two distinct proposed trades:
trading immediately from an independently measured rate surprise, versus
trading after observing the ETF's own reaction (the latter needs a direct
entry-to-exit return test before being called a strategy).

**Data requirements.** Free/cheap — FOMC calendar, intraday country-ETF
data.

**How it differs from what's already been tried.** An ex-U.S. price-
discovery effect created by asynchronous market hours around scheduled
macro events.

**Evidence status:** Peer-reviewed-quality working paper (Oxford). **Rule
correspondence:** High for the price-discovery mechanism; **Low-Medium
for the specific retail trading rule as originally proposed** — this
needs its own direct test. **Implementation class:** Retail/Professional.
**Validation stage:** Literature only for the mechanism; no test yet for
the actual proposed rule.

**Status: untriaged — rename applied, treat the two entry variants (trade
the surprise directly vs. trade after the ETF's reaction) as two separate
hypotheses to test.**

---

### B3. Bond ETF stale-NAV/fair-value dislocations

*(Prior draft: item 7, Tier 2, "High" confidence, framed as a raw-
discount-buying rule. Logical gap identified — the raw rule is downgraded;
a redesigned version is proposed.)*

**Mechanism.** Originally proposed as: buy raw NAV discounts, especially
in high-yield ETFs.

**Source(s).** Fulkerson, Jordan & Riley, *"Predictability in Bond ETF
Returns,"* *Journal of Fixed Income* 23(3) (2013) — ⚠ published in the
**Winter 2014 issue** (the DOI and working-paper date are 2013, but the
print issue is Winter 2014 — a minor but real correction); confirms large
historical return differences, ~0.96%/month long-short alpha in-sample.
Karmaziene & Terrada, *"Fast ETFs, Slow Bonds,"* *Finance Research
Letters* 90 (2026) — verified; rising yields widen discounts primarily in
high-yield ETFs; ETF prices react quickly while NAVs lag because
underlying bonds trade infrequently; discounts subsequently disappear.

⚠ **Important logical gap the prior draft missed:** "the discount
disappears" does **not** necessarily mean the ETF price rises to close
the gap. It may instead disappear because the *stale NAV* falls toward an
already-informed ETF price — i.e., the ETF may have been right all along
and the NAV is the one catching down, not up. A raw discount strategy
that assumes the ETF price will rise is not directly supported by this
mechanism as described.

**Recommended rewrite of the strategy:** estimate an independent fair-
value NAV using recent bond trades, Treasury moves, credit-index moves,
and ETF price discovery. Trade the residual deviation between ETF price
and estimated *contemporaneous* fair value — not the raw reported
premium/discount against a potentially stale official NAV. Before relying
on this, separately measure: subsequent ETF-price return, subsequent
reported-NAV change, the portion of convergence attributable to each, and
bid-ask/shorting costs.

**Data requirements.** Live premium/discount data is often free from
issuers and ETF portals; building a long, clean historical series is the
main friction; the fair-value redesign needs bond-trade/Treasury/credit-
index data too.

**How it differs from what's already been tried.** Fixed-income ETF
market microstructure.

**Evidence status:** Peer-reviewed (both). **Rule correspondence:**
Mechanism only, pending the fair-value-residual redesign — the raw-
discount rule as originally stated is **downgraded**. **Implementation
class:** Professional (fair-value estimation is nontrivial). **Validation
stage:** Literature only.

**Status: untriaged — build the fair-value-residual version, not the raw
discount rule.**

---

### B4. Crypto volatility-premium regime indicator

*(Prior draft: item 14, Tier 3, "Medium/Low" confidence, deprioritized.
New academic support found — promoted out of deprioritized status, but
still an indicator, not a complete strategy.)*

**Mechanism.** Compare Deribit DVOL to realized vol as an indicator of
options-market risk appetite; originally proposed as a complete tradable
rule ("short weekly straddles whenever DVOL exceeds trailing realized
vol").

**Source(s).** Deribit DVOL methodology (practitioner). ⚠ **New academic
support found:** *"Risk Premia in the Bitcoin Market,"* arXiv:2410.15195
— reports Bitcoin has a larger variance risk premium than the S&P 500,
with risk premia varying across volatility regimes. This supports the
broad insurance-premium mechanism but does **not** validate the specific
proposed DVOL-minus-realized-vol trading rule as written. The strategy
still needs to define: option tenor and strike selection, delta-hedging
frequency, whether realized vol is backward-looking or forecast,
skew/jump exposure, bid-ask spreads and settlement assumptions,
margin/collateral/liquidation/venue risk, and tail-loss limits.

**Data requirements.** Free — Deribit public API, CoinGlass/CoinGecko.

**How it differs from what's already been tried.** Delta-neutral, weekly
options-expiry horizon.

**Evidence status:** Academic mechanism now supported (Medium-High);
specific DVOL rule unsupported as a complete strategy (Medium-Low). **Rule
correspondence:** Mechanism only. **Implementation class:** Professional
(real options mechanics). **Validation stage:** Literature only. **Treat
DVOL-minus-realized as an indicator, not a complete tradable return
definition.**

**Status: untriaged — no longer deprioritized given the new citation, but
still needs full rule specification before this is buildable.**

---

# C. Portfolio-construction and research methods

These are not standalone alpha sources — they're methods for improving an
existing strategy set, and should be evaluated by whether they add value
out-of-sample to what you already have, not as independent trades.

### C1. Investment Base Pairs

*(Prior draft: item 5, Tier 1, "High" confidence — largely unchanged,
with an explicit note that the ETF adaptation is a new application, not
a replication.)*

**Mechanism.** Decompose cross-sectional signals into asset-pair
portfolios; keep only pairs with strong own- and cross-asset explanatory
power.

**Source(s).** Christian Goulding & Campbell Harvey, *"Investment Base
Pairs,"* SSRN 5193565 (2025) — verified, high-pedigree.

**Why it matters here.** A genuine methodology critique of conventional
quantile/linear-weighting portfolio construction, not a fragile anomaly
claim.

**Data requirements.** The 1,710-pair academic search is futures/
forwards-heavy; an ETF adaptation is materially different from the
original study and needs its own nested selection, untouched holdout, and
multiple-testing controls — it is not a direct replication.

**How it differs from what's already been tried.** Not "more momentum" or
"another stock ranking" — a multi-asset portfolio-construction hypothesis.

**Evidence status:** Peer-reviewed, high-pedigree. **Rule
correspondence:** Direct for the futures study; the ETF adaptation is a
new, separately-untested application. **Implementation class:**
Professional (full model) / Retail (approximate ETF version). **Validation
stage:** Literature only.

**Status: untriaged.**

---

### C2. QuantEvolve-style automated strategy discovery

*(Prior draft: item 25, Tier 4, "High" confidence on citation. Reclassified
as a method, not a strategy — matches its own original framing.)*

**Mechanism.** Constrained evolutionary/multi-agent search for
interpretable rules across a multi-asset panel, gated by walk-forward and
deflated-Sharpe validation.

**Source(s).** Yun, Lee & Jeon, *"QuantEvolve,"* arXiv:2510.18569 (2025) —
verified, Qraft Technologies, accepted at an ACM ICAIF-affiliated
workshop.

**Why it matters here.** A hypothesis-generation framework, not a
strategy — treat it as research infrastructure. Cap the search budget,
maintain an immutable final holdout, record every attempted rule, and
report multiple-testing-adjusted results if used.

**Data requirements.** Broad multi-asset data plus a strict validation
pipeline.

**How it differs from what's already been tried.** A discovery framework,
not a single hand-built signal.

**Evidence status:** Peer-reviewed-quality (workshop-accepted). **Rule
correspondence:** N/A — this is a method, not a rule. **Implementation
class:** Research infrastructure. **Validation stage:** Literature only.

**Status: untriaged — lowest build priority given the effort to stand up
an evolutionary search pipeline; a method to keep in mind, not a queued
candidate.**

---

# D. Parked hypotheses — cheap falsification only, not research priorities

Each of these has a plausible mechanism but thin, unverified, or off-topic
sourcing. Budget a fast, cheap, frozen-rule test for each rather than
committing real research time; do not iterate specifications after seeing
results. **All marked `deprioritized` in the sense used throughout this
file — not tested, not worth committing real research time to without a
stronger source or a tightly-scoped falsification test.**

- **Crypto session directionality** *(prior draft: item 10, Tier 3, "High"
  confidence — downgraded here).* Wątorek, Skupień, Kwapień & Drożdż
  (*Chaos*, 2023; arXiv:2306.17095, verified real) documents recurring
  crypto *activity* patterns (volume, volatility, session concentration,
  macro-announcement bursts) — it does **not** establish that expected
  returns during these windows have a stable sign, or that a session-
  specific mean-reversion or trend rule is profitable after costs. ⚠
  Downgrade from the prior draft's "single best citation-to-claim match"
  framing — the citation is genuinely strong for *activity patterns*, but
  that's a different claim than *return predictability*. Rename:
  "Session-conditioned crypto return predictability — hypothesis
  motivated by recurring activity patterns." Pre-register exact sessions
  and whether the test is trend, reversal, breakout, or vol-timing before
  running it. **Status: deprioritized (downgraded from active queue).**
- **G10 currency three-factor core** *(prior draft: item 18, Tier 4,
  deprioritized as "fabricated citation" — that verdict was itself
  wrong; see the major correction below).* ⚠ **Major correction:** the
  prior draft stated this citation was fabricated. That was wrong. A real
  paper exists: Sining Liu (Soochow University), Yan Wang (City
  University of Macau) & Lingxiao Zhao (Peking University), *"Survival of
  the Fittest: A Three-Factor Model in the Currency Market,"* Working
  Paper 20260104 (January 2026) — verified real, scheduled for
  presentation at AsianFA 2026 (July 2026). **However, the title was
  slightly misquoted ("Core" vs. "Model") and, more importantly, the
  paper does not support a carry-momentum-value model at all.** Using a
  Bayesian model-comparison framework, it finds a parsimonious **dollar
  (DOL) + carry (CAR) + business-cycle/output-gap (GAP)** three-factor
  model dominates the currency-factor model space — a completely
  different factor structure. Split into two separate candidates:
  **18A, a DOL-CAR-GAP model** (now genuinely supported by this paper,
  status: `untriaged`, promoted out of deprioritized), and **18B, a
  carry-momentum-value FX portfolio** (which needs a different citation
  — e.g., Menkhoff, Sarno, Schmeling & Schrimpf on currency momentum, or
  the AQR "Value and Momentum Everywhere" dataset, neither independently
  re-verified here; status: `deprioritized`, still needs a real
  citation). Do not use the new paper as evidence for momentum or value
  specifically. Keep the HMM regime filter as a separate, untested
  hypothesis regardless of which factor model is used.
- **Ex-US country/factor/hedge rotation** *(prior draft: item 19, Tier 4,
  deprioritized).* Decompose into three separate layers — country
  momentum, within-country factor rotation, FX hedging — and test each
  independently before stacking. The Bräuer paper (A6 above) is evidence
  for inferred FX expectations specifically, not for country momentum or
  factor rotation. **Status: deprioritized**, same underlying-citation
  caution as A6, but here it's load-bearing for the whole idea rather
  than one input among several.
- **Treasury HMM regime rotation** *(prior draft: item 20, Tier 4,
  deprioritized — unchanged, additional methodology requirements
  spelled out).* Needs real-time macro vintages (ALFRED-style, not final
  revised data), expanding-window estimation, no full-sample scaling,
  stable state-label rules, and explicit handling of state-probability
  uncertainty before it's a fair test. Currently rests on a GitHub repo
  and unverified arXiv/SSRN IDs. **Status: deprioritized** — A5 is the
  better-sourced static alternative.
- **Commodity seasonality (harvest/planting cycles)** *(prior draft: item
  21, Tier 4, deprioritized — off-topic citation confirmed, unchanged).*
  The previously cited paper (PMC11305200) is confirmed off-topic — it
  studies commodity/policy-uncertainty co-movement, not planting/harvest
  cycles. Remove until a genuine, contract-specific agricultural-
  seasonality paper is sourced; note ETF proxies may blur contract-month
  and harvest-cycle effects even once sourced. **Status: deprioritized.**
- **ETH/BTC ratio mean reversion** *(prior draft: item 22, Tier 4,
  deprioritized — unchanged, methodology requirements spelled out).*
  Test stationarity, cointegration stability, half-life stability, and
  structural breaks *before* adding DXY/VIX/liquidity filters — each
  added filter is another researcher degree of freedom on an
  all-practitioner-blog evidence base. **Status: deprioritized.**
- **Energy pre-holiday seasonal sleeve** *(prior draft: item 23, Tier 4,
  deprioritized — unchanged).* Few independent holidays in any sample,
  overlapping seasonal effects, high specification flexibility. Run one
  frozen falsification test; do not iterate windows after seeing results.
  Self-flagged by its own original source as non-academic. **Status:
  deprioritized.**
- **Stablecoin credit, liquidity, and technology carry** *(prior draft:
  item 15, "stablecoin yield arbitrage," Tier 3, deprioritized — renamed
  to reflect real risk).* ⚠ Renamed from "stablecoin yield spreads" — it
  is not riskless arbitrage and is not near-zero-risk merely because USD
  price exposure is matched. Include depeg, protocol, bridge, oracle,
  governance, liquidation, withdrawal, and reward-token risk explicitly.
  **Status: deprioritized.**
- **Spot-perpetual/calendar basis** *(prior draft: item 16, Tier 3,
  deprioritized — split into two).* Split into two distinct strategies:
  spot-vs-dated-futures cash-and-carry, and perpetual-vs-dated-futures
  relative value. Funding-rate paths, margin currency, venue default
  risk, collateral haircuts, and liquidation mechanics differ materially
  between the two. **Status: deprioritized (both halves).** See relation
  note above re: infeasible lab candidate #4.
- **Token unlock shorts** *(prior draft: item 17, Tier 3, deprioritized —
  unchanged, more nuance added).* Scheduled unlock amount is not the same
  as immediately sellable float — record recipient type, vesting
  structure, circulating-supply effect, exchange inflows, borrow
  availability, funding rate, and prior announcement date before treating
  this as more than a cheap event study. **Status: deprioritized.**
- **Concentrated-liquidity LP + funding harvest** *(prior draft: item 24,
  Tier 4, "Medium" confidence, kept active — reclassified here as an
  engineering experiment, not a strategy).* Not fully delta-neutral in
  practice (nonlinear inventory from concentrated ranges, gamma/
  impermanent-loss, adverse selection, rebalancing, MEV, gas, oracle, and
  smart-contract exposure all remain). **Status: untriaged as an
  engineering experiment**, not scored on the same axis as a strategy
  candidate.

---

## Where to spend research time first (per the independent review)

**For a home-lab / mostly-ETF workflow — this project's own default:**
1. Dynamic FX hedge overlay (A1)
2. Pre-FOMC Treasury drift replication (A7)
3. Commodity trend (A3)
4. Simplified Treasury curve relative value (A5), clearly labeled an
   approximation
5. LLM headline drift (A8), only with credible point-in-time data
6. Stablecoin stress as a risk overlay (B1)
7. FOMC country-ETF residual-continuation test (B2)

*CIP should not rank first here — FX ETFs do not directly implement the
documented arbitrage (see A9). This reverses the prior draft's #1
ranking.*

**For a professional futures/forwards/options workflow (not this
project's current setup, noted for completeness):**
1. Dynamic FX hedging (A1)
2. Commodity carry and trend as separate sleeves (A2, A3)
3. Treasury term-structure factors, full model (A5)
4. Short-term basis reversal replication (A4)
5. ETF-implied FX expectations (A6)
6. Pre-FOMC Treasury drift (A7)
7. Bitcoin options variance-risk-premium research (B4)
8. Investment Base Pairs as an overlay to established signals (C1)

**For an institutional FX/rates desk (not applicable here):** cross-
currency basis (A9) can rank near the top, since the desk plausibly has
the forwards, swaps, funding, collateral, and balance-sheet access to
actually express the anomaly directly — but even there, keep harvesting-
the-basis-itself distinct from using-the-basis-to-tilt-a-carry-portfolio.

## Minimum validation standard before committing capital to any candidate

Applies on top of, not instead of, this lab's own WFO/NDH/DSR gate
framework:

1. **Freeze the rule before testing** — instruments, timestamps, signal
   lags, rebalance rules, missing-data handling, leverage, entry/exit
   rules, risk limits, all specified in advance.
2. **Use point-in-time data** — futures rolls, ETF holdings, NAVs, macro
   releases, security universes, and event calendars must reflect what
   was actually observable at each historical decision point.
3. **Model the full implementation cost** — spreads, commissions, market
   impact, rolls, financing, borrow, margin, collateral yield, funding
   rates, gas/on-chain costs where applicable, forced-liquidation
   behavior.
4. **Use nested out-of-sample testing** — parameter selection inside
   training windows only; keep a genuinely untouched final holdout,
   especially for Base Pairs, LLM news drift, token unlocks, ETH/BTC mean
   reversion, pre-holiday effects, and any HMM regime model.
5. **Correct for multiple testing** — 25+ candidates across this batch
   alone, many with alternative windows/filters/thresholds, create real
   false-discovery risk. Report deflated Sharpe ratios, a probability-of-
   backtest-overfitting diagnostic, the number of specifications actually
   attempted, and the frozen rule's performance in an untouched sample.
6. **Measure stress behavior, not just full-sample correlation** — equity
   beta, duration beta, dollar beta, liquidity beta, crisis-period
   correlation, tail dependence, max drawdown, and drawdown overlap with
   the existing portfolio. A strategy can look diversified in calm
   conditions and become sharply equity-sensitive during funding stress —
   which is exactly why "low equity correlation" has been removed from
   this register's language until stress-period analysis actually
   demonstrates it.
7. **Run a shadow portfolio** — a sustained paper or low-capital live test
   surfaces timestamp errors, stale NAVs, unavailable borrow, venue
   outages, funding surprises, real slippage, execution delays, data
   revisions, and other operational failure modes no backtest catches.

## Bottom line

This batch is a strong **research-candidate list**, not yet a list of
validated strategies. Nothing here should be read as "verified" beyond
the specific, narrow claim that a cited paper exists and says
approximately what's attributed to it — exact-rule evidence, cost-aware
backtesting, and live validation are separate, mostly unstarted stages.
The defensible path:

> **paper → mechanism → exact rule → replication → shadow portfolio → capital**

not collapsing those stages into a single word like "validated."
