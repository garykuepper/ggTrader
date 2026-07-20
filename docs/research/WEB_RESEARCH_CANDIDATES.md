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

## 2026-07-19 batch — Master Consolidated Cross-Asset Strategy Report (25 candidates)

Source: web-research pass explicitly scoped to non-US-equity asset classes
(FX, commodities, Treasuries/rates, crypto), per the discovery-pivot
recommendation added to `web-strategy-research-prompt.md` after the
2026-07-16 batch's 9 equity-diversification-sleeve candidates all failed
(see `RESEARCH_SNAPSHOT.md` §4/§6). Merges three separate research passes
(8 + 8 + 12 ideas = 28 before dedup) into 25 distinct candidates.

**This supersedes an earlier draft of the same underlying research** that
was pasted in non-compliant format (missing the required per-candidate
"how it differs from what's already been tried" field, tiering used as a
substitute for the fields rather than an addition to them — see
`web-strategy-research-prompt.md`'s "Required output format," which was
tightened specifically because of that gap). This version follows the
correct format: every candidate has all 7 required fields (Name,
Mechanism, Source(s), Confidence, Why it's plausible, Data requirements,
How it differs), organized into the same Tier 1-4 structure. Content below
replaces the prior lettered (A-X) draft entirely — numbering is now 1-25.
One dedup improvement over the prior draft: the two duplicate HMM-regime
Treasury-rotation entries (previously "C-ii" and "S") are now a single
entry (20).

No overlap found with any of the 2026-07-16 batch's 14 candidates or with
this lab's tried/rejected roster — every idea here is a genuinely
different asset class from everything tested so far.

**Two soft relations to flag (not duplicates, different mechanism):**
- **Items 13 (cross-asset carry rotation) and 16 (crypto spot-perp
  calendar basis)** are adjacent to, but distinct from, this lab's
  already-infeasible candidate #4 (perpetual-futures *funding-rate*
  carry, rejected 2026-07-20 — Kraken retains only ~1yr of funding
  history). These trade the *futures-to-spot basis* (quarterly-expiry
  convergence) or *rate differentials across assets*, not the funding
  rate itself — worth re-checking data depth independently rather than
  assuming the same infeasibility carries over.
- **Item 12 (headline/LLM sentiment)** is adjacent to, but distinct from,
  the paused candidate #8 (Google Trends retail-attention). Item 12 uses
  NLP sentiment scoring of news text; #8 uses raw search-volume spikes.
  Different data source, different mechanism — no redundancy, just the
  same broad "alternative/behavioral data" category.

**Status: `untriaged` for 15 of 25** (see per-item status below and the
triage table) — **10 are `deprioritized`**: low-confidence sourcing
(unverified, fabricated, off-topic, or blog-only citations) that isn't
worth building on without a materially better source than what's already
been checked here. None of the 10 were tried/tested — deprioritized means
"looked at, weakly sourced, not worth research time," not NO-GO. Recommended
triage order for the untriaged 15 follows the report's own tiers below.

---

## Tier 1 — Best supported: verified, high-pedigree, low equity correlation

### 1. Cross-currency basis / CIP deviation harvesting

**Mechanism.** Tilt G10 carry positions using the cross-currency basis
swap spread as a signal for dealer balance-sheet stress: when the basis
for a currency widens beyond a threshold, tilt carry weight toward that
currency's funding side, since post-Basel III capital constraints prevent
the basis from arbitraging fully to zero. Combine with existing
carry/momentum signals rather than trading standalone.

**Source(s).** Wenxin Du, Alexander Tepper & Adrien Verdelhan,
*"Deviations from Covered Interest Rate Parity,"* *Journal of Finance* 73
(2018): 915–957 — the foundational paper; verified real, extremely
well-established. Mai Dao, Pierre-Olivier Gourinchas & Oleg Itskhoki,
*"Breaking Parity: Equilibrium Exchange Rates and Currency Premia,"* NBER
WP 34443 / IMF WP 2025/153 (2025) — verified real, high-pedigree
(IMF/Berkeley/Harvard-NBER), independently confirms the same
balance-sheet-constraint mechanism via a newer unifying model of covered
and uncovered currency premia.

⚠ *A third citation elsewhere in the source material — "Du, Tepper,
Verdelhan (2025), arXiv:2605.20137" — misattributes a real paper. That
arXiv ID is solo-authored by Useong Shin (posted May 2026); it builds on
but was not written by Du, Tepper & Verdelhan. Cite it separately under
Shin's name if used.*

**Confidence: High.** Two independently verified, high-pedigree papers
support this from different angles (2018 foundational result, 2025
unifying model) — the single best-supported idea across all three
reports.

**Why it's plausible.** CIP deviations persist because dealer-bank
balance-sheet constraints (regulatory capital, leverage ratio,
risk-weighted assets) limit arbitrage capital; the basis is effectively
the shadow price of balance sheet, widening when dealers are constrained.
This is mainstream, high-quality international finance research, not a
fringe claim.

**Data requirements.** Free — FRED cross-currency basis series (EUR, JPY,
GBP, CAD, AUD, CHF), FX ETFs for implementation.

**How it differs from what's already been tried.** A microstructure/
balance-sheet anomaly, not a macro factor or technical signal — funding-
cost alpha rather than price prediction, exploiting post-GFC regulatory
constraints on dealer intermediation. Distinct asset class from every
prior candidate tried in this lab.

**Status: untriaged.**

---

### 2. Commodity carry + trend + short-term basis reversal

**Mechanism.** Combine three independent commodity signals at the
portfolio level: (1) term-structure carry/roll-yield — long backwardation,
short/flat contango; (2) trend/cross-sectional momentum with a
volatility-regime filter to avoid crash periods; (3) short-term (weekly)
mean-reversion in the front-minus-second-month futures basis. Equal-risk-
weight the three sleeves.

**Source(s).** Koijen, Moskowitz, Pedersen & Vrugt, *"Carry,"* *Journal of
Financial Economics* 127 (2018): 197–225 — foundational carry paper,
verified real, covers commodities among other asset classes. Bloomberg
Professional Services, *"Capturing curve, carry and trend premia in
commodity markets"* (Feb 2026) — verified real practitioner confirmation
(product commentary, not independent research, but useful corroboration).
Alberto Rossi, Yingguang Zhang & Yandi Zhu, *"Short-Term Basis Reversal,"*
SSRN 5250499 (May 2025) — verified real and unusually well-corroborated by
independent practitioner replication (Quantitativo, QuantSeeker, and CXO
Advisory all discuss/replicate it). Documents weekly negative
autocorrelation in the front/second-month basis, pre-cost Sharpe ratios
above 1, present in commodities, equity index futures, and bonds.

**Confidence: High.** Three independently verified sources across two
separate reports — the strongest thematic overlap between any two reports
in the whole pool, and the Rossi/Zhang/Zhu basis-reversal result in
particular is a genuinely new, well-corroborated anomaly.

**Why it's plausible.** Carry compensates for hedging pressure and
convenience-yield effects; trend captures slow price adjustment to
supply/demand shocks; short-term basis reversal reflects differential
price sensitivity to news across the futures curve. Three economically
distinct, low-correlated drivers rather than one signal repeated three
ways.

**Data requirements.** Free/cheap — ETF proxies (PDBC, GCC, DBC, GSG,
COMT) for carry/trend legs; continuous futures data improves the
basis-reversal leg; CFTC COT data (free) for positioning context.

**How it differs from what's already been tried.** Different asset class,
different structural drivers (storage/convenience yield, physical supply
constraints), three independent signal categories rather than a single
equity cross-sectional sort.

**Status: untriaged.**

---

### 3. FOMC-window spillover trading in country ETFs

**Mechanism.** Trade U.S.-listed country ETFs representing markets closed
during the FOMC announcement window. Clean version: classify the Fed
surprise as hawkish/dovish, trade affected country ETFs during the U.S.
session, exit after the foreign market's next opening gap is incorporated.
Retail variant: use the ETF's own first 15–30 minute reaction as the
signal, hold until the next local market open.

**Source(s).** Fatih Kansoy (University of Oxford), *"The Immediate
Global Impact of US Monetary Policy"* (SSRN 5871422, Dec 2025) — verified
real and current. Confirms U.S.-traded country ETFs provide real-time
price discovery for foreign markets during FOMC windows; a geographic
discontinuity test shows ETF returns predict overnight local-index gaps
only for markets closed during the announcement; a typical contractionary
surprise destroys ~$280 billion in foreign equity value within 30
minutes, across 37 countries. Andreas Neuhierl & Michael Weber, *"Monetary
Momentum,"* NBER WP 24748 (2018) — verified real, well-established;
documents broad international FOMC-related return drift as a
complementary (not identical) mechanism.

**Confidence: High.** Both sources independently verified; Kansoy's paper
is a strong, recent, directly on-point primary source.

**Why it's plausible.** The structural edge is time-zone mismatch: when a
foreign cash market is shut but its U.S.-listed ETF is open, price
discovery temporarily migrates to the ETF, and daily-index approaches are
badly contaminated by intervening news relative to a clean 30-minute ETF
window.

**Data requirements.** Cheap/free — FOMC calendar, liquid intraday
country-ETF data from most brokers; the paper's exact orthogonalized
surprise measure needs rate-derivative data, but a reaction-based
implementation can sidestep this.

**How it differs from what's already been tried.** An ex-U.S.
price-discovery effect created by asynchronous market hours around
scheduled macro events — not a U.S. equity cross-sectional signal or
technical overlay.

**Status: untriaged.**

---

### 4. Dynamic FX hedge overlay (carry + value + trend)

**Mechanism.** Vary the hedge ratio on international equity/bond exposure
using three slow-moving signals — FX carry, PPP-style value, and
medium-term trend — instead of a static hedged/unhedged policy. Retail
version: a rules-based hedge score toggling between hedged/unhedged ETFs,
or a standalone G10 FX sleeve.

**Source(s).** Pedro Castro, Carl Hamill, John Harber, Campbell R. Harvey
& Otto Van Hemert, *"The Best Strategies for FX Hedging,"* *Journal of
Portfolio Management* 51(9): 37–78 (2025); SSRN 5047797 — verified real,
now published (not just a working paper), high-pedigree authorship (Man
Group researchers with Campbell Harvey as academic advisor). Confirms
incorporating trend, value, and carry into hedging decisions delivers
significant portfolio benefits over static policies across 20 currencies
since 1973.

**Confidence: High.** Cleanly verified, published, high-pedigree source,
accurately represented.

**Why it's plausible.** Currency exposure from international assets isn't
random noise: carry monetizes rate differentials, value exploits long-run
PPP mean reversion, trend captures persistent macro repricing.

**Data requirements.** Free/cheap and retail-friendly — spot FX, short-
rate differentials or policy-rate proxies, inflation/CPI data. Executable
with plain ETFs, spot FX, or hedged/unhedged ETF switching.

**How it differs from what's already been tried.** A separate
currency/hedging sleeve, not another attempt to refine U.S. equity entries
or a blunt equity risk filter.

**Status: untriaged.**

---

### 5. Cross-asset base-pair selection (portfolio construction)

**Mechanism.** Instead of applying value/carry/momentum one asset at a
time, decompose the opportunity set into asset pairs and keep only pairs
with historically strong own-asset and cross-asset explanatory power,
discarding "junk" pairs. Retail adaptation: a compact ETF universe across
equities, Treasuries, commodities, gold, and major FX, testing whether
"keep only the best pairs" improves diversification versus naïve
cross-asset sleeves.

**Source(s).** Christian L. Goulding & Campbell R. Harvey, *"Investment
Base Pairs"* (SSRN 5193565, March 2025) — verified real, high-pedigree
(Duke/NBER). Using 1,710 futures pair portfolios across equities, bonds,
currencies, and commodities, targeting top pairs and discarding junk pairs
roughly triples average annualized returns at fixed leverage over 20 years
(aggregate portfolio: 3.4% → 10.4%). Own-asset effects contribute 37–51%
and cross-asset effects 32–49% of performance across most signal/
asset-class groups.

**Confidence: High.** Verified, high-pedigree, and accurately represented
— one of the two or three best-supported ideas in the entire pool.

**Why it's plausible.** The core claim isn't "momentum/carry still
works" but that conventional portfolio construction (quantile sorts,
linear weighting) discards valuable cross-asset information — a
structural methodology critique rather than a fragile anomaly claim.

**Data requirements.** The full academic version is futures/forwards-
heavy; a reduced ETF-proxy version is home-lab testable but only an
approximation.

**How it differs from what's already been tried.** Not "more momentum" or
"another stock ranking" — a multi-asset portfolio-construction hypothesis
about cross-asset information and pair selection.

**Status: untriaged.**

---

### 6. Treasury term-structure factors (static version)

**Mechanism.** Build signals on Treasury ETFs (SHY/IEF/TLT) or futures
across duration buckets: steepener/flattener positioning, relative value
across intermediate vs. long duration, or a carry/rolldown proxy based on
investable, non-parametric term-structure factors.

**Source(s).** Filipović, Pelger & Ye, *"Shrinking the Term Structure,"*
NBER WP 32472 (2024); SSRN 4182649 — verified real, high-pedigree
(EPFL/Swiss Finance Institute, Stanford). Proposes investable
term-structure factors; four factors explain time-series variation and
risk premia of Treasury excess returns, including a state-dependent
"complexity premium" that pays off in recessions.

**Confidence: High.** Cleanly verified, high-pedigree, correctly
attributed (this citation was previously conflated with the separate
Adrian-Crump-Moench Treasury model in an earlier draft — now corrected to
attribute the paper only to its actual authors).

**Why it's plausible.** The term structure is driven by monetary-policy
expectations, inflation risk, and duration-risk appetite — slow-moving
drivers supporting medium-horizon signals, in a market too deep to be
fully trivialized when worked across curve segments.

**Data requirements.** Free — Treasury ETF data; a stronger implementation
uses yield-curve or futures data with careful roll handling.

**How it differs from what's already been tried.** Moves entirely out of
equities, using curve/term-structure signals rather than price-only
timing or cross-sectional stock sorting.

**Status: untriaged.** *Note: a related but distinct macro-regime-
conditional duration-rotation approach (HMM classifying growth/inflation
regimes, mapping to duration tilts) also surfaced in the source material
— see item 20 (Tier 4, deprioritized). Prototype this static-factor
version first.*

---

## Tier 2 — Verified citations with a specific attached caveat

### 7. Bond ETF discount-to-NAV reversal (credit/high-yield)

**Mechanism.** Buy bond ETFs at unusually wide discounts to NAV, avoid or
short unusually rich premiums, hold over the short horizon during which
the discount normalizes. Higher-conviction version concentrates on
corporate/high-yield ETFs, where the underlying bond market is illiquid
and NAVs lag price, especially during rate shocks.

**Source(s).** Jon A. Fulkerson, Susan D. Jordan & Timothy B. Riley,
*"Predictability in Bond ETF Returns,"* *Journal of Fixed Income* 23(3):
50 (2013) — verified real; large discounts followed by materially higher
subsequent returns than large premiums, long-short alpha ~0.96%/month
(~11.5%/year) in-sample. Egle Karmaziene & Juan M. Terrada, *"Fast ETFs,
Slow Bonds: Price Adjustment under Monetary Tightening,"* *Finance
Research Letters* 90 (2026) — verified real and current; confirms the
same mechanism was active during the 2022–2023 tightening cycle
specifically in high-yield ETFs, with discounts mean-reverting quickly.

**Confidence: High** on both citations, **with a dating caveat**: treat
the 2013 paper's 11.5%/year alpha figure as dated in-sample evidence; the
2026 paper is the more decision-relevant confirmation the mechanism
remains live.

**Why it's plausible.** A classic liquidity mismatch: bond ETFs trade
continuously, many underlying corporate bonds don't, so the ETF can
reprice faster than stale/matrix-priced bond marks.

**Data requirements.** Live premium/discount data is often free from
issuers and ETF portals; building a long, clean historical series is the
main friction.

**How it differs from what's already been tried.** Fixed-income ETF
market microstructure — not a cross-sectional equity sort, stock factor
sleeve, or technical-timing rule on familiar equities.

**Status: untriaged.**

---

### 8. Stablecoin-stress signal for BTC/ETH (jump-risk hedge)

**Mechanism.** Use major stablecoin stress (depeg, implied
collateral/settlement stress) as a crypto plumbing signal: reduce crypto
beta immediately or run a short-horizon downside trade in BTC/ETH on
detection. A slower overlay distinguishes "adoption" from "risk-premium
compression" using stablecoin balance growth.

**Source(s).** Baptiste Perez Riaza & Jean-Yves Gnabo, *"From depegs to
jumps: The role of stablecoin instabilities in crypto market dynamics,"*
*Journal of International Money and Finance* 155 (2025) — verified real.
A Tether depeg raises BTC/USD jump probability nearly fivefold within 5
minutes (co-jump probability 6.5x), based on high-frequency data across
70 crypto-assets. Yiming Ma, Yao Zeng & Anthony Lee Zhang, *"Stablecoin
Runs and the Centralization of Arbitrage,"* NBER WP 33882 (2025), now in
the *Journal of Finance* — verified real, top-tier venue. Confirms
stablecoin arbitrage/redemption is highly concentrated (Tether allows ~6
agents/month to redeem for cash) — the structural reason pegs can become
stress-transmission channels.

**Confidence: High** for these two verified, top-venue papers.
**Caveat:** a third supporting citation (Adams/Ibert/Liao on crypto
pricing) remains unverified.

**Why it's plausible.** Stablecoins function as settlement rails,
collateral, and working capital for crypto — closer to market plumbing
than a side sentiment indicator, so instability transmits directly into
jump risk via leverage and balance-sheet capacity across venues.

**Data requirements.** Free — recent exchange-level minute bars via free
exchange APIs or CCXT-style pulls; cross-venue data quality (especially
around delistings) is the main practical risk.

**How it differs from what's already been tried.** A crypto-specific
collateral/settlement stress signal tied to market structure — neither
perpetual-funding carry nor generic technical crypto timing.

**Status: untriaged.**

---

### 9. Currency-hedge-flow signal for international equities/FX

**Mechanism.** Two related framings surfaced: (a) track paired
international ETFs holding identical foreign exposure but differing by
currency-hedge status, reading persistent capital flow toward the hedged
or unhedged class as a revealed-preference FX signal; (b) layer this same
signal into a broader ex-US country/factor rotation with a dynamic hedge
overlay (see item 19).

**Source(s).** Leonie Bräuer, *"Exchange Rate Expectations and Currency
Demand"* (working paper, Oct–Dec 2025 drafts) — verified as a real,
current working paper (350 matched hedged/unhedged ETF pairs, 2014–2025,
presented at AFA and a central-bank workshop).

⚠ **Significant correction:** the source material describes this paper's
finding as showing portfolio-implied expectations extracted from
hedge/unhedged allocations *predict* future FX returns and *outperform*
survey expectations. Based on the paper's own text, the actual finding
appears closer to the opposite: ETF investors' hedge-ratio choices show no
significant sensitivity to model-implied expectations once controlling
for time trends, while **survey expectations are the ones that are highly
significant** in explaining those allocations. That's closer to "ETF
hedging flows are consistent with survey expectations" than to "ETF-
derived expectations beat surveys at forecasting FX returns." A separate
citation elsewhere in the source material (SSRN ID 6447259) for what
appears to be the same paper was not reconciled with the ID found via
direct verification (5047797 is a different, unrelated paper — the FX
hedging paper in item 4 above).

**Confidence: Low-Medium.** Real source, but re-read the paper directly
before building anything on the specific predictive claim — as currently
sourced, the claim appears to have the causality reversed.

**Why it's plausible (mechanism, independent of the citation issue).**
Extracting beliefs from capital-backed choices (hedge ratio selection) is
conceptually more grounded than survey-based sentiment, since it reflects
money actually at risk — even if this particular paper's own conclusion
runs the other direction.

**Data requirements.** Feasible but tedious: matched ETF pairs, consistent
AUM history, a clean "same exposure, different hedge status" mapping.

**How it differs from what's already been tried.** A revealed-preference
FX signal from investor hedging choices, not another U.S. equity
characteristic sort.

**Status: untriaged — re-confirm the paper's actual finding directly
before scoping any build**, per the correction above. Not marked
deprioritized since the underlying source is real and verified, just
possibly mischaracterized in the source material.

---

## Tier 3 — Plausible mechanisms, thin or unverified sourcing

### 10. Session-aware crypto intraday mean reversion/trend

**Mechanism.** Segment crypto returns by time-of-day/session; trade only
periods with persistent structure — fade flow imbalances around session
transitions, trend-follow around high-persistence blocks (Asian/European/
US sessions, macro-release windows).

**Source(s).** Wątorek, Skupień, Kwapień & Drożdż, *"Decomposing
cryptocurrency high-frequency price dynamics into recurring and noisy
components,"* *Chaos* 33, 083146 (2023); arXiv:2306.17095 — verified real.
Confirms three enhanced-activity phases aligned with Asian, European, and
U.S. sessions, plus recurring bursts around major U.S. macro releases —
matching the proposed mechanism closely.

**Confidence: High** on the citation — the single best citation-to-claim
match found across all three reports, though narrower in scope (a single
intraday signal) than the portfolio-level Tier 1 ideas.

**Why it's plausible.** Crypto trades 24/7 but liquidity/participation
still cycles with global sessions — a structural, not chart-pattern,
source of repeatable short-horizon behavior.

**Data requirements.** Intraday OHLCV; free for major pairs at
minute-to-hour resolution.

**How it differs from what's already been tried.** Different market and
holding-period horizon than the equity ensemble; not an overnight-gap or
leveraged-ETF rule.

**Status: untriaged.**

---

### 11. Pre-FOMC drift in long Treasuries (test-and-discard)

**Mechanism.** Long TLT/IEF/EDV the day before scheduled FOMC meetings,
exit at/around the announcement (or hold through the press conference if
drift extends). Explicitly framed as a fast test with a likely-decayed
post-2023 edge.

**Source(s).** The source material cites an unverified "AEA 2026 draft"
claiming possible post-2023 decay. **Better-anchored alternative:** the
already-verified Neuhierl & Weber "Monetary Momentum" paper (item 3 above)
independently documents this same broad FOMC-drift phenomenon with a
real, peer-reviewed citation.

**Confidence: Medium** — the underlying phenomenon is real (via the
better citation in item 3), but the specific "has it decayed post-2023"
claim is unverified. Cheap and fast to test directly (~30 minutes per the
source material), so this is well-postured regardless of citation
quality.

**Why it's plausible.** An uncertainty premium — dealers hedge gamma/vega
ahead of FOMC, bidding up long-duration convexity. If Fed communication
became more transparent post-2023, this premium may have compressed.

**Data requirements.** Free — Yahoo Finance TLT/IEF/EDV, FOMC calendar.

**How it differs from what's already been tried.** A pure event-driven
calendar anomaly (8–12 trades/year), low correlation to everything else in
the pool.

**Status: untriaged.**

---

### 12. Headline/LLM sentiment on small/mid-cap equities

**Mechanism.** Use an LLM or NLP classifier on news headlines to generate
a daily long/flat/short signal, trading the delayed response after
publication in smaller names where underreaction is more plausible.

**Source(s).** Lopez-Lira & Tang, *"Can ChatGPT Forecast Stock Price
Movements? Return Predictability and Large Language Models,"*
arXiv:2304.07619 (2023), forthcoming *Journal of Financial Economics* —
verified real. GPT-4 scores predict out-of-sample daily returns, stronger
in smaller stocks and after negative news. Saqur, Kato, Vinden & Rudzicz,
*"NIFTY Financial News Headlines Dataset,"* arXiv:2405.09747 (2024) —
verified real; a public, point-in-time-usable headline dataset.

**Confidence: High**, with a link fix applied — the original source
material had these two arXiv IDs cross-linked to the wrong papers; both
are now correctly attributed.

**Why it's plausible.** Attention constraints mean headlines are
processed slowly/inconsistently in smaller names, creating an
underreaction channel distinct from technical timing.

**Data requirements.** A clean, point-in-time headline feed mapped to
tickers; survivorship-safe historical coverage is the main feasibility
risk.

**How it differs from what's already been tried.** Not another
price-action or fundamental sort on the same large-cap universe. See
relation note above re: paused candidate #8 (Google Trends) — different
mechanism, no redundancy.

**Status: untriaged.**

---

### 13. Cross-asset carry rotation (risk-on/off via carry)

**Mechanism.** Rotate across liquid multi-asset ETFs (equities, bonds,
commodities, gold/FX) using a carry framework — favor positive
carry/roll-down/forward premium — rather than momentum, combined with a
volatility budget.

**Source(s).** Same AQR "Carry" paper as item 2 — legitimately reused, as
it explicitly documents carry premia across global equities, bonds,
currencies, commodities, and Treasuries.

**Confidence: High** on the citation — effectively the same evidentiary
base as item 2, applied more broadly.

**Why it's plausible.** Carry compensation is often more durable than a
pure directional forecast; a multi-asset sleeve diversifies further than
another single-asset-class anomaly.

**Data requirements.** ETF yields, distributions, and roll behavior —
relatively retail-feasible.

**How it differs from what's already been tried.** Shifts the edge search
into other liquid asset classes entirely, applied at the broadest
multi-asset level. See relation note above re: infeasible candidate #4 —
different mechanism (cross-asset carry, not perp funding rate).

**Status: untriaged.**

---

### 14. Crypto options volatility risk premium (DVOL vs. realized vol)

**Mechanism.** Harvest the volatility risk premium in BTC/ETH options:
compare Deribit's DVOL (model-free implied vol) to realized vol; short
delta-hedged straddles/strangles when DVOL exceeds realized vol by a
threshold, long vol in the reverse case, on a weekly options-expiry cycle.

**Source(s).** Deribit DVOL methodology and several practitioner pieces
(Delphi Digital, RegimeRisk, Harbourfront Quant) — **not independently
verified in this pass.** The underlying volatility-risk-premium mechanism
(implied systematically exceeding realized vol on average, delta-hedged
harvest) is standard, well-established options-market theory generally,
just not confirmed specifically for crypto via any citation checked here.

**Confidence: Medium** on mechanism (standard finance), **Low** on the
specific crypto-market citations, none independently checked.

**Why it's plausible.** Options market-makers and hedgers pay a
persistent premium for downside protection; in crypto, endogenous hedging
demand plus retail speculation plausibly creates a net vol risk premium
analogous to equity index options.

**Data requirements.** Free — Deribit public API, CoinGlass/CoinGecko.

**How it differs from what's already been tried.** Delta-neutral, weekly
options-expiry horizon — a distinct economic mechanism (insurance
provision) from any directional crypto strategy already tried.

**Status: deprioritized — low confidence, do not resurface without a
verified crypto-specific source.**

---

### 15. Stablecoin yield arbitrage (CeFi/DeFi rate differential)

**Mechanism.** Borrow the cheapest stablecoin on a lending protocol, lend/
deploy the highest-yielding stablecoin instrument, net the spread
delta-neutrally, capped by battle-tested-stablecoin selection and
per-protocol allocation limits.

**Source(s).** Industry reports (Galaxy Research, DeFiLlama Yields API,
Messari, Portals.fi) — none academic, none independently verified. The
underlying rate fragmentation across DeFi protocols is closer to
observable market structure than a contested empirical claim.

**Confidence: Medium** on mechanism, **Low** on specific cited figures.

**Why it's plausible.** Stablecoin lending markets are genuinely
fragmented across protocols/chains with heterogeneous borrower needs and
token-emission subsidies, creating a persistent, structural (not
price-prediction) spread.

**Data requirements.** Free — DeFiLlama Yields API.

**How it differs from what's already been tried.** Near-zero directional
risk, high-frequency rebalance, yield capture rather than price
prediction.

**Status: deprioritized — low confidence, do not resurface without
academic verification.**

---

### 16. Crypto spot-perp calendar basis trade

**Mechanism.** Trade the calendar spread between quarterly futures (CME,
Deribit, Binance) and perpetual swaps: short quarterly/long perp when the
annualized quarterly basis exceeds perp funding by a threshold, and vice
versa, holding to expiry or an early roll.

**Source(s).** CME Group and CME OpenMarkets commentary, a GitHub repo, a
practitioner cross-exchange stat-arb piece — none academic, none
independently verified. The underlying mechanism (fixed-expiry futures
must converge to spot; the basis reflects funding-expectation term
structure) is well-documented as real institutional practice, notably the
post-spot-ETF CME-basis trade.

**Confidence: Medium** on mechanism (well-known real institutional
practice), **Low** on the specific secondary sources cited.

**Why it's plausible.** Quarterly futures have a fixed expiry and must
converge to spot; perpetuals are anchored by funding instead — the basis
between the two reflects genuine term-structure and convenience-yield
information.

**Data requirements.** Free — CoinGecko/CoinGlass, CME public data.

**How it differs from what's already been tried.** Term-structure/
calendar-spread mechanism, delta-neutral, defined expiry horizon. See
relation note above re: infeasible candidate #4 — trades the futures-spot
basis, not the funding rate itself.

**Status: deprioritized — low confidence, do not resurface without a
verified source.**

---

### 17. Token unlock event-driven short

**Mechanism.** Monitor token unlock calendars; short ahead of large
scheduled cliff unlocks (anticipatory insider/VC selling pressure), cover
after the unlock and consider a mean-reversion long if price overshoots.

**Source(s).** Data-vendor calendars (DefiLlama, CoinGlass, Tokenomist.ai)
and a trading-guide write-up — no empirical academic study found showing
this specific edge exists in crypto, though the equity IPO-lockup-expiry
analogy is real, established literature.

**Confidence: Low-Medium** — reasonable mechanism, but no cited study
demonstrates the edge specifically in crypto; worth a cheap backtest
before trusting the narrative.

**Why it's plausible.** Cliff unlocks are predictable, scheduled supply
shocks with clear incentives for early/mechanical selling — publicly
known but plausibly behaviorally underreacted to, similar in spirit to
lockup expiry.

**Data requirements.** Free — the unlock calendars are the product.

**How it differs from what's already been tried.** A genuinely novel,
catalyst-driven, short-biased mechanism with a defined event window.

**Status: deprioritized — low confidence, do not resurface without an
empirical crypto-specific study.**

---

## Tier 4 — Weakest sourcing: fabricated, misattributed, off-topic, or purely anecdotal citations

### 18. G10 currency three-factor core (carry + momentum + value)

**Mechanism.** Equal-risk-weight carry, cross-sectional momentum, and
PPP/REER value across G10 currency ETFs, with volatility targeting and an
HMM-based regime filter to avoid carry-crash episodes.

**Source(s).** ⚠ **The primary citation appears fabricated.** "Survival of
the Fittest: A Three-Factor Core in the Currency Market" (claimed SSRN
6609879) could not be found anywhere under that title or ID — no matching
paper on SSRN, Google Scholar, RePEc, or general web search. This is a
different, more serious problem than the citation slips found elsewhere
(wrong link, conflated authors), where a real paper was always findable
nearby; here, nothing matches.

**Confidence: Low** on the citation specifically — **this was originally
ranked #1 with "most robust academic support"; that characterization does
not currently hold.** Replace the citation with a real paper before
allocating meaningful research time. **Medium** on the underlying
three-factor concept given the broader literature.

**Why it's plausible (mechanism, independent of the fabricated citation).**
The three-factor combination itself is a real, broadly supported concept
in FX literature generally — see item 4 above, and decades of published
carry/momentum/value FX research (e.g., Menkhoff, Sarno, Schmeling &
Schrimpf on currency momentum; the well-known AQR "Value and Momentum
Everywhere" dataset, not independently re-verified here but extremely
well-established).

**Data requirements.** Free — Yahoo Finance for G10 ETFs, FRED for basis
and PPP/REER data.

**How it differs from what's already been tried.** Different asset class,
different macro/funding-structure drivers, low correlation to equities —
this framing is sound regardless of the citation problem.

**Status: deprioritized — fabricated citation, do not resurface without a
real replacement source (e.g. Menkhoff/Sarno/Schmeling/Schrimpf or AQR's
"Value and Momentum Everywhere," both real alternatives already named
above).**

---

### 19. Country/factor rotation ex-US with dynamic currency hedge

**Mechanism.** Two-layer rotation: country momentum across developed
ex-US equity ETFs, plus factor rotation within countries, plus a dynamic
currency-hedge overlay using carry/momentum/PPP signals.

**Source(s).** Largely the same underlying signal and citation issue as
item 9 above (the Bräuer paper, cited here under a different, unreconciled
SSRN ID). AQR ex-US factor data and various practitioner pieces (Robeco,
Alpha Architect, iShares) were not independently re-verified.

**Confidence: Low-Medium** — carries forward the same caution as item 9
about the currency-hedge-signal claim specifically.

**Why it's plausible.** Combines two well-established, separately verified
ideas (factor investing in non-US markets + dynamic FX hedging from item
4) into a country-selection framework — reasonable in principle.

**Data requirements.** Free/cheap for ETF prices and macro data; hedged/
unhedged flow data is more manually assembled.

**How it differs from what's already been tried.** Different geography,
explicit currency factor layered on top of country rotation.

**Status: deprioritized — same citation caution as item 9, but here it's
load-bearing for the whole idea rather than one input among several; do
not resurface without directly re-confirming the Bräuer paper's actual
finding.**

---

### 20. Treasury duration rotation via HMM regime-switching

**Mechanism.** A 3-state Gaussian HMM on ~12 FRED macro series classifies
growth/inflation regimes, mapping each to a duration tilt (e.g., long
duration in slowing growth, short duration in overheating), with a
momentum confirmation filter and volatility targeting.

**Source(s).** A GitHub repo, an unverified arXiv paper, an unverified
SSRN paper on volatility scaling, and an MDPI *FinTech* paper — **none
independently verified in this pass.**

**Confidence: Low**, pending direct verification of every cited source —
prototype item 6 first.

**Why it's plausible (mechanism, independent of citations).** Duration is
a genuine first-order macro asset class with well-understood growth/
inflation sensitivities; regime-switching models for macro allocation are
a real, established technique broadly, even where these specific
citations are unconfirmed.

**Data requirements.** Free — Yahoo Finance, FRED, MOVE index.

**How it differs from what's already been tried.** Different asset class,
macro regime-conditional rather than a static factor sort — see item 6
for the better-sourced static alternative.

**Status: deprioritized — low confidence, do not resurface without
verified sourcing; item 6 is the better-supported version of this same
asset class.**

---

### 21. Commodity seasonality (harvest/planting cycles)

**Mechanism.** Trade agricultural commodity proxies using seasonal windows
tied to planting/growing/harvest calendars, conditioned on inventory/price
context.

**Source(s).** ⚠ The cited paper (PMC11305200, *Heliyon* 2024) is real but
studies wavelet-based co-movement between commodity prices and
economic-policy-uncertainty indices across crisis periods — it does not
address planting/harvest seasonality at all. Real paper, off-topic
relative to the claim.

**Confidence: Low** pending a genuine seasonality citation — source a
paper that actually studies planting/harvest-cycle effects before relying
on this.

**Why it's plausible (mechanism, independent of the citation).**
Agricultural commodities are tied to biological production cycles and
storage constraints, giving seasonality a genuine economic basis distinct
from arbitraged equity calendar effects.

**Data requirements.** Feasible with commodity ETFs; better with
contract-level futures data.

**How it differs from what's already been tried.** A supply-cycle anomaly
in a different market, not a large-cap equity or price-action signal.

**Status: deprioritized — low confidence, do not resurface without a
genuine on-topic citation.**

---

### 22. ETH/BTC ratio mean-reversion with macro regime filter

**Mechanism.** Z-score mean-reversion on the ETH/BTC ratio, gated by a
DXY/liquidity/VIX macro regime filter, beta-neutral long/short.

**Source(s).** All practitioner/blog sources (BacktestEverything,
Ecoinometrics, Acheron Trading, Galaxy) — no academic backing found, none
independently verified.

**Confidence: Low** — weakest source base among the higher-effort ideas;
treat as fast-test-and-discard.

**Why it's plausible.** The ETH-as-utility-asset vs. BTC-as-store-of-value
framing is a widely used industry narrative with real behavioral logic,
but rests entirely on commentary rather than any peer-reviewed source.

**Data requirements.** Free — CoinGecko/Yahoo Finance, FRED.

**How it differs from what's already been tried.** Market-neutral within
crypto, macro-conditioned relative value.

**Status: deprioritized — low confidence, do not resurface without
academic backing.**

---

### 23. Energy pre-holiday seasonal sleeve

**Mechanism.** Long energy exposure (USO/UGA) in the days before major
U.S. holidays, exploiting travel-driven fuel-demand anticipation.

**Source(s).** Quantpedia, *"Pre-Holiday Effect in Commodities"* (2024) —
explicitly self-flagged by its own original source as not a canonical
academic paper.

**Confidence: Low**, correctly self-labeled by its own source as the
thinnest idea in that report — good fast-falsification candidate given
how cheap it is to test.

**Why it's plausible.** A physical-demand story (travel-heavy holidays
create fuel pre-positioning), not pure pattern-mining.

**Data requirements.** Very light — exchange holiday calendar, daily ETF
prices.

**How it differs from what's already been tried.** A short-window
commodity seasonality sleeve tied to physical demand and calendar
structure.

**Status: deprioritized — low confidence, do not resurface without
academic backing.**

---

### 24. DeFi delta-neutral concentrated LP + funding harvest (long-shot)

**Mechanism.** Provide concentrated Uniswap v3 liquidity in stable-
volatile pairs, delta-hedged with a short perp position, earning both
concentrated-LP swap fees and perp funding.

**Source(s).** Industry pieces (Delphi Digital, Thrive.fi, Galaxy,
Hyperdash) — no academic backing, none independently verified.

**Confidence: Medium** on mechanism, but correctly flagged by its own
source as a long-shot for execution-complexity reasons rather than
citation reasons.

**Why it's plausible.** Concentrated-liquidity fee mechanics and
positive-average perp funding are both observable market facts, not
contested claims — closer to an engineering problem than an unproven
anomaly.

**Data requirements.** Free data, but operationally heavy: smart contract
deployment, keeper bots, multi-venue connectivity.

**How it differs from what's already been tried.** A different tech stack
entirely (on-chain DeFi primitive plus CeFi/DeFi perp funding).

**Status: untriaged — kept active despite Tier 4 grouping; the concern
here is engineering effort, not citation confidence, so it doesn't fit
the "weakly sourced, don't resurface" reason for deprioritizing the other
Tier 4 items.**

---

### 25. Automated multi-asset strategy discovery (QuantEvolve-style)

**Mechanism.** Use a constrained evolutionary/multi-agent search system to
discover simple, interpretable rules across a multi-asset panel, keeping
only strategies that survive strict walk-forward and deflated-Sharpe
validation.

**Source(s).** Yun, Lee & Jeon, *"QuantEvolve: Automating Quantitative
Strategy Discovery through Multi-Agent Evolutionary Framework,"*
arXiv:2510.18569 (2025) — verified real, authors from Qraft Technologies,
accepted for oral presentation at an ACM ICAIF-affiliated workshop (Oct
2025).

**Confidence: High** on the citation itself; correctly framed by its
original source as the most speculative *application* in that report
(a framework, not a validated strategy) even though the paper backing it
is solid.

**Why it's plausible.** A structured search system can surface
interactions humans wouldn't hand-engineer outside an already-saturated
hypothesis space; overfitting risk is severe and explicitly the main
caveat.

**Data requirements.** Broad multi-asset data plus a strict validation
pipeline.

**How it differs from what's already been tried.** A discovery framework,
not a single hand-built signal.

**Status: untriaged — citation is solid, but this is a meta-framework, not
a strategy; lowest priority to actually build given effort required to
stand up an evolutionary search pipeline.**

---

### Summary triage table (25 candidates)

| # | Candidate | Asset class | Data cost | Confidence | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | Cross-currency basis / CIP harvesting | FX | Free | High | untriaged | Best-supported idea overall |
| 2 | Commodity carry+trend+basis reversal | Commodities | Free/cheap | High | untriaged | 3 independent verified sources |
| 3 | FOMC country-ETF spillover | FX/equity (intl) | Free/cheap | High | untriaged | Strong recent primary source |
| 4 | Dynamic FX hedge overlay | FX | Free/cheap | High | untriaged | Published, high-pedigree |
| 5 | Cross-asset base-pair selection | Multi-asset | Futures-heavy (ETF-proxy feasible) | High | untriaged | Portfolio-construction insight |
| 6 | Treasury term-structure factors (static) | Rates | Free | High | untriaged | Prototype this variant first |
| 7 | Bond ETF discount-to-NAV reversal | Credit/HY bonds | Free | High | untriaged | Lean on 2026 paper, not 2013 figure |
| 8 | Stablecoin-stress jump-risk hedge | Crypto | Free/cheap | High | untriaged | 2 verified, high-tier papers |
| 9 | Currency-hedge-flow signal | FX/equity (intl) | Free/cheap | Low-Medium | untriaged | Re-confirm Bräuer paper's actual claim directly |
| 10 | Session-aware crypto intraday | Crypto | Free/cheap | High | untriaged | Small scope, cleanest citation match |
| 11 | Pre-FOMC Treasury drift | Rates | Free | Medium | untriaged | Cheap, fast to test regardless of citation |
| 12 | Headline/LLM sentiment (small/mid caps) | Equity (alt-data) | Free/cheap (LLM cost) | High | untriaged | See relation note re: #8 |
| 13 | Cross-asset carry rotation | Multi-asset | Free/cheap | High | untriaged | Shares item 2's evidentiary base |
| 14 | Crypto options vol risk premium | Crypto derivatives | Feasibility risk | Medium/Low | **deprioritized** | Standard mechanism, crypto sourcing thin |
| 15 | Stablecoin yield arbitrage | Crypto/DeFi | Free (industry reports) | Medium/Low | **deprioritized** | No academic verification |
| 16 | Crypto spot-perp calendar basis | Crypto derivatives | Free/cheap | Medium/Low | **deprioritized** | See relation note re: #4 |
| 17 | Token unlock event-driven short | Crypto | Free (calendars) | Low-Medium | **deprioritized** | Untested in crypto specifically |
| 18 | G10 currency 3-factor core | FX | Free/cheap | Low/Medium | **deprioritized** | ⚠ fabricated citation, real concept |
| 19 | Country/factor rotation ex-US + FX hedge | FX/equity (intl) | Free/cheap | Low-Medium | **deprioritized** | Same citation issue as #9, load-bearing here |
| 20 | Treasury HMM regime rotation | Rates | Free | Low | **deprioritized** | Item 6 is the better-sourced alternative |
| 21 | Commodity seasonality | Commodities | Free | Low | **deprioritized** | ⚠ cited paper is off-topic |
| 22 | ETH/BTC ratio mean-reversion | Crypto | Free | Low | **deprioritized** | Blog-only sourcing |
| 23 | Energy pre-holiday seasonal | Commodities | Free | Low | **deprioritized** | Self-flagged thin by its own source |
| 24 | DeFi delta-neutral LP + funding harvest | Crypto/DeFi | Feasibility risk (infra-heavy) | Medium | untriaged | Engineering long-shot, not citation risk |
| 25 | Automated multi-asset strategy discovery | Meta/framework | N/A | High (citation) | untriaged | Framework, not a strategy — lowest build priority |

**Where to spend research time first (per the source report):**
- **Tier 1 — start here (6 ideas, fully verified, genuinely
  diversifying):** items 1-6, in roughly that priority order.
- **Tier 2 — good second wave, verified with a fixable caveat:** items 7
  and 8 next; re-confirm item 9 directly before using it.
- **Tier 3 — plausible, worth cheap tests:** item 10 has the cleanest
  citation despite narrower scope; items 11 and 12 are solid once anchored
  to their corrected citations; item 13 shares item 2's evidentiary base.
- **Needs a citation fix before serious investment:** item 18 (replace
  the fabricated citation), item 9/19 (re-confirm the Bräuer paper's
  actual claim), item 1's misattributed arXiv ID (easy fix — correct
  author is Useong Shin).
- **Deprioritized (10 items: 14, 15, 16, 17, 18, 19, 20, 21, 22, 23) —
  do not resurface without a materially better source than what's already
  been checked.** None were tried/tested — they're weak on sourcing, not
  NO-GO. `web-strategy-research-prompt.md` names these specific niches so
  a future discovery pass doesn't waste output resurfacing them.
- **Kept active despite Tier 4 grouping (24, 25):** neither is
  citation-weak — 24's issue is engineering effort, 25 is a solid citation
  for a meta-framework rather than a strategy. Lowest build priority, not
  lowest confidence.
