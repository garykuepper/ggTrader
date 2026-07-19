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

### 4. Crypto perpetual-futures funding-rate carry (delta-neutral)
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

### 5. Informed liquidity-supplying short sellers ("stealthy shorts")
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

### 8. Retail-attention-conditioned factor anomalies
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

### 14. Option-implied volatility skew as a return predictor — feasibility-limited
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
