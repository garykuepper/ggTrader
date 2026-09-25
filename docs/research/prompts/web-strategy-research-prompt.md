# Prompt: Discover New Trading-Strategy Ideas (External Research Only)

**Portable prompt — paste this whole thing into a web-research tool (e.g.
Google Gemini's or Claude's research/web-search feature).** It assumes only
general web-browsing/search ability, nothing else — no code execution, no
file access, no specific tools or plugins. Do not write code, backtest
anything, or assume access to any codebase. Your job is research and
recommendation only; someone else will implement and test whatever you find.

Regenerate the "context" section below from `docs/research/RESEARCH_SNAPSHOT.md`
whenever it's materially out of date — last synced 2026-09-25.

---

## Your task

You are a quantitative-strategy research scout. Search broadly across
academic finance research, quantitative-trading blogs and newsletters,
practitioner write-ups, and relevant online communities for **genuinely new
trading-strategy ideas or market anomalies** — the kind of thing a skilled
independent/retail quant could plausibly build and test, not a purely
institutional strategy requiring infrastructure a home-lab setup can't
replicate.

Good sources to draw on (not exhaustive — use your judgment and cite
whatever you actually find, don't limit yourself to this list): academic
papers (SSRN, arXiv's q-fin category, NBER working papers), practitioner
research blogs (e.g. Alpha Architect, Two Sigma's and AQR's public research,
QuantConnect/Quantopian-style community writeups), reputable quant-finance
forums and communities, and recent (last 1-3 years, so it reflects a
still-relevant market regime) coverage of trading anomalies or novel
factor/signal research.

## Context: what's already been tried (don't just resurface these)

The system this feeds is a systematic US-equities (and, secondarily,
crypto) trading research lab. It already tests every candidate through a
rigorous walk-forward-optimization framework with overfitting-detection
gates before deploying anything, and only deploys strategies that beat a
buy-and-hold S&P 500 benchmark on a risk-adjusted (Sharpe) basis, not just
raw return.

**Currently deployed — and the honest bar:** a majority-vote ensemble of
five technical indicators (Bollinger Bands, RSI, EMA crossover, MACD
divergence, volume-confirmed Bollinger Bands) buying individual S&P 500
stocks on short-term mean-reversion, flat ~3% position sizing per trade.
Earlier tests showed it beating the index, but a stricter re-test that only
lets it trade stocks on days they were actually in the S&P 500 (earlier
tests had accidentally allowed trading stocks before they joined or after
they left) cut it to roughly **Sharpe ~0.6 and ~3% a year versus the
index's ~0.8 Sharpe and ~13% a year** over 2021–2026. Its one real virtue is
shallow drawdowns (about -5% versus the index's -22%). **So nothing in this
system currently beats simply holding the S&P 500.** The most useful ideas
now either beat the index on their own after costs, or combine with an
index holding to improve its risk-adjusted return or cut its drawdowns. The
account actually holds that stock-picking sleeve **plus the S&P 500 index
itself**, because idle cash is swept into an S&P 500 ETF. Measured that way,
the combined book roughly **ties** the index (Sharpe ~0.80 vs ~0.78) and has
slightly shallower drawdowns. So in practice the account is mostly an index
holding. **Any new idea is effectively funded by selling some of that index
holding, so it has to earn more than the index would have on the same
days.** A blend of the stock-picking strategy across three U.S. equity
indices was also tried live and dropped after it trailed the single-index
version. Treat any number
this scout is given (including from other projects) as provisional until
it's been through that kind of scrutiny — this system's own headline
number has been revised down more than once.

**Already tried and rejected — don't propose close variants of these
without a genuinely different mechanism:**
- A machine-learning classifier gating which technical-indicator signals to
  trade (proved anti-predictive/worthless across several redesigns).
- Weighting the ensemble's votes by each indicator's historical predictive
  skill, instead of equal-weighting (hurt risk-adjusted return; re-tested
  under the stricter membership rules and still worse).
- Position-sizing by the Kelly criterion or by per-trade "conviction"
  strength, instead of a flat size (both underperformed flat sizing).
- Take-profit / time-based exit rules layered on or replacing the current
  exit logic (worse than the existing rule-based exit).
- Cross-sectional or absolute price momentum ranking on large-cap U.S.
  equities (well-arbitraged, no edge found).
- A blunt volatility/VIX-level regime filter that blocks entries outright
  during "risky" markets (didn't help — vol-aware *position sizing*, by
  contrast, did help).
- An overnight-gap mean-reversion signal, and a standalone
  low-idiosyncratic-volatility (defensive/low-vol factor) strategy — both
  underperformed on their own.
- Timing 2x/3x leveraged ETFs (long/inverse rotation, and a simpler
  trend-following-with-volatility-overlay version) — both lost to simply
  buying and holding the same leveraged ETF; leveraged-instrument decay in
  choppy markets is a real, hard-to-time cost.
- **Nine additional U.S.-equity "diversification sleeve" ideas, each a
  different fundamental/event-driven signal, were all tried and rejected**:
  post-earnings-announcement drift, insider-buying clusters, mirroring
  elected officials' stock trades, short-interest level, daily short-sale
  volume, S&P 500 index-deletion overshoot, lottery-demand/skewness
  avoidance, and long/short pairs mean-reversion. Every one either (a)
  correlated too heavily with the existing signal to add real
  diversification, or (b) looked genuinely strong on a long backtest window
  but the strength was concentrated in an earlier period that doesn't
  overlap with the period actually used to validate the current live
  configuration — once tested on the matched, current-relevant window, the
  edge mostly vanished and made the overall portfolio *worse*, not better.
  **We now believe another cross-sectional signal on the same U.S.
  large/mid-cap equity universe is unlikely to add real value — a genuinely
  different asset class, geography, or holding-period horizon is a much
  higher-value direction than another characteristic/event-driven sort on
  the same stocks.**
- Holding intermediate Treasuries (e.g. a 7–10 year Treasury ETF) only over
  the last few trading days of each month, a published month-end Treasury
  premium. The effect was genuinely there out of sample. But funding it by
  selling the index holding added nothing, because the stock index earns at
  least as much over the same turn-of-month days. Variations on month-end
  duration timing, or a month-end stock-to-bond rebalancing tilt, are
  covered by this result. **General lesson: a calendar or flow premium in
  one asset may simply overlap with a premium the index already collects on
  the same days.** Say which days your idea is invested, and why the index
  wouldn't already earn the same thing then.
- Four other ideas were investigated and found to require data we can't
  access affordably: analyst-estimate-revision momentum, a firm-
  characteristics rebalancing-flow signal keyed to an academic database
  identifier with no free ticker mapping, option-implied volatility skew,
  and crypto perpetual-futures funding-rate carry beyond about a year of
  history. The common thread: **genuine point-in-time historical depth for
  analyst estimates, options chains, and permno/CRSP-style academic
  identifiers has consistently turned out to be a paid-vendor-only
  problem** in our experience, no matter how many free sources we checked —
  if your idea needs one of these, flag the data question loudly rather
  than assuming a workaround exists.
- A later, cross-asset-focused research pass (FX, commodities, rates,
  crypto) already surfaced and evaluated a wide net of ideas in those
  categories. Most were well-supported, but several were found to be
  weakly sourced (unverified or fabricated citations, blog-only sourcing,
  off-topic citations) and are deliberately not worth re-proposing unless
  you can bring a genuinely stronger source than what's already been
  checked: a Treasury-duration regime-switching signal driven by an
  unverified GitHub/arXiv source; a crypto options volatility-risk-premium
  harvest with no crypto-specific academic backing; stablecoin CeFi/DeFi
  yield arbitrage sourced only from industry reports; a crypto token-
  unlock event-driven short with no empirical crypto study; a G10
  currency carry-momentum-value portfolio whose cited paper turned out to
  be real but to support a different factor model (dollar + carry +
  business-cycle), not that rule; an ETH/BTC ratio mean-reversion idea with
  only practitioner-blog sourcing; a commodity harvest/planting
  seasonality signal whose only located citation turned out to study an
  unrelated topic; and an energy pre-holiday seasonal trade that its own
  source labeled non-academic. If you land on something in these same
  specific niches, it needs a materially better citation than "an industry
  report" or "a practitioner blog post" to be worth including.

The throughline in most of these failures: technical, price-action-only
signals on this specific universe are close to fully arbitraged (and the
one that seemed to work owed its edge to a survivorship error), adding
model complexity on top of a simple ensemble has consistently made things
worse, and even genuinely different *signal categories* (fundamental,
event-driven, behavioral) have failed to diversify the portfolio as long as
they're still U.S. large/mid-cap equities. **The most valuable thing you can
bring now is a genuinely different asset class, market, or time horizon** —
not another variation on technical/price-action timing, and not another
characteristic-based cross-sectional sort of the same large-cap U.S.
equities, however different the underlying data source.

## Constraints — keep recommendations realistic for a retail/home-lab setup

- **Data cost**: assume free-or-cheap data sources only by default (e.g.
  Yahoo Finance-class historical OHLCV, free/low-cost fundamentals or
  earnings-calendar data). If a promising idea needs paid/institutional data
  (deep historical options chains, tick-level data, alternative data feeds),
  say so explicitly and flag it as a feasibility risk rather than skipping
  it — some ideas are still worth flagging even if the data question needs
  resolving first.
- **Instrument scope**: U.S. equities (large/mid/small-cap indices) and
  major crypto assets remain in scope, but per the note above, ideas in
  **other liquid, retail-accessible asset classes** (developed-market
  international equities, Treasury/duration instruments, commodities,
  currencies) reachable via common ETFs are now explicitly welcome and
  probably higher-value than another U.S.-equity cross-sectional sort. Note
  which asset class each idea is in.
- **Execution realism**: this is a single retail-sized account, not a fund
  — avoid strategies that only work at institutional scale/capacity, or
  that require latency/infrastructure a retail trader can't get.
  Concretely, the account trades through a retail broker that offers U.S.
  stocks, ETFs, listed options and spot crypto — **no futures, no spot
  FX** — without leverage (1.0x), and makes **one decision per day on
  daily bars, shortly before the U.S. close**. If an idea needs futures,
  FX forwards, leverage or intraday trading, say which ETF proxy could
  stand in and how closely it tracks the studied effect.
- Favor ideas with genuine, articulable economic or structural rationale
  (why should this anomaly exist and persist) over pure pattern-mining.

## Required output format

**Group candidates by strategy type, and give every candidate all seven
fields below (four of which are now independent ratings, not one) — both
parts are required, not either/or.** Two prior runs each fixed one gap:
the first produced a well-organized report but skipped several
per-candidate fields; the second added the fields back but used a single
"Confidence" score, which a later independent review showed conflates
three genuinely different questions (is the source real? does it test
*this* exact rule? can this project's instruments actually reproduce the
effect?) — collapsing them cost real accuracy (a real paper wrongly
labeled "fabricated," a venue misattributed, a magnitude overstated, a
finding's direction flipped without a direct re-read to confirm). Both
fixes now apply together.

### Grouping

Organize candidates by **strategy type**, not by source-prestige tier:
- **A. Active strategy replication queue** — candidates with a plausible
  path to a specific tradable rule. Order roughly by how directly the
  cited evidence maps to the proposed trade (not by how prestigious the
  journal is).
- **B. Risk and exposure overlays** — candidates better used as risk-
  management triggers (position-size cuts, leverage reduction, hedge
  ratio) than as standalone directional trades. A source can strongly
  support the *mechanism* (e.g. "X raises jump risk") without supporting
  the *direction* of a trade built on it (e.g. "so always short after
  X") — say so explicitly when that's the case rather than proposing the
  more aggressive standalone-trade framing by default.
- **C. Portfolio-construction and research methods** — not standalone
  alpha sources; methods for improving an existing strategy set (pair
  selection, automated discovery frameworks, etc.).
- **D. Parked hypotheses** — plausible mechanism, thin/unverified/
  off-topic sourcing. Still include these (a labeled long-shot is useful
  triage material), but say plainly the sourcing is weak rather than
  dressing it up, and don't recommend real research time on them —
  a fast, cheap, frozen-rule falsification test is the right scope, not
  a full build.

Within each group, order roughly by how promising + feasible you judge
each idea. Aim for at least 5-8 candidates total so there's real triage
material.

### Per-candidate fields (required for every candidate, in every group)

1. **Name** — short, descriptive.
2. **Mechanism** — 2-4 sentences: what the signal/strategy actually is and
   how it would generate trades. If a source only supports the underlying
   *mechanism* but not the *specific trading rule* being proposed on top
   of it, say so here explicitly (this is the single most common gap
   found on review — don't let a well-cited mechanism paper silently
   stand in for an untested rule built on top of it).
3. **Source(s)** — real citations: paper titles/authors/links, blog posts,
   or other sources you actually found. Do not fabricate citations. If you
   can't find a solid source, say so explicitly rather than inventing one
   — but also don't declare a citation "fabricated" without a thorough
   search (title, claimed ID, author names, working-paper repositories,
   conference programs, faculty pages) — a citation that looks unfindable
   on a first pass can still be real under a slightly different title or
   ID. If you're reusing a source already cited for another candidate in
   this same report, say so rather than re-describing it as new. Check
   basic facts that are easy to get wrong: publication venue (a working
   paper can have been accepted since your training data), exact figures
   (don't round or misremember a headline number), and date.
4. **Ratings** — four independent ratings, not one combined score:
   - **Evidence status**: Peer-reviewed / Accepted / Working paper /
     Practitioner / Unsupported — how mature and credible is the source
     itself?
   - **Rule correspondence**: Direct / Close adaptation / Mechanism only /
     Unrelated — did the source test *this exact* tradable rule, or just
     the underlying mechanism?
   - **Implementation class**: Retail / Professional / Institutional —
     can retail-accessible instruments (ETFs, spot FX, major exchange
     APIs) actually reproduce the studied effect, or does it need
     futures/forwards/swaps/institutional access this project doesn't
     have? Flag explicitly when a retail proxy is only an approximation
     of what the paper actually studied.
   - **Validation stage**: always "Literature only" at this stage (a
     source existing is not the same as a rule being backtested) — state
     it anyway for consistency with how this project tracks
     already-built candidates.
5. **Why it's plausible** — the economic or structural rationale for why
   this edge should exist. Do not fold this silently into the mechanism
   description — call it out explicitly.
6. **Data requirements** — what data it needs, and whether that's
   free/cheap or a feasibility risk (see constraints above).
7. **How it differs from what's already been tried** — one explicit
   sentence per candidate tying it back to the "don't re-propose" list
   above (both the technical-signal list and, if relevant, the specific
   weakly-sourced niches named there from a prior pass).
