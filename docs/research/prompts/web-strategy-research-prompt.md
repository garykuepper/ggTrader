# Prompt: Discover New Trading-Strategy Ideas (External Research Only)

**Portable prompt — paste this whole thing into a web-research tool (e.g.
Google Gemini's or Claude's research/web-search feature).** It assumes only
general web-browsing/search ability, nothing else — no code execution, no
file access, no specific tools or plugins. Do not write code, backtest
anything, or assume access to any codebase. Your job is research and
recommendation only; someone else will implement and test whatever you find.

Regenerate the "context" section below from `docs/research/RESEARCH_SNAPSHOT.md`
whenever it's materially out of date — last synced 2026-07-19.

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

**Currently deployed and working:** a majority-vote ensemble of five
technical indicators (Bollinger Bands, RSI, EMA crossover, MACD divergence,
Volume-confirmed Bollinger Bands) on individual S&P 500 stocks, flat ~3%
position sizing per trade — achieves roughly Sharpe 1.1 vs. the index's
~0.6-0.8. On top of that, a blended portfolio across three large-cap/mid-cap
U.S. equity indices (S&P 500, MidCap 400, Nasdaq-100), volatility-weighted
and capped at 1x leverage, pushes that to roughly Sharpe 1.1-1.2 with
meaningfully lower drawdown.

**Already tried and rejected — don't propose close variants of these
without a genuinely different mechanism:**
- A machine-learning classifier gating which technical-indicator signals to
  trade (proved anti-predictive/worthless across several redesigns).
- Weighting the ensemble's votes by each indicator's historical predictive
  skill, instead of equal-weighting (hurt risk-adjusted return).
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

The throughline in most of these failures: technical, price-action-only
signals on this specific universe are close to fully arbitraged, adding
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
- Favor ideas with genuine, articulable economic or structural rationale
  (why should this anomaly exist and persist) over pure pattern-mining.

## Required output format

Produce a ranked markdown list. For **each** candidate idea, include:

1. **Name** — short, descriptive.
2. **Mechanism** — 2-4 sentences: what the signal/strategy actually is and
   how it would generate trades.
3. **Source(s)** — real citations: paper titles/authors/links, blog posts,
   or other sources you actually found. Do not fabricate citations — if you
   can't find a solid source for something you believe is promising, say so
   explicitly rather than inventing one.
4. **Why it's plausible** — the economic or structural rationale for why
   this edge should exist.
5. **Data requirements** — what data it needs, and whether that's
   free/cheap or a feasibility risk (see constraints above).
6. **How it differs from what's already been tried** — one sentence tying
   it back to the "don't re-propose" list above, so it's clear this isn't a
   near-variant of a rejected idea.

Aim for at least 5-8 candidates so there's real triage material, ranked
roughly by how promising + feasible you judge each to be. It's fine — good,
even — to include a couple of longer-shot or more speculative ideas at the
bottom, clearly labeled as such.
