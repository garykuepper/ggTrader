# ggTrader Research Prompt Kit

Nov 5, 2026 · @Gary

## How to use

Run these as separate turns in one conversation, pasting Turn 0 at the top of each. You review and approve each turn's output before the next one builds on it.

The main change from the original template: it asked the model to build a WFO harness from scratch. You already have one (wfo\_pipeline with DSR gating, WFE tracking, regime state machine). So this kit has the model produce **candidate signal modules that plug into your pipeline**, and spends its rigor on the parts your pipeline can't check: whether the literature is real, whether the signal timing is right, and how many trials were burned getting there.

Other fixes baked in:

- Citations only from retrieved sources; unverifiable IDs flagged, not invented.
- Fill convention stated explicitly (signal on close of t, fill at open of t+1) instead of a vague `.shift(1)`.
- Trial counting and deflated Sharpe required on every grid.
- Your actual fee tier (0% maker / 0.02% taker) as the base case, since taker fees were the gate that killed higher-frequency candidates.
- An adversarial review turn at the end, a four-level verdict, fragility tests, and an independent P&L check (adapted from a second review).

Fill the `[BRACKETS]` before sending. Use a model with web search for Turn 1.

## Turn 0: Shared context

Paste this at the top of every turn.

```text
CONTEXT (applies to every response in this conversation)

Hypothesis under study: [HYPOTHESIS, e.g. "ATR-percentile regime filter on a slow EMA trend signal reduces drawdown vs. buy-and-hold without giving up more than X% CAGR"]

Universe (test both asset classes, report them separately, never pool their statistics):
- Crypto: [BTC/USD, ETH/USD, DOGE/USD] spot on Binance.US. 24/7, no session gaps.
- Stocks/ETFs: [SPY, QQQ, + named single stocks] via [BROKER]. Regular session only unless stated; overnight and weekend gaps are real and must not be smoothed. Use split- and dividend-adjusted prices for signals, and state whether returns include dividends. For single stocks, name the survivorship-bias risk of the chosen list.
Long-only unless I say otherwise; no perps, no shorting, no margin.
Bars: [1d] for both (a shared bar size keeps the comparison fair), timestamps in UTC. Crypto data comes from my TimescaleDB via ccxt; equity data from [SOURCE]. Assume one DataFrame per symbol with a UTC DatetimeIndex and columns open/high/low/close/volume. Do not forward-fill missing equity sessions into fake bars.
Period: [START] to [END]. Treat the final [12] months as a locked holdout; do not tune on it.

Costs (base case):
- Crypto: taker 0.02% per side, maker 0%; assume taker fills unless the strategy posts limit orders. Slippage [5] bps per side.
- Stocks/ETFs: commission [0 / broker rate], half-spread [1-2] bps for SPY/QQQ and [5-10] bps for single stocks, plus SEC/FINRA fees on sells. Model open fills at the opening auction price with extra slippage of [5] bps, since gap opens fill worse.
Sensitivity sweep at 2x and 4x total cost for both.

Stack: Python 3.11, vectorbt (open-source) [VERSION], pandas, numpy. No vectorbt PRO APIs.

Timing chain (state all four for every rule): (1) data available: bar t's OHLCV once bar t closes; (2) signal: computed at bar t close; (3) order submitted: immediately after; (4) earliest fill: bar t+1 open. Every input to a decision must be available by step (1), including rolling stats, percentiles, and regime labels. State this explicitly in code (shift signals once, pass price=open). Never shift both the indicator and the signal.

Rules for you:
- If you are unsure, say so. Do not invent citations, API arguments, or results.
- Count every parameter combination you evaluate. Start every response with N = cumulative combinations tested so far in this conversation, including reruns and abandoned variants.
- Report failed experiments and negative findings. Rejecting the strategy is a successful outcome, not a failure.
- Keep answers to what the turn asks; I will ask for more.
```

## Turn 1: Verified literature

Needs web search on. Spot-check two of the returned IDs yourself before moving on.

```text
TASK: Find 3-5 papers that test this hypothesis or its closest mechanism. Cover both crypto and equities: at least one out-of-sample study on each, and note whether any paper tests the same signal on both. Label each paper's asset class.

Search arXiv (q-fin), SSRN, and journal sites. Only list a paper you actually retrieved in this session. For each:
- Citation with arXiv ID or DOI, plus the URL you opened. If you could not open it, mark UNVERIFIED.
- The signal, written as an equation using t and t-1.
- Sample (assets, dates, bar size) and whether results were out-of-sample.
- Reported net-of-cost performance and the cost assumption used.
- Failure modes: regimes where it broke, decay after publication, capacity/turnover limits.

Then, in 3-5 sentences: does the evidence support the hypothesis, contradict it, or not address it at my bar size and cost level? Flag any paper whose result disappears at 4 bps round-trip or more.

Do not write code in this turn.
```

## Turn 2: Strategy spec

The key output here is the parameter grid, sized before anything runs. A small grid keeps the deflated Sharpe hurdle reachable.

```text
TASK: Turn the hypothesis into a precise spec. No code yet.

1. Entry, exit, and filter rules as equations. Every indicator value used at decision time t must be computable from bars <= t. Mark which bar's price fills each order (t+1 open).
2. Stops/targets: if used, state how a bar that touches both the stop and the target is resolved, and that this is an assumption the backtest cannot verify from OHLC.
3. Position sizing (fixed fraction or volatility-scaled) as an equation.
4. Parameter grid: list each parameter, its range, and why. Report the total number of combinations N. Keep N under [200] unless you justify more. Prefer parameters with an economic reason over fine-grained sweeps.
5. Expected turnover (round trips per year) and the break-even cost per round trip, given the gross edge the literature suggests.
6. Benchmarks: buy-and-hold per asset, an equal-weight basket per asset class, and SPY for the equity side. If the rules need to differ between crypto and stocks (session gaps, volatility scale), state each difference and why, and count any separately tuned parameters toward N.
7. What result would falsify the hypothesis? Write the kill criteria now, before seeing results.
```

## Turn 3: vectorbt implementation

Paste your pipeline's signal interface where marked so the output drops straight into wfo\_pipeline. The truncation test is the single most useful check against look-ahead; don't skip it.

```text
TASK: Implement the approved spec from Turn 2.

Deliver two things:

A. A signal module matching this interface:
[PASTE your wfo_pipeline signal/strategy interface: function signature, expected inputs, return types]

B. A standalone sanity script that:
1. Computes indicators with vbt indicator .run() calls, passing parameter lists so the grid broadcasts as MultiIndex columns. No Python loops over parameters or rows.
2. Builds entries/exits from bar-t data, shifts them one bar, and calls vbt.Portfolio.from_signals with close, open/high/low (if stops are used), price=open, fees, slippage, init_cash, and an explicit freq.
3. Runs the full grid at base cost, 2x, and 4x.
4. Reports per combination: total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, trade count, win rate, profit factor, expectancy, turnover, average holding period, and exposure %. Also report the same for the benchmarks.
5. Includes a look-ahead test: for 20 random cut points t, recompute signals on data truncated at t and assert they match the full-history signals up to t.
6. Includes an accounting cross-check: for 5 trades from the best combination, recompute entry price, exit price, fees, slippage, and P&L in plain pandas, independent of vectorbt, and assert they match the portfolio's trade records within 1e-8.

Constraints:
- Runnable as-is against a DataFrame in the stated format; no TODOs.
- Only use vectorbt arguments you are confident exist in [VERSION]. If unsure of one, say so instead of guessing.
- List any place where vectorization changes behavior vs. an event-driven fill (intrabar stop ordering, partial fills, cash constraints). If any of these could materially change results, implement that part with vbt.Portfolio.from_order_func instead and compare the two.
- Pin every package version in a requirements block and set any random seeds.
```

Run it yourself before Turn 4, and paste back the metrics table and any errors.

## Turn 4: Overfitting audit

This turn works on results you paste back, not on numbers the model makes up. Your pipeline's DSR gate does part of this; the point here is a second, independent read before a candidate goes in.

```text
TASK: Audit the results I pasted below for overfitting. Use only these numbers.

[PASTE metrics table, trade counts, and the total number of combinations N tested across ALL turns so far, including any reruns]

1. Deflated Sharpe ratio for the best combination, using N trials and the skew/kurtosis of its returns. Show the calculation.
2. Probability of backtest overfitting via combinatorially symmetric cross-validation (CSCV) on the return matrix. Give code if I need to run it.
3. Parameter stability: is the best combination on a plateau or a spike? Describe performance of its neighbors.
4. Per-asset-class (crypto vs. stocks), per-asset, and per-regime breakdown (trend up / trend down / range, defined by a rule you state). Does the edge come from one asset or one period?
5. Cost survival: at which cost multiple does net edge vs. benchmark go to zero?
6. Minimum track record length needed for the observed Sharpe to be significant at 95%.
7. Fragility tests (give code if I need to run them): drop the 5 best trades and the single best month, then recompute; delay all fills by one extra bar; shift each parameter one grid step in every direction.
8. Verdict against the kill criteria written in Turn 2, as exactly one of:
   - REJECTED: no robust edge after costs.
   - INCONCLUSIVE: evidence too thin or too sensitive to assumptions.
   - RESEARCH CANDIDATE: worth more independent testing.
   - SHADOW CANDIDATE: meets all kill criteria; ready for shadow-mode forward testing in ggTrader.
   One sentence why. Never call anything production-ready from a backtest alone.
```

## Turn 5: Adversarial review

Run this in a **fresh conversation**, ideally with a different model. A model reviewing its own code tends to defend it.

```text
You are reviewing a backtest someone else wrote. Your job is to find reasons not to trade it.

[PASTE Turn 0 context, the Turn 2 spec, the Turn 3 code, and the Turn 4 audit]

Check for:
- Look-ahead: any indicator, rolling stat, normalization, or regime label that uses data after t, including full-sample z-scores or percentiles.
- Fill timing: signal and fill on the same bar, double shifts, or price=close where the spec says open.
- Cost errors: fees applied once instead of per side, slippage missing on exits, maker fees assumed for market orders.
- Survivorship and data issues: delisted pairs and stocks, exchange downtime, bad ticks, missing bars filled forward, unadjusted splits or dividends, stock lists picked with hindsight.
- Selection bias: trials not counted, holdout touched during tuning, kill criteria changed after results.
- Live gaps: anything that works in the backtest but not on Binance.US or [BROKER] (minimum order size, rate limits, order type support, latency at bar open, pattern-day-trader rules, halts, and opening-auction fills).

Rank issues by how much they could inflate results. For each, cite the exact line or step and give a fix. If you find nothing serious, say so plainly.
```
