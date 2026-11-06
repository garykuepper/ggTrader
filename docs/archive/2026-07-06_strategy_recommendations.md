# ggTrader Quantitative Evaluation & Strategy Roadmap

Reviewed as a senior quant/PM against the actual codebase (`wfo.py`, `gates.py`,
`simulate.py`, `ensemble.py`, `ensemble_ic.py`, `ic_weights.py`, `data.py`,
`roadmap.md`, git log) — not just the stack description. A prior draft of this
document made some generic or incorrect claims about the gating mechanics and
proposed a strategy the data layer can't support; this version corrects those
against source.

---

## 1. Architectural Critique

**Strengths, confirmed in code:**

- WFO folds (`wfo.py:171-187`) are rolling 12-month train / 3-month test
  windows, advancing by the test width — deliberately overlapping training data
  is normal for WFO, not a bug.
- Two independent statistical gates must both pass before a fold's in-sample
  winner is trusted to deploy: **NDH** (neighborhood-density hurdle — ≥85% of a
  parameter cell's immediate neighbors must be co-positive on Sharpe and
  expectancy, `gates.py:56-118`) and **DSR** (Deflated Sharpe Ratio, Bailey &
  López de Prado, penalizing the grid size as multiple trials, `gates.py:185+`).
  A fold that fails either gate falls back to a global min-drawdown anchor set
  rather than deploying an untrusted winner. This is genuinely institutional-
  grade discipline — most retail frameworks don't bother with either check, let
  alone both.
- The vol-targeting scalar is correctly lagged one bar (`compute_vol_scalar`,
  `simulate.py:66-82`) — no look-forward leak there.
- IC weighting's trailing-IC window explicitly drops the last `horizon` bars
  whose forward return isn't realized yet (`ic_weights.py:73-84`) — a proper
  causal guard, not a naive rolling-correlation leak.

**Real vulnerabilities (verified against source, not speculative):**

- **No purge/embargo gap at the fold boundary.** `Fold(cursor, train_end,
  train_end, test_end)` (`wfo.py:185`) sets `train_end == test_start` exactly.
  Any indicator whose rolling window straddles that boundary bar can leak
  information from the edge of training into the first bars of test. Low
  severity for daily EMA/RSI/BB windows, but it's a real correctness gap worth
  naming rather than assuming away.
- **DSR's multiple-testing correction only counts the current grid**
  (`n_trials = len(grid)`, `wfo.py:520`), not the cumulative number of distinct
  strategy hypotheses the project has actually tried (ensembles, IC-weighting,
  Kelly sizing, ML gates, exit-rule sweeps — well over a dozen). The deflation
  is structurally optimistic relative to the true multiple-comparisons problem
  the research program as a whole is running.
- **The composite fold score is an unswept, hand-picked linear blend**
  (`0.5·Sharpe + 0.3·Sortino − 0.2·|MaxDD|`, `wfo.py:199-225`) — never itself
  validated against alternative weightings, so fold "winners" are sensitive to
  an arbitrary choice that hasn't been stress-tested.
- **A previously-flagged conviction-sizing bug** (staff code review,
  2026-06-25): `_generate_signals_with_sizes` ignored the `voters` constructor
  argument and hardcoded all six indicators, silently reintroducing the
  rejected MTF voter into any conviction-sized backtest. A later archive note
  marks it fixed — worth a direct spot-check of `ensemble.py:348-419` before
  trusting any conviction-sized run.
- **Data infrastructure is daily-bars-only.** `STOCK_BASE_CONFIG["FREQ"]="1d"`
  and the equity loader only ever requests `interval="1d"` (`data.py`). There is
  no intraday, tick, or true order-flow data in TimescaleDB today. This isn't
  itself a bug, but it hard-rules-out a category of microstructure strategies
  until intraday ingestion is built — see §2.
- **Cash-sharing under high turnover is unverified at the edges.**
  `from_orders`/`from_signals` both run with `cash_sharing=True`; simultaneous-
  entry competition for limited cash under N-of-M voting hasn't been explicitly
  stress-tested. Flagging as unverified, not confirmed broken.

---

## 2. What to Avoid (Gap Analysis)

Every item below is already closed, with fold-level evidence in `roadmap.md` —
re-researching any of them wastes time the team doesn't need to spend:

- **More price-derived technical voters** (Stochastic, Keltner, Ichimoku) — same
  autocorrelation family as the existing BB/RSI/EMA/MACD/VBB set. The project's
  own pattern is that "winning" tweaks in this family win only 3/17 or 7/17
  folds — regime-specific noise the gates correctly reject, not real edge.
- **IC-weighted voting variants** or deeper signal-blending layers — closed
  2026-06-28, OOS Sharpe 1.01 vs 1.12 baseline, worse drawdown.
- **Kelly or other trade-history-conditioned sizing** — closed 2026-07-06,
  Sharpe 0.98, DD widened to −17%, no multiplier won fold-majority.
- **Exit-rule redesigns** (take-profit grids, time-stops, trailing stops) —
  closed 2026-06-28; unmodified RSI exit remains WFO-preferred.
- **ML/LightGBM entry-timing filters** of any kind (probability gate, EV
  regressor, volatility filter) — falsified three separate ways 2026-06-28.
- **Cross-sectional or dual momentum** on the current large-cap universe —
  NO-GO, large-cap momentum is over-arbitraged.
- **MTF (weekly) reversion** — NO-GO, actively harmful to Sharpe.
- **Sector-constrained/sector-neutral diversification "from scratch"** — this
  is **already mid-implementation, uncommitted right now**
  (`data/universe/sp500_sectors.json`, `tests/lab/test_sector_constraints.py`,
  modified `registry.py`/`ensemble.py`/`ensemble_ic.py`/`momentum.py`).
  Recommending it as a fresh idea would duplicate in-flight work — see §3-C for
  how to build on top of it instead.
- **Crypto funding-rate carry** — not a research gap. `FundingCarryBTC` is
  already built and backtested (Sharpe 4.4–8.6 on tiny notional); it's capital-
  and data-gated for deployment, not awaiting a research decision.
- **Order-flow imbalance / signed-volume microstructure signals** — infeasible
  with the current data layer. This looks like it fits the vectorbt broadcast
  model, but daily bars have no reliable up-tick/down-tick attribution; it
  needs new intraday/tick ingestion before it's even testable, so it is not a
  cheap next step despite appearances.

---

## 3. Strategy Candidates to Research First

Ranked for genuine orthogonality to the already-exhausted price-derivative
family, and for fitting the *existing* daily-bar data layer with no new
ingestion work.

### A. Overnight Gap / Session Return Decomposition

- **Quantitative premise:** Decompose each day's return into close→open
  (overnight) and open→close (intraday) legs. The overnight leg concentrates
  the equity risk premium (scheduled news, institutional order flow) with
  structurally different variance than the intraday leg (retail-flow-driven,
  often mean-reverting after large gaps). This is a session-structure anomaly,
  not another price-momentum/reversion flavor — genuinely orthogonal to the
  existing 6-voter family.
- **VectorBT implementation vector:** Compute `gap_t = (open_t - close_{t-1}) /
  close_{t-1}` and `intraday_t = (close_t - open_t) / open_t` as new matrix
  columns. Feed into `Portfolio.from_signals` via the existing
  `simulate_signals` path, with entries priced at `open` rather than `close`
  (a small `simulate.py` change to accept a per-leg price series — the `open`
  field is already loaded in the OHLCV frame, just unused by current voters).
  Z-score the gap on a rolling window; test both a fade-the-gap
  (mean-reversion) signal and a hold-overnight-only (carry) signal.
- **Data requirements:** None new — `open`/`close` are already ingested per
  bar.
- **WFO/robustness integration:** Sweep the gap Z-score threshold and lookback
  window exactly like existing indicator params; runs through the same
  NDH/DSR gates unmodified. This is the cheapest possible test to falsify or
  confirm — no new data engineering, no new simulation machinery.

### B. Cross-Sectional Idiosyncratic Volatility (defensive/lottery-premium)

- **Quantitative premise:** Stocks with volatility unexplained by the market
  (residual variance vs. a benchmark) have historically underperformed —
  retail overpays for positive-skew "lottery" exposure. This is a
  cross-sectional relative-value premium, not a time-series signal, and is
  mechanically orthogonal to the existing per-symbol indicator voting.
- **VectorBT implementation vector:** Rolling OLS (vectorized or Numba-
  compiled) of each symbol's daily return against SPY over a ~20-day window;
  extract residual variance per symbol per day; rank cross-sectionally; go
  long the bottom quintile and short/underweight the top quintile. This is a
  weight-based strategy — it reuses the existing `simulate_weights` /
  `Portfolio.from_orders(size_type="targetpercent")` path already built for
  the momentum sleeves, not new simulation machinery.
- **Data requirements:** Daily returns for the universe plus SPY as
  benchmark — already available.
- **WFO/robustness integration:** Sweep the regression window and quantile
  cutoff. Because it's cross-sectional and long/short, it's naturally less
  correlated to the existing long-only reversion sleeve — a good diversifier
  candidate even if its standalone Sharpe is modest.

### C. Sector-Neutral Relative-Value Reversion

- **Quantitative premise:** Existing reversion signals are single-name and
  fully beta-exposed. A sector-neutral version — rank names within each GICS
  sector by short-term reversion signal, go long the most-oversold and
  short/de-weight the least-oversold within the same sector — hedges out
  market and sector beta explicitly, which the current ensemble does not do
  at all.
- **VectorBT implementation vector:** Cross-sectional rank within each sector
  bucket per rebalance date, mapped to target weights via the same
  `from_orders` weight-based path. This is deliberately sequenced to land
  right after the in-progress sector-constraint work lands — it reuses the
  sector map being built right now (`data/universe/sp500_sectors.json`)
  instead of requiring separate new research.
- **Data requirements:** Already satisfied by the in-flight sector work; no
  new ingestion.
- **WFO/robustness integration:** Sweep the within-sector reversion lookback
  and the long/short split. The sector-neutral construction should itself
  reduce factor-crowding risk relative to the existing single-name reversion
  sleeve, making genuine (non-regime-specific) edge more likely to clear the
  DSR gate.

### D. Calendar / Seasonality Effects (cheap bonus falsification pass)

- **Quantitative premise:** Turn-of-month and month-end fund-flow rebalancing
  effects are well-documented, structurally distinct from price-derivative
  signals, and cost essentially nothing to test.
- **VectorBT implementation vector:** A boolean day-of-month/day-of-week
  matrix multiplied against existing position-weight logic — trivial to
  implement in an afternoon.
- **Data requirements:** None — pure calendar arithmetic on existing
  timestamps.
- **WFO/robustness integration:** Because it's nearly free to build, run it
  as a quick falsification pass before or alongside A–C rather than treating
  it as a standalone research investment.

---

## 4. Next Steps & Priority Ranking (Alpha-to-Complexity)

1. **Overnight Gap Decomposition (A)** — highest ratio. Zero new data
   engineering, reuses `open`/`close` fields the ensemble currently ignores,
   orthogonal mechanism to every rejected price-derivative lever, and fits the
   exact same WFO/gate pipeline unmodified. **Build this first.**
2. **Cross-Sectional Idiosyncratic Volatility (B)** — reuses the
   `simulate_weights`/`from_orders` path already built for momentum sleeves;
   a genuinely structural risk premium rather than a directional bet; good
   candidate as a permanent diversifying sleeve regardless of its standalone
   Sharpe.
3. **Sector-Neutral Relative-Value Reversion (C)** — deliberately sequenced
   after the in-progress sector-constraint work lands, since it reuses that
   infrastructure directly instead of duplicating it. Explicitly hedges the
   beta exposure the current ensemble carries unhedged.
4. **Calendar/Seasonality (D)** — lowest priority/investment; worth a fast
   afternoon pass given it's nearly free, but not a structural bet on par
   with A–C.

**Execution directive:** Build Overnight Gap Decomposition next in
`src/ggTrader/lab/strategies/`, wire it into `simulate_signals` with an
`open`-priced entry/exit option, and run it through the existing
`ggt lab --wfo` harness before touching anything else — it is the only
candidate here requiring zero new data infrastructure and zero new simulation
machinery.
