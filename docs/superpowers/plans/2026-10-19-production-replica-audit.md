# Production-Replica Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer "does the strategy we actually run beat an appropriate passive alternative?" in
three parts:
1. A faithful replica of the live paper trader.
2. A pass bar fixed in advance: statistical significance plus an after-tax margin.
3. Re-checks of past results that bugs could have hurt.

**Architecture:**
- **Replica engine.** A new lab module, `replica.py`, backtests the live trader's portfolio rules
  in an event loop. It calls the trader's own pure decision functions (sweep, risk, catastrophe
  stop), so the rules cannot drift from production.
- **Statistics.** A new module, `significance.py`, adds Newey-West excess-return statistics.
- **Driver.** A script runs each pinned window and variant, applies the verdict rule fixed in
  advance, and writes JSON.
- **Side work.** Fix the deflated-Sharpe (DSR) gate, re-run momentum and fomc_drift on the
  corrected harness, and put bounds on A23's distribution-tax number.

**Tech Stack:** Python 3.12, pandas, numpy and scipy (all already dependencies). The TimescaleDB
tape is read through the existing `load_ohlcv`. Tests use pytest via `scripts/run_tests.sh`,
and code is checked with ruff.

**Spec:** `docs/research/briefs/2026-10-19-production-replica-audit.md` (the brief). Read it
first. Its §4 pass bar and §3 design values are binding.

## Global Constraints

- **Live is frozen until about 2026-12-29.**
  - Do not edit `src/ggTrader/paper/`, `.env`, `docker-compose.yaml` or `Dockerfile`.
  - Do not touch the `ggtrader_live` container, and do not deploy.
- **Live imports are restricted.** Lab code may import only `ggTrader.paper.cash_sweep`,
  `ggTrader.paper.risk` and `ggTrader.paper.catastrophe_stop`, which are pure. Never import
  `trader`, `signal_runner`, `alpaca_broker`, `notifier` or `persist`. They can place orders or
  send Telegram messages.
- **The database is SELECT only.** The one exception is the cache writes `load_ohlcv` makes
  through `CachedYFinanceLoader`.
- **Pinned windows:**

  | Window | Start | End |
  |---|---|---|
  | Primary | 2021-01-31 | 2026-04-30 |
  | Holdout | 2010-01-31 | 2020-12-31 |
  | fomc_drift | 2012-01-01 | 2026-04-30 |

  Warm-up is 420 calendar days before each start.
- **Bar constants from the brief, with the same names in the code:**

  | Constant | Value | Meaning |
  |---|---|---|
  | `T_BAR` | 3.0 | Newey-West t-stat threshold |
  | `NW_LAGS` | 5 | Newey-West lags |
  | `CAGR_BAR_PP` | 1.0 | Minimum excess CAGR over SPY, in points per year |
  | `HOLDOUT_COVERAGE_FLOOR` | 0.70 | Minimum share of holdout member-days with data |
  | Tax bracket | 24% / 15% | After-tax step |

- **Variant values:**
  - Stocks cost 5 bp per side, with 3 bp and 10 bp sensitivity runs.
  - SPY costs 1 bp per side.
  - Cash earns T-bills (FRED `TB3MS`), with a 0% sensitivity run.
  - The signal lag is 1 bar, with a lag-0 diagnostic run.
- **Artifacts.** Every number you cite must come from a JSON a driver writes under
  `docs/research/`. Drivers checkpoint after each unit of work and support `--help`.
- **Tests and lint.** Run tests only through `scripts/run_tests.sh`. Code must pass
  `ruff check` and `ruff format` (line length 100).
- **Git.**
  - Work on branch `audit/production-replica`.
  - Stage only the files each task names. Never stage the untracked
    `.claude/skills/*-workspace/` dirs.
  - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Do not edit the brief after Task 1 commits it.** Any change needs a new dated brief.

## Review Focus

1. **Cash never goes negative.** On a day with a funding sell and buys, cash must never drop
   below 0. A buy that doesn't fit is skipped, never partly filled. Test in Task 4.
2. **Delisted stocks are sold.** If a held stock's prices stop (delisted or acquired), it is sold
   at its last valid close with reason `"delisted"`, and equity never goes NaN. Test in Task 4.
3. **Missing SPY bars are handled.** On a day with no SPY bar, mark SPY with a forward-fill and
   make no sweep trades. Test in Task 4.
4. **Point-in-time membership is respected.** An entry outside index membership never becomes a
   buy. Test in Task 5.
5. **The 15% drawdown halt fires in bear markets.** A book that is about 95% SPY crosses it in
   2020 and 2022. While halted, all trading stops, the sweep included, but marking to market
   continues. The halt lifts once NAV recovers to within 15% of the peak. Test in Task 4. The
   report must state the number of halted days per window.

---

### Task 1: Pre-register the brief

**Files:**
- Commit: `docs/research/briefs/2026-10-19-production-replica-audit.md`
- Commit: `docs/superpowers/plans/2026-10-19-production-replica-audit.md`

- [ ] **Step 1: Get owner sign-off on every *(planner's call)* item in the brief.** Edit the brief
  only on the owner's instruction.
- [ ] **Step 2: Create the branch**

```bash
git switch -c audit/production-replica
```

- [ ] **Step 3: Commit the two documents**

```bash
git add docs/research/briefs/2026-10-19-production-replica-audit.md \
        docs/superpowers/plans/2026-10-19-production-replica-audit.md
git commit -m "research(audit): pre-register production-replica audit brief and plan"
```

- [ ] **Step 4: Verify the commit**

Run: `git log -1 --stat`
Expected: exactly the two files above.

---

### Task 2: Fix the deflated-Sharpe gate

**Files:**
- Modify: `src/ggTrader/lab/gates.py:133-202`
- Modify: `src/ggTrader/lab/wfo.py:715-722`
- Modify: `scripts/lever_diagnostic.py:239-246`
- Modify: `docs/changelog.md`
- Test: `tests/lab/test_gates.py` (replace the tests at :253-290 and add new ones)

**Interfaces:**
- Produces `dsr_check(observed_sr_annual: float, n_obs: int, n_trials: int, skew: float,
  kurtosis_excess: float, threshold: float = 0.80, periods_per_year: int = 252) -> DsrResult`.
  - It takes the annualized Sharpe that `metrics.curve_stats` returns and divides it by
    `sqrt(periods_per_year)` internally.
  - `DsrResult.expected_max_sr` is reported **annualized**.
- `psr`, `expected_max_sr` and `_sr_standard_error` stay per-period primitives. Their docstrings
  must say so.

- [ ] **Step 1: Write the failing tests.** Replace `test_dsr_check_strong_signal_passes` and
  `test_dsr_check_threshold_respected`. Rename the kwarg in `test_dsr_check_data_mined_fails`
  to `observed_sr_annual`. Add these tests:

```python
import math

from ggTrader.lab.gates import _sr_standard_error


def test_sr_standard_error_normal_matches_lo_2002():
    # Normal returns: Var(SR) = (1 + SR^2/2) / (T-1); pins the (excess + 2)/4 kurtosis term.
    assert _sr_standard_error(0.1, 253, 0.0, 0.0) == pytest.approx(
        math.sqrt(1.005 / 252), rel=1e-12
    )


def test_dsr_check_reference_value():
    r = dsr_check(observed_sr_annual=1.0, n_obs=252, n_trials=48, skew=0.0, kurtosis_excess=0.0)
    assert r.dsr_value == pytest.approx(0.1036, abs=0.002)
    assert r.expected_max_sr == pytest.approx(2.265, abs=0.01)  # annualized
    assert r.passed is False


def test_dsr_check_strong_signal_passes():
    r = dsr_check(observed_sr_annual=3.0, n_obs=504, n_trials=20, skew=0.0, kurtosis_excess=0.0)
    assert r.passed is True
    assert r.dsr_value == pytest.approx(0.9898, abs=0.002)


def test_dsr_check_annual_1p5_over_two_years_is_not_significant():
    r = dsr_check(observed_sr_annual=1.5, n_obs=504, n_trials=20, skew=0.0, kurtosis_excess=0.0)
    assert r.passed is False
    assert r.dsr_value == pytest.approx(0.586, abs=0.005)


def test_dsr_check_threshold_respected():
    kw = dict(observed_sr_annual=1.0, n_obs=252, n_trials=48, skew=0.0, kurtosis_excess=0.0)
    strict, relaxed = dsr_check(**kw, threshold=0.5), dsr_check(**kw, threshold=0.05)
    assert relaxed.passed and not strict.passed
    assert relaxed.dsr_value == strict.dsr_value
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `scripts/run_tests.sh tests/lab/test_gates.py -v`
Expected: FAIL. You should see `TypeError: unexpected keyword argument 'observed_sr_annual'`
and an SE mismatch.

- [ ] **Step 3: Implement the fix**
  - In `_sr_standard_error`, the numerator becomes
    `1.0 - skew * sr + (kurtosis_excess + 2.0) / 4.0 * sr**2`.
    - Reason: γ₄ is raw kurtosis, which is excess + 3, so (γ₄−1)/4 equals (excess+2)/4.
  - In `dsr_check`, compute `sr = observed_sr_annual / math.sqrt(periods_per_year)` and use it
    everywhere. Return `expected_max_sr * math.sqrt(periods_per_year)`.
  - Rename the kwarg at both call sites: `wfo.py:716` and `lever_diagnostic.py:240`.
- [ ] **Step 4: Run the tests and confirm they pass**

Run: `scripts/run_tests.sh tests/lab/test_gates.py -v`, then `scripts/run_tests.sh tests/lab -q`.
Expected: PASS.
- If a walk-forward (WFO) test fails because its synthetic fixture relied on the old gate
  always passing, change that test's expected value to the corrected outcome.
- Name each changed test in the commit message.

- [ ] **Step 5: Add a changelog entry to `docs/changelog.md`.** It must say:
  - The DSR gate compared an annualized Sharpe against per-day thresholds and had a wrong
    kurtosis term, so it passed almost everything. For example, an annual Sharpe of 1.0 over
    252 days with 48 trials scored 1.0 when it should have scored 0.10.
  - Every past "x/17 gates" count reflects the density gate only.
  - With the corrected gate, walk-forward folds will fall back to the anchor much more often.
- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/gates.py src/ggTrader/lab/wfo.py scripts/lever_diagnostic.py \
        tests/lab/test_gates.py docs/changelog.md
git commit -m "fix(lab): DSR gate used annualized Sharpe against per-day thresholds"
```

---

### Task 3: Excess-return significance helpers

**Files:**
- Create: `src/ggTrader/lab/significance.py`
- Test: `tests/lab/test_significance.py`

**Interfaces:**
- Produces the module constants `T_BAR = 3.0`, `NW_LAGS = 5` and `CAGR_BAR_PP = 1.0`.
- Produces `newey_west_se(x: np.ndarray, lags: int) -> float`, the standard error of the mean.
- Produces `@dataclass(frozen=True) class ExcessStats`, with these fields:
  - `n_obs: int`
  - `t_stat: float`
  - `ann_excess_pct: float`
  - `tracking_error_pct: float`
  - `information_ratio: float`
  - `cagr_diff_pp: float`
- Produces `excess_stats(strategy_equity: pd.Series, benchmark_equity: pd.Series,
  lags: int = NW_LAGS, periods_per_year: int = 252) -> ExcessStats`.
- Produces `years_for_t(information_ratio: float, target_t: float = 2.0) -> float`.
- Produces `passes_pretax_bar(stats: ExcessStats, t_bar: float = T_BAR,
  cagr_bar_pp: float = CAGR_BAR_PP) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_newey_west_lag0_is_plain_se():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert newey_west_se(x, 0) == pytest.approx(np.std(x, ddof=0) / np.sqrt(4))


def test_newey_west_widens_under_positive_autocorrelation():
    x = np.sin(np.arange(200) / 10.0)
    assert newey_west_se(x, 5) > newey_west_se(x, 0)


def test_identical_curves_have_zero_excess():
    eq = pd.Series(np.linspace(100, 120, 300), index=pd.bdate_range("2024-01-01", periods=300))
    s = excess_stats(eq, eq)
    assert s.t_stat == 0.0 and s.cagr_diff_pp == pytest.approx(0.0, abs=1e-9)


def test_constant_outperformance_is_detected():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=1260)
    rb = rng.normal(0.0004, 0.01, len(idx))
    rs = rb + 0.0001 + rng.normal(0.0, 0.0005, len(idx))
    bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    strat = pd.Series(100 * np.cumprod(1 + rs), index=idx)
    s = excess_stats(strat, bench)
    assert s.ann_excess_pct == pytest.approx(2.52, abs=1.0)  # sampling noise ~0.35 pt
    assert s.t_stat > 3.0


def test_years_for_t():
    assert years_for_t(0.5) == pytest.approx(16.0)
    assert years_for_t(0.0) == math.inf
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `scripts/run_tests.sh tests/lab/test_significance.py -v`
Expected: FAIL with `ModuleNotFoundError: ggTrader.lab.significance`.

- [ ] **Step 3: Implement**
  - **Newey-West.** Demean x to get d. Then γ_l = Σ_{t>l} d_t·d_{t−l} / n, and
    S = γ_0 + 2·Σ_{l=1..L} (1 − l/(L+1))·γ_l. Return √(S/n).
  - **`excess_stats`, alignment and returns.** Inner-join the two curves on dates and drop
    NaNs. Take daily `pct_change` of each and compute excess = strategy − benchmark.
  - **`excess_stats`, the statistics:**
    - t = mean / NW SE. If both the SE and the mean are 0, t is 0.0.
    - `ann_excess_pct` = mean · 252 · 100.
    - `tracking_error_pct` = std(ddof=1) · √252 · 100.
    - IR = `ann_excess_pct / tracking_error_pct`, or 0.0 when the tracking error is 0.
    - `cagr_diff_pp` = the difference of `metrics.curve_stats(curve)["cagr_pct"]` between the
      two aligned curves. Reuse that helper rather than computing CAGR again.
- [ ] **Step 4: Run the tests and confirm they pass**

Run: `scripts/run_tests.sh tests/lab/test_significance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/significance.py tests/lab/test_significance.py
git commit -m "feat(lab): Newey-West excess-return statistics for pre-registered bars"
```

---

### Task 4: Replica engine

**Files:**
- Create: `src/ggTrader/lab/replica.py`
- Test: `tests/lab/test_replica.py`

**Interfaces:**
- **Consumes** these live functions:
  - `ggTrader.paper.cash_sweep.compute_sweep_buy`
  - `ggTrader.paper.cash_sweep.compute_sweep_sell_for_funding`
  - `ggTrader.paper.cash_sweep.SWEEP_TRADE_REASON`
  - `ggTrader.paper.cash_sweep.DEFAULT_RESERVE_PCT` (0.05)
  - `ggTrader.paper.cash_sweep.DEFAULT_BUY_TRIGGER_PCT` (0.08)
  - `ggTrader.paper.cash_sweep.DEFAULT_MIN_CLIP_USD` (500)
  - `ggTrader.paper.risk.RiskConfig` and `RiskGuard`
  - `ggTrader.paper.catastrophe_stop.find_catastrophe_stops`
  - `ggTrader.paper.catastrophe_stop.CATASTROPHE_STOP_REASON`
  - `ggTrader.paper.catastrophe_stop.DEFAULT_CATASTROPHE_STOP_PCT` (−0.25)
- **Produces** `REASON_SIGNAL = "signal"` and `REASON_DELISTED = "delisted"`.
- **Produces** `@dataclass(frozen=True) class ReplicaConfig`:

  | Field | Type | Default |
  |---|---|---|
  | `start_cash` | float | 100_000.0 |
  | `risk` | RiskConfig | `RiskConfig(max_positions=3)` |
  | `reserve_pct` | float | `DEFAULT_RESERVE_PCT` |
  | `buy_trigger_pct` | float | `DEFAULT_BUY_TRIGGER_PCT` |
  | `min_clip_usd` | float | `DEFAULT_MIN_CLIP_USD` |
  | `catastrophe_stop_pct` | float | `DEFAULT_CATASTROPHE_STOP_PCT` |
  | `stock_cost_bps` | float | 5.0 |
  | `spy_cost_bps` | float | 1.0 |
  | `signal_lag_bars` | int | 1 |

- **Produces** `@dataclass class ReplicaResult`:

  | Field | Type | Meaning |
  |---|---|---|
  | `equity` | pd.Series | NAV after each close, over every index bar. Bars before the first decision hold `start_cash`. |
  | `trades` | pd.DataFrame | One row per fill, in execution order. Columns: `date`, `symbol`, `side` ("buy"/"sell"), `notional`, `price`, `reason`. |
  | `exposure` | pd.DataFrame | Columns `stocks`, `spy`, `cash`, each as a fraction of NAV after each close. |
  | `halted_days` | int | Days when a halt blocked trading. |
  | `truncated_days` | int | Days with more buy candidates than open slots. |

- **Produces** `run_replica(close: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame,
  spy_close: pd.Series, cfg: ReplicaConfig = ReplicaConfig(),
  cash_rate_daily: pd.Series | None = None) -> ReplicaResult`.
  - `close` holds adjusted stock closes, dates × symbols. `entries` and `exits` are boolean
    frames aligned to `close`. `cash_rate_daily` holds simple daily rates; `None` means 0.

**The daily algorithm.** It copies `trader.py`'s order of operations. Decision day *t* is
`index[i]` for `i ≥ signal_lag_bars`; the signal bar *s* is `index[i - signal_lag_bars]`.

1. **Accrue interest.** Multiply cash by `1 + cash_rate_daily[t]`.
2. **Mark and handle delistings.**
   - Mark stocks at `close[t]`. If a held stock's close is NaN and it has no later valid
     close, sell it at its last valid close (reason `delisted`). Otherwise mark it at the
     forward-filled close.
   - Mark SPY at `spy_close[t]`, forward-filled.
   - NAV = cash + stock value + SPY value.
3. **Halts.**
   - The day is halted if `guard.check_drawdown_halt(nav)[0]` or
     `guard.check_daily_loss(nav, prev_nav)[0]` is true. `prev_nav` is the previous bar's
     recorded equity.
   - Then call `guard.update_peak(nav)`.
   - If halted: add 1 to `halted_days`, record equity and exposure, and skip to the next day.
     Nothing trades, the sweep included.
4. **Catastrophe stops.** Build `{sym: {"cost_basis": dollars_paid, "unrealized_pl": value −
   dollars_paid}}` for the held stocks and pass it to `find_catastrophe_stops(...,
   cfg.catastrophe_stop_pct)`. Sell each result (reason `CATASTROPHE_STOP_REASON`) and
   remember it as stopped today.
5. **Strategy sells** (reason `signal`). Sell a held stock if either:
   - `exits.at[s, sym]` is true (the regular last-bar exit), or
   - any `exits[sym]` is true on a bar strictly after its fill date, up to and including *s*.
     This is the live catch-up rule, `trader.py:104-130`.
6. **Buy candidates.** Take the symbols, in alphabetical order, where `entries.at[s, sym]` is
   true, the symbol is not held, it was not stopped today, and `close[t, sym]` is finite.
   - `slots = guard.max_new_positions(held_count)`, counted after today's sells.
   - Buy the first `slots` candidates. If there are more candidates than slots, add 1 to
     `truncated_days`.
   - Each buy's size is `notional = guard.position_notional(nav)`.
   - Keep the `guard.check_concentration(...)` call for parity.
7. **Funding sell.** If SPY has a finite close today, call
   `compute_sweep_sell_for_funding(cash, nav, notional * len(buys), spy_value,
   cfg.reserve_pct)` and execute any sell it returns.
8. **Execute buys.** A buy whose notional exceeds the cash on hand is skipped, never partly
   filled.
9. **Sweep buy.** If SPY has a finite close today, call
   `compute_sweep_buy(cash, nav, cfg.reserve_pct, cfg.min_clip_usd, cfg.buy_trigger_pct)` and
   execute any buy it returns.
10. **Fill prices.** Fill at the day's close × (1 ± cost bps / 1e4). Shares are fractional.
    Record equity and exposure after the trades.

- [ ] **Step 1: Write the failing tests.** They share one builder,
  `_panel(prices: dict[str, list[float]], spy: list[float], entries: dict[str, list[int]] | None
  = None, exits: dict[str, list[int]] | None = None)`. It returns `(close, entries_df, exits_df,
  spy_series)` on `pd.bdate_range("2024-01-01", periods=len(spy))`, with True at the listed
  integer bar positions. In the tests, `C4` means `{s: [10.0] * 5 for s in ["DDD", "AAA", "CCC",
  "BBB"]}`, all four symbols with an entry on bar 1.

```python
def test_buys_fill_alphabetically_up_to_three_slots():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    buys = res.trades.query("reason == 'signal' and side == 'buy'")
    assert list(buys.symbol) == ["AAA", "BBB", "CCC"]
    assert (buys.date == close.index[2]).all()  # signal bar 1, filled bar 2
    assert res.truncated_days == 1


def test_lag_zero_fills_on_signal_bar():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy, ReplicaConfig(signal_lag_bars=0))
    assert (res.trades.query("reason == 'signal'").date == close.index[1]).all()


def test_position_is_3_3pct_of_nav():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    first = res.trades.query("reason == 'signal'").iloc[0]
    assert first.notional == pytest.approx(0.033 * res.equity.iloc[1], rel=1e-5)  # cents rounding


def test_sweep_buys_down_to_reserve():
    close, ent, ext, spy = _panel({"AAA": [10.0] * 3}, [100.0] * 3)
    res = run_replica(close, ent, ext, spy)
    assert res.exposure.loc[close.index[1], "spy"] == pytest.approx(0.95, abs=1e-3)
    assert res.exposure.loc[close.index[1], "cash"] == pytest.approx(0.05, abs=1e-3)


def test_funding_sell_precedes_buys_and_cash_never_negative():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    day2 = res.trades[res.trades.date == close.index[2]]
    assert (day2.iloc[0].reason, day2.iloc[0].side) == (SWEEP_TRADE_REASON, "sell")
    assert (res.exposure.cash >= -1e-9).all()


def test_exit_signal_sells_on_next_bar():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 6}, [100.0] * 6, entries={"AAA": [1]}, exits={"AAA": [3]}
    )
    res = run_replica(close, ent, ext, spy)
    sells = res.trades.query("reason == 'signal' and side == 'sell'")
    assert list(sells.date) == [close.index[4]]


def test_exits_before_fill_are_ignored_but_fill_bar_exit_counts():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 5}, [100.0] * 5, entries={"AAA": [1]}, exits={"AAA": [0, 1]}
    )
    assert run_replica(close, ent, ext, spy).trades.query("side == 'sell' and symbol == 'AAA'").empty
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 5}, [100.0] * 5, entries={"AAA": [1]}, exits={"AAA": [2]}
    )
    sells = run_replica(close, ent, ext, spy).trades.query("side == 'sell' and symbol == 'AAA'")
    assert list(sells.date) == [close.index[3]]


def test_catastrophe_stop_sells_at_minus_25pct():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0, 10.0, 10.0, 7.4, 7.4]}, [100.0] * 5, entries={"AAA": [1]}
    )
    res = run_replica(close, ent, ext, spy)
    stop = res.trades[res.trades.reason == CATASTROPHE_STOP_REASON]
    assert list(stop.symbol) == ["AAA"] and list(stop.date) == [close.index[3]]


def test_delisted_position_sold_at_last_close():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0, 10.0, 10.0, 11.0, np.nan, np.nan]}, [100.0] * 6, entries={"AAA": [1]}
    )
    res = run_replica(close, ent, ext, spy)
    d = res.trades[res.trades.reason == REASON_DELISTED]
    assert list(d.date) == [close.index[4]]
    assert d.iloc[0].price == pytest.approx(11.0 * (1 - 5e-4))
    assert res.equity.notna().all()


def test_missing_spy_bar_skips_sweep():
    close, ent, ext, spy = _panel({"AAA": [10.0] * 3}, [100.0, np.nan, 100.0])
    res = run_replica(close, ent, ext, spy)
    assert res.trades.query("symbol == 'SPY'").date.min() == close.index[2]
    assert res.equity.notna().all()


def test_drawdown_halt_stops_all_trading():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 7}, [100, 100, 100, 84, 84, 84, 100], entries={"AAA": [3, 4]}
    )
    res = run_replica(close, ent, ext, spy)
    assert res.trades[res.trades.date.isin(close.index[3:6])].empty
    assert res.halted_days == 3
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `scripts/run_tests.sh tests/lab/test_replica.py -v`
Expected: FAIL with `ModuleNotFoundError: ggTrader.lab.replica`.

- [ ] **Step 3: Implement `replica.py`** following the algorithm above.
  - Positions are a plain `dict[str, {"shares", "cost", "fill_date"}]`.
  - SPY is tracked separately from the stock positions, the same way it is in live.
  - Put the module docstring on the first line: "Event-loop replica of the live paper trader's
    portfolio rules, for parity backtests."
- [ ] **Step 4: Run the tests and confirm they pass**

Run: `scripts/run_tests.sh tests/lab/test_replica.py -v`
Expected: 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/replica.py tests/lab/test_replica.py
git commit -m "feat(lab): event-loop replica of the live trader's portfolio rules"
```

---

### Task 5: Signal panel and live parity check

**Files:**
- Modify: `src/ggTrader/lab/replica.py` (add `SignalPanel` and `build_signal_panel`)
- Create: `scripts/replica_parity_check.py`
- Test: `tests/lab/test_replica.py` (add one test)
- Output: `docs/research/_replica_parity_<YYYYMMDD>.json`, dated by the run date

**Interfaces:**
- Produces `class SignalPanel(NamedTuple)` with fields `close: pd.DataFrame`,
  `entries: pd.DataFrame`, `exits: pd.DataFrame` and `spy_close: pd.Series`.
- Produces `build_signal_panel(ohlcv: pd.DataFrame, eval_start: pd.Timestamp,
  eval_end: pd.Timestamp, eligible_fn: Callable[[pd.Timestamp, pd.DataFrame], list[str]],
  sweep_symbol: str = "SPY") -> SignalPanel`.
  - `ohlcv` is the `(symbol, field)` frame that `load_ohlcv` returns, including SPY.
  - Strategy: `EnsembleSignal(LabConfig(min_history_bars=60))` with no other arguments. Its
    defaults *are* the live config.
  - Plans: `{d: strategy.select(d, stocks, eligible_fn(d, stocks.loc[:d])) for d in
    rebalance_dates(stocks.index, eval_start, eval_end)}`, where `stocks` is `ohlcv` without
    the sweep symbol.
  - Signals: `targets = strategy.to_targets(plans, stocks)`.
  - Close: `close` uses the `extract_close` helper that `ensemble.py` imports, reindexed to
    `targets.entries`.

- [ ] **Step 1: Write the failing mask test**
  `test_build_signal_panel_masks_non_members`. Fixture: symbols `AAA` and `BBB` share the same
  path, 300 flat bars with noise σ=0.1% followed by a 4-bar drop of 18%, with constant volume.
  SPY is flat. The test passes `eligible_fn = lambda asof, past: ["AAA"]` and asserts:
  - `panel.entries["AAA"].any()`
  - `not panel.entries["BBB"].any()`

  If the drop doesn't fire 2 voters, deepen it. The assertion is the point of the test, not
  the exact numbers.
- [ ] **Step 2: Run the test and confirm it fails**

Run: `scripts/run_tests.sh tests/lab/test_replica.py::test_build_signal_panel_masks_non_members -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `build_signal_panel`, then run the test**

Expected: PASS.

- [ ] **Step 4: Write `scripts/replica_parity_check.py`.** It takes `main()` and argparse, with
  `--since 2026-09-29` and `--out`.
  - **Load live orders.** Read `paper_trades` (`run_date ≥ since`) through
    `ggTrader.lab.persist.get_engine()`, SELECT only.
    - Strategy orders are every row whose `reason` is not in `{SWEEP_TRADE_REASON,
      CATASTROPHE_STOP_REASON}`.
    - Print `SELECT DISTINCT reason` first, so any new reason value is visible.
  - **Build the panel.**
    - Data: `load_ohlcv(members + ["SPY"], since − 500 days, today)`.
    - `eligible_fn` returns the live universe:
      `[normalize_yf_ticker(m) for m in universe_members_asof("sp500", today)]`. This is the
      2026-06-02 snapshot that live uses, deliberately *not* the point-in-time list.
  - **Explain each live order.** For each live run date *d*, the signal bar is the last panel
    date strictly before *d*.
    - A BUY of `sym` is explained if `entries.at[bar, sym]` is true.
    - A SELL is explained if either of these holds:
      - `exits.at[bar, sym]` is true. This is the regular last-bar path, and it covers an exit
        on the fill day's own bar.
      - Some exit for `sym` falls on a bar strictly after its last live BUY date and on or
        before *bar*. This is the catch-up path.

      This is the same union as replica rule 5.
    - Also record, for each *d*, how many entry candidates the panel had against how many buys
      live made. That shows the A→Z truncation.
  - **Output.** Write a JSON with per-order `explained` flags, counts and the percentage
    explained.
- [ ] **Step 5: Run the parity check**

Run: `.venv/bin/python scripts/replica_parity_check.py --since 2026-09-29 --out docs/research/_replica_parity_$(date +%Y%m%d).json`
Expected: 100% of strategy orders explained. That is the brief's stop condition.
- Below 100%, investigate each unexplained order and write the cause into the JSON's `notes`.
- Fix a replica bug if that's the cause.
- Do **not** start Task 6 until every mismatch is either fixed or documented as a known
  live-only effect (for example, a partial intraday bar).

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/replica.py tests/lab/test_replica.py scripts/replica_parity_check.py \
        docs/research/_replica_parity_*.json
git commit -m "research(audit): signal panel + replica parity check against live orders"
```

---

### Task 6: Audit driver (data checks, runs, verdict)

**Files:**
- Create: `scripts/production_replica_audit.py`
- Test: `tests/scripts/test_production_replica_audit.py`
- Output: `docs/research/_production_replica_results.json`
- Output: `docs/research/_production_replica_curves.csv`

**Interfaces:**
- **Consumes:** `build_signal_panel` and `run_replica` (Tasks 4–5), and `excess_stats` and
  `passes_pretax_bar` (Task 3).
- **Module constants:**

  | Constant | Value |
  |---|---|
  | `WINDOWS` | `{"primary": ("2021-01-31", "2026-04-30"), "holdout": ("2010-01-31", "2020-12-31")}` |
  | `WARMUP_DAYS` | 420 |
  | `SEAM_THRESHOLD` | 0.40 |
  | `HOLDOUT_COVERAGE_FLOOR` | 0.70 |
  | `ARMS` | see below |

  `ARMS` is a `dict[str, dict]`, the arm name followed by its settings:

  | Arm | Settings |
  |---|---|
  | `primary` | lag 1, stocks 5 bp, T-bill cash |
  | `lag0` | lag 0 |
  | `cost3` | stocks 3 bp |
  | `cost10` | stocks 10 bp |
  | `cash0` | cash at 0% |

- **Helper functions:**
  - `tbill_daily(index: pd.DatetimeIndex) -> pd.Series`: loads `load_fred_series("TB3MS",
    ...)`, which is in annual percent, forward-fills it to `index` and returns
    `(1 + r/100) ** (1/252) − 1`.
  - `exposure_matched(spy_close: pd.Series, exposure: pd.DataFrame, cash_rate: pd.Series,
    start: float) -> pd.Series`: the daily return is `(stocks + spy)` weight from *t−1* × SPY
    return + `cash` weight from *t−1* × cash rate, compounded from `start`.
  - `seam_scan(close: pd.DataFrame, threshold: float) -> list[dict]`
  - `coverage(close: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[int, float]`: for each
    date, take the share of `universe_members_asof("sp500", date)` (normalized) that has a
    finite close. Average those shares by year.
  - `verdict(results: dict) -> dict`, which returns `{"verdict": "GO"|"NO-GO"|"INCONCLUSIVE",
    "stage": "pretax_pass"|"pretax_fail"|"stopped", "reasons": [...]}` and encodes brief §4.
    Check these conditions in order:
    1. Primary window has a stop flag → `INCONCLUSIVE` / `stopped`.
    2. Primary arm fails `passes_pretax_bar` → `NO-GO` / `pretax_fail`. A primary fail is
       conclusive on its own.
    3. Holdout has a stop flag, or mean coverage below 0.70 → `INCONCLUSIVE` / `stopped`.
    4. Holdout fails → `NO-GO` / `pretax_fail`.
    5. Otherwise → stage `pretax_pass`. Task 7 decides the final verdict.

- [ ] **Step 1: Write the failing tests.** All three are pure. They build `results` dicts by
  hand, with no DB access.
  - **`test_verdict_requires_both_windows`:**
    - Primary passes and holdout fails → `NO-GO`.
    - Both pass → `stage == "pretax_pass"`.
    - Primary passes and holdout coverage is 0.65 → `INCONCLUSIVE`.
    - Primary fails and holdout coverage is 0.65 → `NO-GO`.
    - A stop flag on the primary window → `INCONCLUSIVE` / `stopped`.
  - **`test_tbill_daily_converts_annual_percent`:** a flat 4.0% gives
    `(1.04) ** (1/252) − 1` within 1e-12.
  - **`test_exposure_matched_all_cash_earns_cash_rate`:** exposure of `{stocks: 0, spy: 0,
    cash: 1}` compounds at exactly the cash rate.
- [ ] **Step 2: Run the tests and confirm they fail**

Run: `scripts/run_tests.sh tests/scripts/test_production_replica_audit.py -v`
Expected: FAIL (import).

- [ ] **Step 3: Implement the driver**

  **Per window:**
  1. Point-in-time universe: `equity_universe_between(start, end, "sp500")`.
  2. Load data with `load_ohlcv(universe + ["SPY"], start − 420 days, end, use_negative_cache=True)`.
  3. Build the panel with `build_signal_panel(..., eligible_fn=lambda asof, past:
     eligible_at(asof, past, LabConfig(min_history_bars=60), universe="sp500")[0])`.
  4. Slice the panel so the first trading day ≥ `start` is the first decision day. Keep one
     bar before it, which is the lag bar, and run through `end`.
  5. Run the data checks:
     - **Seam scan.** Flag every bar where the absolute value of the adjusted one-day return
       is ≥ 0.40. Compare each flagged bar against a fresh `yfinance.download(sym,
       auto_adjust=True)`. A bar is an *artifact* if its return differs from the fresh one by
       more than 5 pt, or if the fresh data is missing.
     - **Coverage.**
  6. Run every arm in `ARMS`.
  7. For each arm, compute `curve_stats`, `excess_stats` against SPY buy-and-hold and against
     the exposure-matched passive, the mean exposure, trade counts, `halted_days`,
     `truncated_days`, and `years_for_t(IR)`.
  8. Raise the stop flag `tainted_trade` if any replica trade falls within 5 bars of an
     artifact bar of the same symbol.

  **Output and control:**
  - Rewrite the JSON after each `(window, arm)` and skip arms already recorded unless `--force`
    is set.
  - Append the curves to the CSV.
  - Flags: `--window {primary,holdout,all}`, `--max-symbols N` (smoke test only; the JSON
    must then carry `"citable": false`), `--out` and `--force`.
- [ ] **Step 4: Run the tests and confirm they pass**

Run: `scripts/run_tests.sh tests/scripts/test_production_replica_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke run**

Run: `.venv/bin/python scripts/production_replica_audit.py --window primary --max-symbols 40 --out /tmp/replica_smoke.json`
Expected: all five arms are recorded, `"citable": false`, equity has no NaN, and
`exposure.cash ≥ 0`.

- [ ] **Step 6: Full primary run**

Run: `.venv/bin/python scripts/production_replica_audit.py --window primary`
Expected: it completes in under 30 minutes, and the JSON holds `primary/*` arms with
`seam_scan` and `stop_flags`.

- [ ] **Step 7: Full holdout run.** The first run downloads a lot, so it may take 20–60
  minutes. Run it in the background:

```bash
nohup .venv/bin/python scripts/production_replica_audit.py --window holdout \
  > logs/production_replica_holdout.log 2>&1 &
```

Expected: the log ends with the `verdict` block, and the JSON holds `holdout/*`, `coverage`
and `verdict`. If it crashes, re-run the same command; it resumes from the checkpoint.

- [ ] **Step 8: Commit**

```bash
git add scripts/production_replica_audit.py tests/scripts/test_production_replica_audit.py \
        docs/research/_production_replica_results.json docs/research/_production_replica_curves.csv
git commit -m "research(audit): production-replica runs, primary + holdout, pre-registered verdict"
```

---

### Task 7 (conditional): after-tax replay

**Trigger:** run this task only if the Task 6 JSON has `verdict.stage == "pretax_pass"`.
Otherwise skip it and record "after-tax not run: pre-tax bar failed (brief §4)" in the report.

**Files:**
- Create: `src/ggTrader/lab/tax.py`
- Modify: `scripts/static_leverage_aftertax.py`, so it imports `Lot` and `Book` from there
- Modify: `src/ggTrader/lab/replica.py`
- Modify: `scripts/production_replica_audit.py`
- Test: `tests/lab/test_tax.py`

**Interfaces:**
- **Produces** `TaxBook`. This is `Book` from `scripts/static_leverage_aftertax.py:114-232`,
  moved and generalized:
  - `lots` becomes a `defaultdict(list)` keyed by any symbol.
  - `buy` and `sell` take a `cost_bps` argument instead of the module-level `COST_BPS`.
  - `distributions(..., taxed_as)` comes from Task 9.
- **Produces** a new `run_replica` parameter, `tax: TaxBook | None = None`.
  - When it is set, the `TaxBook` holds the positions and cash.
  - Tax is computed at each calendar year-end and paid from cash on the first trading day of
    the following April.
  - Everything is liquidated on the final bar.
- **Produces** an `aftertax_primary` arm in the driver, at the 24/15 bracket, plus a
  benchmark: SPY buy-and-hold through the same `TaxBook`.

- [ ] **Step 1: Write the failing tests**
  - `test_short_hold_taxed_at_st_rate`: a 10-day round trip with a $100 gain gives
    `year_st == 100`, and the tax due after `year_end()` is 24.
  - `test_long_hold_taxed_at_lt_rate`: a 400-day hold with a $100 gain gives tax of 15.
- [ ] **Step 2: Run the tests and confirm they fail**

Run: `scripts/run_tests.sh tests/lab/test_tax.py -v`
Expected: FAIL (import).

- [ ] **Step 3: Implement the move.** The default run of `static_leverage_aftertax.py` must still
  reproduce the committed `_static_leverage_aftertax_results.json` headline numbers exactly.
  Check that with `--out /tmp/a23_regression.json` and diff the two files' headline numbers.
- [ ] **Step 4: Run the tests and confirm they pass**

Run: `scripts/run_tests.sh tests/lab/test_tax.py -v`
Expected: PASS.

- [ ] **Step 5: Wire up the after-tax run.** Add the `tax` parameter and the driver arm, re-run
  both windows with `--force` restricted to the new arm, and compute the final verdict using
  brief §4 item 3.
- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/tax.py src/ggTrader/lab/replica.py \
        scripts/static_leverage_aftertax.py scripts/production_replica_audit.py \
        tests/lab/test_tax.py docs/research/_production_replica_results.json
git commit -m "research(audit): after-tax replay for the production replica"
```

---

### Task 8: Re-runs on the corrected harness

**Depends on:** Task 2 (the DSR fix) and Task 3 (`excess_stats`).

**Files:**
- Modify: `scripts/anchor_fix_reproduction_wfo.py:93` (`run_core`)
- Create: `scripts/audit_reruns_wfo.py`
- Test: `tests/lab/test_fomc_drift.py` (add one test)
- Output: `docs/research/_audit_reruns_<YYYYMMDD>.json`

**Interfaces:**
- **Produces** `run_core(eval_start: str, eval_end: str, strategy: str = CORE_STRATEGY,
  return_equity: bool = False) -> dict`. When `return_equity` is True, the result also carries
  `"oos_equity": wfo_result.oos_equity` and `"spy_close": spy_close`. Both are `pd.Series` and
  must be popped before any JSON dump. The default behaviour does not change.

- [ ] **Step 1: Write the failing fomc_drift test**
  `test_event_signal_session_semantics`.
  - Fixture: 10 business days with close-time timestamps (16:00 UTC) and an event on day 5.
  - `_event_signals(index, ["TLT"], [event], entry_offset_days=1, hold_days=0)` gives an entry
    on day 4 and an exit on day 5.
  - `in_position_mask(...)` then covers exactly the day-5 return.
  - This pins which session each parameter pair captures on a correctly dated tape.
- [ ] **Step 2: Run the test.** It should pass on current code, because this pins existing
  behaviour.

Run: `scripts/run_tests.sh tests/lab/test_fomc_drift.py -v`
Expected: PASS. If it fails, stop: the tape assumption in the brief is wrong, so report back
before running anything.

- [ ] **Step 3: Add `return_equity` to `run_core`.**
- [ ] **Step 4: Write `scripts/audit_reruns_wfo.py`.** It takes `main()`, argparse and
  `--strategies` (default `xs_momentum dual_momentum fomc_drift`). It checkpoints the JSON after
  each strategy and skips strategies already recorded unless `--force` is set.
  - **Momentum:** for each of `xs_momentum` and `dual_momentum`:
    1. Run `run_core("2021-01-31", "2026-04-30", strategy=name, return_equity=True)`.
    2. Build the SPY curve from `spy_close` reindexed to the OOS dates.
    3. Compute `excess_stats(oos, spy_curve)`.
    4. The verdict is `"NO-GO stands"` unless `passes_pretax_bar` holds. If it holds, the
       verdict is `"pretax pass — needs its own after-tax brief"`.
    5. Also record the fold winners and the corrected DSR gate counts.
  - **fomc_drift, walk-forward:** mirror the `ggTrader.lab.cli` `--wfo` branch
    (`cli.py:188-205`) with strategy `fomc_drift`, universe `fomc_treasury`, eval
    `2012-01-01`→`2026-04-30`, and the strategy's `sweep_params` grid.
  - **fomc_drift, event study:** for the pair `(entry_offset_days=1, hold_days=0)`, which is the
    original study's day-before entry (`2026-07-20-fomc-drift-nogo.md` §1), run
    `run_event_study_from_signals`.
    - Gross: on the raw returns.
    - Net: subtract 0.0010 from the return of each event's first in-position day, as in
      `2026-07-20-fomc-drift-nogo.md:179-187`.
    - Report the other five pairs as context only.
    - The verdict is `"NO-GO stands"` unless net t ≥ 3.0 **and** the walk-forward WFE ≥ 0.50.
- [ ] **Step 5: Run it**

Run: `.venv/bin/python scripts/audit_reruns_wfo.py`
Expected: the JSON holds all three strategies with `verdict` fields. Momentum takes about 25
minutes per strategy.

- [ ] **Step 6: Commit**

```bash
git add scripts/anchor_fix_reproduction_wfo.py scripts/audit_reruns_wfo.py \
        tests/lab/test_fomc_drift.py docs/research/_audit_reruns_*.json
git commit -m "research(audit): momentum + fomc_drift re-runs on corrected harness"
```

---

### Task 9: A23 distribution-tax bounds

**Files:**
- Modify: `scripts/static_leverage_aftertax.py`, at `Book.distributions` (:189-196), `run`
  (:235) and `main` (:357)
- Test: `tests/scripts/test_static_leverage_aftertax.py`
- Output: `docs/research/_static_leverage_aftertax_dist_bounds.json`
- Modify: `docs/research/2026-09-30-static-fractional-leverage.md` (add a dated addendum)
- Modify: `docs/next_steps.md` (one line under the A23 "Rate clause")

**Interfaces:**
- **Produces** `Book.distributions(s: str, per_share: float, px: float, date: pd.Timestamp,
  taxed_as: str = "lt") -> None`. `"st"` adds to `year_st` and `"lt"` adds to `year_lt`.
- **Produces** a new `run(...)` kwarg, `dist_tax: dict[str, str] | None = None`. It maps each
  symbol to its treatment and defaults to all `"lt"`, which is the current behaviour.
- **Produces** two CLI flags, `--sso-dist-tax {lt,st}` and `--bil-dist-tax {lt,st}`. Both
  default to `lt`.

- [ ] **Step 1: Write the failing test**

```python
def test_distribution_taxed_as_short_term_goes_to_st_bucket():
    from scripts.static_leverage_aftertax import Book

    b = Book(st_rate=0.24, lt_rate=0.15, wash=False)
    b.cash = 1000.0
    b.buy("SSO", 1000.0, 10.0, pd.Timestamp("2024-01-02"))
    b.distributions("SSO", 1.0, 10.0, pd.Timestamp("2024-06-03"), taxed_as="st")
    # Shares held before reinvestment: $1000 at 10 * (1 + 1 bp) (COST_BPS = 1.0), times $1.
    assert b.year_st == pytest.approx(1000.0 / 10.001, rel=1e-12)
    assert b.year_lt == 0.0
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `scripts/run_tests.sh tests/scripts/test_static_leverage_aftertax.py -v`
Expected: FAIL (`unexpected keyword 'taxed_as'`).

- [ ] **Step 3: Implement the parameter, the kwarg and the flags.** Then run the test.

Expected: PASS.

- [ ] **Step 4: Regression check.** The default run must reproduce the committed numbers.

Run: `.venv/bin/python scripts/static_leverage_aftertax.py --main-only --out /tmp/a23_default.json`
Expected: the 70/30 after-tax CAGR and SPY's after-tax CAGR at 24/15 equal the values in
`_static_leverage_aftertax_results.json` (12.00% / 10.40%) to the stored precision.

- [ ] **Step 5: Bounds run at today's bills**

Run: `.venv/bin/python scripts/static_leverage_aftertax.py --main-only --rf-offset 0.025 --sso-dist-tax st --bil-dist-tax st --out docs/research/_static_leverage_aftertax_dist_bounds.json`
Expected: a new after-tax margin at +2.5 pt bills. Compare it with the +0.80 pt/yr from
`_static_leverage_aftertax_rf_plus0.025.json`.

- [ ] **Step 6: Write it up**
  - Add the addendum to the A23 report: "Distribution-tax bounds (date)". It gives the margin
    under the all-long-term treatment (+0.80) and the all-ordinary treatment (new number).
    State that SSO's real character lies between the two.
  - Add one line to `next_steps.md` under "Rate clause": "pre-entry information for the owner;
    terms unchanged".
- [ ] **Step 7: Commit**

```bash
git add scripts/static_leverage_aftertax.py tests/scripts/test_static_leverage_aftertax.py \
        docs/research/_static_leverage_aftertax_dist_bounds.json \
        docs/research/2026-09-30-static-fractional-leverage.md docs/next_steps.md
git commit -m "research(A23): bound after-tax margin with distributions taxed as ordinary income"
```

---

### Task 10: Report, record corrections and snapshot

**Files:**
- Create: `docs/research/<run-date>-production-replica-audit[-nogo].md`. Use the `-nogo`
  suffix if the verdict is NO-GO.
- Modify, adding a dated "Correction (2026-10-xx)" note only and leaving the body untouched:
  - `docs/research/2026-08-19-anchor-fix-reproduction.md`: the 1.14 run was pinned to the
    re-run's window, so eval-end drift did not cause the 1.14→0.68 drop. Point to the 08-18
    fixes instead, and label this as inferred.
  - `docs/research/2026-09-24-core-spy-blend-keep.md`: KEEP tested walk-forward-picked
    parameters and same-close fills, not the live config. This report supersedes it.
- Modify:
  - `docs/next_steps.md`: the freeze-lift recommendation from brief §5, marked as the owner's
    decision.
  - `docs/changelog.md`: an entry for the replica, the significance helpers and the audit.
  - `AGENTS.md`: the strategy registry count line, "36 entries as of 2026-08-21", becomes the
    current count and date.
- Regenerate: `docs/research/RESEARCH_SNAPSHOT.md`, using the `research-snapshot` skill.

- [ ] **Step 1: Write the report** following `docs/research/TEMPLATE-research-report.md`, all
  six sections.
  - Every number must cite a JSON key: `_production_replica_results.json`, the parity JSON,
    `_audit_reruns_*.json` or the bounds JSON.
  - It must include:
    - the verdict and the reason for it
    - the lag-0 versus lag-1 gap, which is the cost of the one-day delay
    - selection skill against the exposure-matched passive
    - halted and truncated days
    - holdout coverage and the survivorship caveat
    - the live design findings: A→Z selection, the 15% halt in bear markets, and the
      sweep's same-day churn
- [ ] **Step 2: Second read.** Run the `research-report-review` skill on the report and apply
  its corrections.
- [ ] **Step 3: Apply the record corrections and doc edits** listed under Files.
- [ ] **Step 4: Regenerate the snapshot.** Run the `research-snapshot` skill. The snapshot must
  add this audit, a DSR caveat that past gate counts are void, and the studies currently
  missing from it: cross-asset, A23 and A24.
- [ ] **Step 5: Verify the suite and lint**

Run: `scripts/run_tests.sh -q && ruff check . && ruff format --check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add docs/research/*production-replica-audit*.md docs/research/2026-08-19-anchor-fix-reproduction.md \
        docs/research/2026-09-24-core-spy-blend-keep.md docs/next_steps.md docs/changelog.md \
        AGENTS.md docs/research/RESEARCH_SNAPSHOT.md
git commit -m "research(audit): production-replica audit report + record corrections"
```
