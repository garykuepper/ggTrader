# CLI Reference Guide

The `ggt` command-line tool is your command center. You will use it to run strategy simulations, view database statistics, download historical price data, and launch paper trading.

---

## Command Summary

| Command | Purpose (Plain English) |
|---|---|
| `ggt lab` | Run trading simulations (backtests) to see how strategies would have performed in the past. |
| `ggt paper` | Run virtual trading (paper trading) using real-time data and fake money. |
| `ggt ingest` | Removed (it never worked). See §3. |
| `ggt db` | Run diagnostics, clean up, or manage the database. |

---

## Study and test tooling (not `ggt` subcommands)

Study code lives with its study, in the study's own `code/` folder (shared helpers in the category's `_lib/` folder), not in `src/`. These scripts keep that tree honest; each has `--help`.

```bash
.venv/bin/python scripts/study.py new CATEGORY SLUG --kind screen --placement 9   # scaffold a study
.venv/bin/python scripts/study.py check STUDY_FOLDER                              # naming, card, drivers, imports
.venv/bin/python scripts/study.py repro STUDY_FOLDER                              # re-run a pinned study in a scratch clone
.venv/bin/python scripts/study.py close STUDY_FOLDER --verdict NO-GO              # write PIN.json, print the index row
.venv/bin/python scripts/verify_pins.py --summary   # every code hash pinned in the evidence still resolves
scripts/run_tests.sh            # fast core (default)
scripts/run_tests.sh --full     # everything but `integration`: the merge, deploy and vectorbt-upgrade gate
scripts/run_tests.sh --studies  # the study tier: tests inside each study's `code/tests/` and each category's `_lib/tests/`
```

`research/CODEMOVED.csv` lists every file that left `src/`, `scripts/` or `tests/` for a study (old path, new path, hashes before and after).

## Prospective intraday practice journal

`intraday_observe.py` (in `research/intraday/spy-momentum-prospective-observer/code/`) is a standalone research recorder and hypothetical journal.
It does not submit orders or use the production paper account. The standalone observer and its four user timers were activated on October 27, 2026.
First observation is October 30; see the [activation record](../research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md). Independent review findings were
corrected and regression-tested; the final full capped suite passes.

Safe offline commands:

```bash
.venv/bin/python research/intraday/spy-momentum-prospective-observer/code/intraday_observe.py --help
.venv/bin/python research/intraday/spy-momentum-prospective-observer/code/intraday_observe.py plan
.venv/bin/python research/intraday/spy-momentum-prospective-observer/code/intraday_observe.py rehearse --work-dir /tmp/ggtrader-observer-demo
```

The [operator runbook](../deploy/intraday_observer/OPERATOR_RUNBOOK.md) describes
the dedicated frozen runtime, persistent private credentials, separate initialization
and activation steps, daily health checks and calendar amendments. Daily jobs are separately installed; invoking help or rehearsal does not install
them. `init` prints the actual on-disk manifest hash; a corrupt ledger
report is unresolved and has no return statistics. Resume cannot waive ongoing
corruption.

`rehearse` requires a new empty path and runs three synthetic sessions with fake
prices and a simulated clock. It saves a practice report; it proves neither real
source access nor strategy profitability. No credentials or network are needed.

`init` prepares a new state directory from an explicit calendar snapshot, protocol
hash, activation date and assumed source delay. It does not enable collection.
`inputs`, `decide`, `archive` and `recover` are manual session stages; `revise`
retains corrections separately. `verify` checks evidence and `report` produces
local descriptive JSON. Use each subcommand's `--help` for required parameters.
Live activation requires an owner approval record and a manifest-bound readiness
JSON supplied with `--readiness-file`; a label or checksum alone is insufficient.
The readiness file records verified source access/delay, calendar provenance and
synthetic rehearsal. Its assertions require operator-checked source evidence;
the offline build does not perform or certify that real-source preflight. Do
not invoke `activate` as part of trying the offline demo. Four separate user timers are installed for the activated study.
Unit templates and the operator runbook live under `deploy/intraday_observer/`.
The installed timers run the frozen release cut at commit `280d4a0`, which still keeps `intraday_observe.py` and `intraday_observer_job.py` in the `scripts` folder (the layout before the move); the linked runbook describes that release and is unchanged. The commands in this section use the study's own copy of the code.
The calendar-aware timer entry point is:

```bash
.venv/bin/python research/intraday/spy-momentum-prospective-observer/code/intraday_observer_job.py --help
```

It accepts `--stage recover|inputs|decide|archive` and `--state-dir PATH`;
non-session and out-of-study dates are no-ops, except final next-session recovery.
Do not run stages against live state before the separate activation approval.
Read [current readiness evidence](../research/intraday/spy-momentum-prospective-observer/2026-10-26-intraday-observer-readiness.md)
for verified source samples and remaining evidence limitations.
The issuer importer supports `--date-format mdy` and `--col-cusip CUSIP` for
State Street's multi-fund workbook. Missing identity mapping or capital-gain
columns keep the comparison incomplete; blank gains are not converted to zero.
See the [prospective protocol](../research/intraday/spy-momentum-prospective-observer/2026-10-25-intraday-prospective-observation.md).

## Intraday Stage 0A coverage probe

```bash
.venv/bin/python research/intraday/spy-momentum-feasibility/code/intraday_coverage_probe.py --help
.venv/bin/python research/intraday/spy-momentum-feasibility/code/intraday_coverage_probe.py
```

The standalone research script uses inherited `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` credentials for read-only paper-calendar and market-data
requests. It does not load `.env`. Its fixed SPY sample, rate limits and evidence
rules are defined in the [Stage 0A contract](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-stage0a-contract.md).
It stores timestamps, volumes, auction venues and response hashes, discarding
prices. The default artifact is `research/intraday/spy-momentum-feasibility/_intraday_probe.json`; `--output`
selects a different artifact path. Existing artifacts are never overwritten.
The hard request deadline uses Linux main-thread signals; run this as a standalone
CLI rather than from a background Python thread.
A clean sample does not establish full-window coverage or a trading edge.
Stage 0B and strategy backtesting require their separate authorization.

## Intraday r8 research pipeline

```bash
.venv/bin/python research/intraday/spy-momentum-feasibility/code/run.py --help
.venv/bin/python research/intraday/spy-momentum-feasibility/code/run.py plan
```

The standalone script implements the [approved r8 proxy protocol](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-feasibility-r8.md).
The historical r8 study is now closed UNRESOLVED (data); do not re-pull or score
its frozen dataset. See the [completed investigation](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-gap-investigation.md).
The commands below describe the preserved implementation, not permission to reopen it.
Help, no command and `plan` need no credentials, network or writes. Approval 1
covers mocked implementation only; real collection and candidate scoring require
approval 2 after code review. The intended data are SPY raw SIP bars, the paper
calendar and corporate-action metadata; no auctions, IEX, broker orders or DB.

For a later approved run, `collect --output-dir <new-directory>` collects/resumes
immutable chunks. `validate --input-dir <directory> --output <validation.json>`
performs signal-blind endpoint/provenance checks. `score --input-dir <directory>
--validation <validation.json> --output <result.json>` requires a matching PASS,
rehashes inputs and claims the collection directory before computing. Existing
results and a claimed study cannot be overwritten or rescored by changing the
output path. A crash after claiming remains refused, preserving the audit trail.

Default validation/result paths are `research/intraday/spy-momentum-feasibility/_intraday_momentum_validation.json`
and `research/intraday/spy-momentum-feasibility/_intraday_momentum_results.json`. The default collection input
is `data/external/intraday_momentum_r8`; credentials are inherited through
`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, never read from `.env` by this CLI.
The HTTP deadline uses Linux main-thread signals. The exact future source
manifest and outstanding coverage limitations must be reviewed before approval 2.

## A23 advisory rebalance check

`scripts/a23_rebalance_check.py` checks the owner's 70% SPY / 30% SSO equity
allocation. It proposes trades only after the last NYSE session of a month if
SSO is outside 22–38%. It then restores SSO to 30%. Trades are indicative for
the next session; the script never sends an order. Cash is used first. Sale
estimates use highest-basis long-term lots before short-term lots and flag
possible wash sales for review.
The displayed SSO weight uses SPY plus SSO market value; cash is available
funding for the proposed rebalance.

```bash
.venv/bin/python scripts/a23_rebalance_check.py \
  --lots docs/examples/a23_lots_example.csv \
  --prices docs/examples/a23_prices_example.csv --asof 2026-09-30
```

The lots CSV needs `symbol,shares,cost_basis_total,acquired_date`; the optional
price CSV needs `symbol,close,date` with one unadjusted close per symbol. Without
`--prices`, the script fetches closes from yfinance and prints their dates. Both
price dates must equal the last completed NYSE session on or before `--asof`,
including when calendar month-end falls on a weekend or holiday. Running on
the month-end session before its close causes an error. `--cash` and
`--new-cash` are available dollars; unused cash is printed. `--st-rate` and
`--lt-rate` are fractions (defaults `0.24` and `0.15`). Use `--json` for machine
output. Off-cycle checks report the current SSO weight and propose no trade.
Wash-sale warnings inspect the supplied open lots and proposed buys; review
other purchases and reinvestments within 30 days before or after a loss sale.

---

## 1. ggt lab (Simulation Engine)

This command runs trading strategies through historical data to test their performance, on the
vectorbt 1.2.0 engine with the Rust simulator (`ggTrader.engine`). It always reads the frozen tape
(`data/snapshots/`, a hashed read-only Parquet file) and never writes to the price tables.

### How to use it:
```bash
ggt lab --strategy <strategy_name> --wfo [options]     # walk-forward one strategy
ggt lab --blend "<strategy>@<universe>,..." [options]  # walk-forward several sleeves and blend them
```

One of `--wfo` or `--blend` is required.

> **Removed in the engine rebuild (P5c, 2026-11-06).** The hand-built legacy engine is deleted, so
> there is only one path. `--engine {legacy,vbt}` is gone (the vectorbt engine is the only one),
> as are `--sweep`, `--max-sector-count` and the plain no-mode run (a single simulation that wrote
> to `lab_runs`). A one-fold walk-forward replaces the last. Strategies that were not ported also
> fail here, with a pointer to git tag `legacy-engine-vbt0285` (see below). Old reports and
> commands that use the removed flags ran on the legacy engine (vectorbt 0.28.5) and are not
> re-run; they stay reproducible at the tag with `scripts/legacy_env.sh`.

### Required Arguments:

- `--strategy`: The name of a ported strategy (see below). Needed with `--wfo`; `--blend` takes its strategies from the `strategy@universe` list.

---

### Choosing a Strategy

Ten strategies are ported to the engine contract (`ggTrader.lab.strategies`, one registry in
`registry.py`). They are of two types:
- **Weight Strategies** (`kind="weights"`): decide **how much** of your money goes into each stock.
- **Signal Strategies** (`kind="signals"`): decide **when** to buy and sell.

#### Weight Strategies
| Strategy Name | What it does in plain English |
|---|---|
| `xs_momentum` | **Cross-Sectional Momentum:** Ranks all stocks by their returns over a past lookback window and splits money equally among the top performers. |
| `cross_asset_sleeve` | **SPY plus a cross-asset sleeve:** 80% SPY and 20% split across TLT, GLD and PDBC, optionally trend-timed (the static arm is a paper-shadow GO). |
| `retail_attention` | **Retail attention:** ranks stocks by search-interest spikes (paused, no verdict; the interest table is empty). |

#### Signal Strategies
| Strategy Name | What it does in plain English |
|---|---|
| `ema_cross` | **EMA Crossover:** buys when a fast moving average crosses above a slow one, sells when it crosses back. |
| `bb_reversion` | **Bollinger Band Mean Reversion:** buys when the price falls below the lower band, sells when it returns to the middle band. |
| `rsi_reversion` | **RSI Mean Reversion:** buys when the Relative Strength Index drops below an oversold level, sells when it recovers. |
| `macd_divergence` | **MACD Divergence:** buys new price lows where the MACD momentum is no longer falling. |
| `volume_bb_reversion` | **Volume-Confirmed BB Reversion:** the Bollinger entry, only on a volume spike. |
| `ensemble` | **Voting Strategy:** the five voters above (not `ema_cross` alone) vote; a trade needs `min_agree` of them. This is the deployed strategy. |
| `conviction_bb` | **Band-Distance Sizing:** a Bollinger strategy that invests more the further the price is below the lower band. |

#### Adding your own

Drop a module that implements the strategy contract into `src/ggTrader/lab/strategies/` and `--strategy <its name>` finds it (the registry scans the package; no registration step). A malformed module raises an error naming the file. Worked example: [adding_a_strategy.md](adding_a_strategy.md).

#### Strategies that are not ported (28)

`commodity_trend`, `congress_trades`, `dual_momentum`, `ensemble_conviction`, `ensemble_ic`,
`ensemble_kelly`, `fomc_drift`, `fx_hedge_overlay`, `headline_sentiment`, `idio_vol`,
`index_deletion_fade`, `insider_cluster_buy`, `leveraged_rotation_{nasdaq100,russell2000,sp500}`,
`leveraged_trend_{nasdaq100,russell2000,sp500}`, `max_effect`, `month_end_treasury`,
`mtf_reversion`, `overnight_gap`, `pairs_stat_arb`, `pead`, `short_interest`,
`short_volume_ratio`, `treasury_curve`, `wfo_tournament`.

These are closed NO-GO or untested research (decision OD11(b) of the engine spec). Their code and
tests were removed from the main tree in P5c; every research report is kept. They are
reproducible at git tag `legacy-engine-vbt0285`:
`scripts/legacy_env.sh /path/to/legacy-engine`, then run the legacy `ggt lab` from that tree with
its own interpreter. `ggt lab --strategy <one of them>` exits with that instruction. Check
`research/RESEARCH_SNAPSHOT.md` for each verdict before spending a run on one. The authoritative
lists are `REGISTRY` and `LEGACY_ONLY_STRATEGIES` in `src/ggTrader/lab/strategies/registry.py`.

---

### Optional Arguments:

| Flag | Default | Description (Plain English) |
|---|---|---|
| `--market` | `equity` | Choose `equity` for stocks or `crypto` (parked). |
| `--universe` | `sp500` | The stock list to trade. Nine choices (`lab/cli.py`): `sp500`, `nasdaq100`, `russell2000`, `midcap400`, plus instrument baskets used by their matching research strategies — `fx_hedge`, `fomc_treasury`, `commodity_trend`, `treasury_curve`, `cross_asset_sleeve`. |
| `--eval-start` | `2021-01-31` | The starting date for the test simulation (YYYY-MM-DD). |
| `--eval-end` | Today's date | The ending date for the test simulation (YYYY-MM-DD). ⚠️ **Always set this explicitly on any run whose number you will quote.** The default is "now", so it moves between runs and two results become silently incomparable. The tell is SPY's own Sharpe changing between runs. |
| `--top-n` | 50 | Limit the simulation to the top N stocks (weight strategies). |
| `--lookback` | 252 | Trading days of history for momentum (weight strategies). |
| `--skip` | 21 | Most recent days skipped in the momentum measure (weight strategies). |
| `--max-stocks` | None | Cap the total number of stocks loaded (useful for super-fast test runs). |
| `--sweep-param` | none | Override one axis of the strategy's parameter grid, for example `--sweep-param rsi_oversold=20,25,30,35` (repeat the flag for several axes). |
| `--membership SPEC` | none (see below) | Where the S&P 500 membership list is read from: `db` (latest vintage), `db:<load_id>` or `fixture:<name>`, for example `fixture:sp500_v1_2026-06-02` (the frozen vintage 1 that every committed reference number was made on). **There is deliberately no default**: without the flag the `GGT_MEMBERSHIP` environment variable is used, and if that is also unset the run stops at once with `MembershipSourceNotSelected`, before loading any data. The flag is needed for any universe, not only `sp500`. |
| `--expect-membership-sha SHA256` | none | Refuse to run unless the chosen vintage's content sha256 equals this (pins a reproduction to the exact list). |
| `--snapshot PATH` | newest in `data/snapshots/MANIFEST.json` | The Parquet tape to read. |
| `--result-json PATH` | `results/lab/wfo_<strategy>_<universe>_<start>_<end>_vbt.json` | `--wfo`: where to write the run record. Every record carries a provenance block (vectorbt and Rust versions, the engine that actually ran, tape sha256, statistic definitions, baseline epoch, and the membership list used: `membership_sha256` plus a `membership` block with the load id or fixture name, coverage and ages). With `--blend` the same block is stored on the `lab_runs` row. |

### Simulation Modes (Choose One)

| Flag | Mode Name | What it does in plain English |
|---|---|---|
| `--wfo` | **Walk-Forward Optimization** | Simulates a realistic trading setup where the parameters are continuously re-optimized on a rolling training block of past data and then tested on a subsequent test block. ONE grouped Rust simulation per strategy configuration over the whole history (all grid combinations x all symbols, the point-in-time membership mask applied on the parameter columns), sliced per fold. Prints the fold table and writes the JSON record; **writes nothing to the database**. |
| `--blend` | **Portfolio Sleeve Blending** | Walk-forwards several `strategy@universe` sleeves (for example `ensemble@sp500,xs_momentum@nasdaq100`) and blends their out-of-sample curves with rolling inverse-volatility to target-volatility scaling. Persists the result as a `blend:` row in `lab_runs` (the first blend run adds the nullable `provenance` column to the table). |

Statistics are vectorbt's own (Sortino, bar-annualised CAGR); Sharpe, volatility and drawdown use 252 bars. The data window ends on the legacy exclusive end (`--eval-end` 2026-04-30 ends on the 2026-04-29 bar), so a run is comparable with the legacy references. Pin both `--eval-start` and `--eval-end` for any number you cite; engine and legacy numbers are different baselines and are never compared as one series (see `research/programme/vectorbt-engine/P5-completion.md`).

#### Portfolio Blending Parameters
When running a `--blend`, you can customize the portfolio combination overlay using:
- `--target-vol`: The annualized target volatility for the blended portfolio (default: `0.068` or 6.8%).
- `--blend-window`: The lookback window in days for computing sleeve covariance and returns volatility (default: `60` days).
- `--max-leverage`: Capping the portfolio leverage scale (default: `1.0`, the deployed value). Until 2026-11 the default was `2.0` (the idealised research value) and a blend quoted from it overstated what the live account could achieve; that caused an invalid run twice (2026-07-13 and 2026-08-18). Pass `--max-leverage 2.0` explicitly for an unconstrained comparison.

---

### Lab Examples

```bash
# Walk-forward the voting (ensemble) strategy on the pinned 17-fold window (frozen tape),
# on the frozen vintage-1 S&P 500 list (the list the reference numbers were made on)
ggt lab --strategy ensemble --wfo --eval-start 2021-01-31 --eval-end 2026-04-30 \
    --membership fixture:sp500_v1_2026-06-02

# The same list read from the database (load 1 is the same bytes; the result must be identical)
ggt lab --strategy ensemble --wfo --eval-start 2021-01-31 --eval-end 2026-04-30 --membership db:1

# Or choose once for the shell
export GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02

# The same on Nasdaq-100 stocks instead of the S&P 500
ggt lab --strategy ensemble --wfo --universe nasdaq100 --eval-start 2021-01-31 --eval-end 2026-04-30

# A quick 2-fold smoke run on 40 stocks
ggt lab --strategy ensemble --wfo --eval-end 2022-07-31 --max-stocks 40

# Test custom RSI levels in the grid
ggt lab --strategy rsi_reversion --wfo --sweep-param rsi_oversold=20,25,30,35 --eval-end 2026-04-30

# Cross-sectional momentum on a specific 2-year window
ggt lab --strategy xs_momentum --wfo --eval-start 2023-01-01 --eval-end 2024-12-31

# Walk-forward two sleeves and blend them (writes a blend row to lab_runs)
ggt lab --blend "ensemble@nasdaq100,ensemble@midcap400" --eval-start 2021-01-31 --eval-end 2026-05-18
```

---

### Understanding the Simulation Output

A `--blend` run writes its results to the tables below (`--wfo` writes only its JSON record). The
**seven** tables are all created by `src/ggTrader/lab/persist.py`:

| Table | What it holds |
|---|---|
| `lab_runs` | One row per simulation — strategy, market, eval window, params, status. |
| `lab_summary` | The headline metrics per run (Sharpe, drawdown, etc.) plus the benchmark's, and gate diagnostics. |
| `lab_returns` | The daily return series for the run. |
| `lab_equity` | The daily equity curve, alongside the benchmark's. |
| `lab_plans` | The actual holdings plan at each rebalance date, with eligibility counts and coverage. |
| `lab_sweeps` / `lab_sweep_combos` | Parameter-sweep runs and the per-combination results (written only by the removed legacy `--sweep`; kept for the history already stored). |

> **Note:** earlier versions of this document (and of `AGENTS.md`) referred to
> a `lab_periods` table holding per-fold metrics. **No such table exists or
> ever has.** Per-fold detail lives in `lab_returns` / `lab_equity`, and
> per-run metrics in `lab_summary`.

You can inspect these tables using SQL:

```sql
-- View the 10 most recent simulation runs with their headline metrics
SELECT r.run_id, r.strategy, r.eval_start, r.eval_end, r.created_at, s.metrics
FROM lab_runs r
LEFT JOIN lab_summary s USING (run_id)
ORDER BY r.created_at DESC LIMIT 10;

-- Pull the equity curve for a specific run
SELECT date, strategy_equity, benchmark_equity
FROM lab_equity WHERE run_id = '<run_id>' ORDER BY date;
```

---

## 2. ggt paper (Paper Trading)

This command runs one cycle of virtual trading (trading with fake money using live price feeds). The scheduled run is `scripts/paper_trade.sh` at 12:45 PT (just before the stock market closes) on weekdays, calling `ggt paper --live`.

> **Status 2026-11-06: the trader is stopped by the owner and its cron lines (12:45 PT run, 06:00 P&L report) are commented out in the crontab. Cron stays disabled until the owner re-enables it.** No date, finished phase or green check re-enables it, and an agent never edits the crontab or runs `ggt paper --live`. The signal path was rebuilt on the vectorbt engine (P7); the owner's dry run comes first (spec 8.4).

### The Paper Trading Pipeline:
1. **Engine check**: `engine.selfcheck` proves the Rust engine is dispatched (no numba or legacy fallback; failure means no orders).
2. **Signal Generation**: Runs the voting (`ensemble`) strategy, the validated `Ensemble.build()` with the live defaults, on the latest 365 days of bars, then reads buys, sells and last exits off the last bar. Input and output **sanity gates** (`src/ggTrader/paper/gates.py`) stop the run with no orders on an empty frame, duplicate or weekend days, too small a universe, too little history, a thin final bar (other than today's not-yet-complete one, which is dropped), or an absurd buy/sell count.
3. **Late-run gate**: signals ready after 12:55 PT (earlier on a half day) place no orders, because a DAY order after the close queues to the next open.
4. **Risk Guardrails**: Safety rules check the portfolio. `MAX_POSITIONS` (code default 30, the environment sets the live value) limits open positions, a single stock is capped at 5% of the portfolio, and trading freezes if today's loss exceeds 3% or the overall drawdown reaches 15%.
5. **Untradable symbols**: names in `UNTRADABLE_SYMBOLS`, names the broker marks `tradable: false` (held or about to be bought), and any order the broker rejects as "asset is not active" (the stuck AVB holding) are skipped and alerted, never traded and never a crash.
6. **Order Execution** (`--live` only): sends the approved orders to Alpaca (our paper trading broker).
7. **Database Storage** (`--live` only): records the executed trades and portfolio values to TimescaleDB.
8. **Telegram Notifications**: a summary of trades and portfolio value.

```bash
ggt paper                      # read-only dry run: logs intended orders, writes nothing, no Telegram
ggt paper --notify             # same, forwarding the DRY RUN lines to Telegram with a [DRY RUN] prefix
ggt paper --dry-run-writes     # the pre-rebuild dry run: also writes ledger/snapshot/peak/tape state
ggt paper --live               # places real paper orders (the only way to; refuses the two flags above)

# The S&P 500 list is named at the entry point: since slice M4 ggt paper reads the latest
# database vintage ("db") behind the membership gate below. There is no flag to change it and
# no fallback to a file.
```

**Membership gate (slice M4, redefined in M4b per spec Appendix C3).** Before any bar is fetched, a
wall-clock run on the latest `db` vintage must pass `ggTrader.data.membership_gate`. The gate counts
from an *independent* match: the last `matched` row that `ggt db membership check-holdings` wrote
(the list equals the SPDR SPY holdings for the holdings' own as-of date). Two clocks decide.
**Freshness:** the last match may be no more than **45 days** old (override: environment variable
`GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS`, a positive whole number; a bad value stops the run; it drops
to 10 once the weekly job exists). A list that was never matched is refused outright. **Lag:** the
day may trail `known_through` (the newest matched holdings as-of date, never earlier than the
vintage's last change date) by under **90 days** (warn from 90) and under **180 days** (refuse
from 180). A quiet spell without index changes therefore raises no alarm while the matches keep
coming; a dead pipeline does. A `check-upstream` result extends neither clock. A refusal raises
`SignalGateError` with code `membership_stale` (including "never matched"), `membership_lag`,
`membership_unavailable` (database unreadable, table empty or the vintage fails its integrity
check) or `membership_gate_config`, sends the usual "Signal gate - no orders placed" message, and
stops the run before any order is possible. Each message names its fix
(`ggt db membership check-holdings`, and `ggt db load-sp500` if it reports a mismatch). A warning
is logged and sent through the notifier ("Membership list: ..."); in a dry run it follows the
`DryRunNotifier` rules (Telegram only with `--notify`). The gate never applies to fixtures, to a
pinned `db:<id>` or to a caller-injected `now` (replays and tests). A match belongs to one vintage:
loading a newer vintage starts again from "never matched". With vintage 1 loaded and no match
recorded the gate refuses today.

A dry run without `--live` submits no order. Since P7 it is also read-only by default: no schema DDL, ledger, pending-order booking, snapshot, peak, split-state, dividend or price-tape write, and bars are fetched from yfinance into memory instead of the caching loader. The late-run gate only warns in a dry run so it can be rehearsed at any hour. See `research/programme/vectorbt-engine/P7-completion.md` for what the old dry run wrote.

---

## 3. ggt ingest (removed)

> **Removed in P10 (2026-11-06).** `ggt ingest` never worked: it hardcoded a two-symbol crypto
> list with the real sync call commented out and printed "Ingestion complete." having written
> nothing, and since 2026-08-22 it refused to run and exited `1`. The command, its module
> (`cli/cmd_ingest.py`) and the unused crypto inserter behind it
> (`data/historical/postgres_ingestor.py`) are deleted; both stay in git history and at tag
> `legacy-engine-vbt0285`.
>
> **How data actually gets loaded:** research reads the frozen tape
> (`--snapshot data/snapshots/*.parquet`) or the database read-only through
> `ggTrader.data.bars.load_bars`; equity OHLCV for the live path is fetched on demand and cached
> by `CachedYFinanceLoader` (`src/ggTrader/data/live/cached_yfinance_loader.py`). There is
> nothing to run by hand. Crypto ingestion is parked along with the rest of the crypto arc.

---

## 4. ggt db (Database Management)

Commands for diagnosing and maintaining your local database.

```bash
ggt db <subcommand>
```

| Subcommand | What it does in plain English |
|---|---|
| `diag` | Prints table sizes and total row counts (lets you see how much data you have). |
| `clean` | Deletes broken, orphaned, or incomplete rows. |
| `truncate` | Wipes out database tables completely (asks for confirmation first). |
| `compression` | Turns on or off TimescaleDB's compression features (saves hard drive space). |
| `export` | Creates a database backup file. |
| `load-sp500` | Validates an S&P 500 membership file and stores it as the next **vintage** in schema `universe` (dry run unless `--commit`). |
| `membership` | `apply-ddl` creates or refreshes schema `universe` (safe to repeat); `verify` compares a stored vintage with a file or a registered fixture on every snapshot date; `check-upstream` compares the public upstream list with the latest vintage and records the check (informational); `check-holdings` matches the latest vintage against the SPDR SPY holdings file and records the check (this is what keeps the live gate open); `status` shows the latest vintage and what the live gate would decide; `diff --from A --to B` compares two vintages. |

### Examples:
```bash
# Print database diagnostic info
ggt db diag

# Clean up database tables
ggt db clean

# Enable database compression to save disk space
ggt db compression --enable

# Back up the database to a backup.sql file
ggt db export > backup.sql
```

### Versioned S&P 500 membership (`universe` schema)

Spec: `docs/superpowers/specs/2026-11-07-membership-database-design.md`. Slice M1 built the
schema, the loader and `verify`; slice M2 added the read door (`ggTrader.data.membership`, below)
and the fixtures manifest. No backtest, oracle or trade reads the new door yet (slice M3
migrates the readers), so no reference number moves.

```bash
# M0: create / refresh the schema (sql/universe_schema.sql; idempotent)
ggt db membership apply-ddl

# M1: dry run (default) prints the diff against the latest vintage and writes nothing
ggt db load-sp500 --from-file data/universe/sp500_constituents_history.csv.gz \
    --source fixture-import --retrieved-at 2026-06-09T00:00:00Z --parent none

# ... then write it (one transaction under an advisory lock)
ggt db load-sp500 --from-file data/universe/sp500_constituents_history.csv.gz \
    --source fixture-import --retrieved-at 2026-06-09T00:00:00Z --parent none --commit \
    --note "committed f8c7efe0; true retrieval time unknown"

# Compare vintage 1 with the file on all 2,712 snapshot dates and every calendar day (read-only)
ggt db membership verify --load-id 1

# The same, against a registered fixture (its three hashes are checked before the comparison)
ggt db membership verify --load-id 1 --against-fixture sp500_v1_2026-06-02
```

`verify` takes `--against-file PATH` (default: the committed vintage-1 file) or
`--against-fixture NAME`, not both.

**SPY holdings match (slice M4b, spec Appendix C).**

```bash
# Download the public SPDR SPY holdings workbook, compare its tickers with the latest vintage's
# members on the holdings' own as-of date, record one check row (and the raw bytes)
ggt db membership check-holdings     # exit 0 matched, 2 mismatch (reported, nothing loaded), 1 failure
```

The workbook (`https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx`,
fetched with a browser-style User-Agent, redirects followed) is read with the Python standard
library only (`ggTrader.utils.xlsx_min`; no spreadsheet dependency). The header row is the first
row with a `Ticker` cell, the date is the "As of DD-Mon-YYYY" text above it. Cash, placeholder and
other non-equity lines (a `-` or `USD` ticker, a ticker that is not letters/digits plus an optional
share class, a `Cash`/`Other` sector) are dropped and listed in `detail.excluded_lines`. Both
sides go through `normalize_yf_ticker` (`BRK.B` and `BRK-B` are the same ticker). A duplicate ticker,
or a count outside 480..520, means the structure is not what we expect and is `fetch_failed`.

`check-holdings` appends exactly one row to `universe.checks` per run and archives the raw bytes
(gzip, with their sha256) in `universe.check_blobs` whenever bytes arrived. It never touches
`loads`, `membership` or `snapshot_dates`. Outcomes: `matched` (equal sets; `matched_load_id` is
the vintage; **the only outcome that advances the live gate**), `mismatch` (`detail.only_in_list`
and `detail.only_in_holdings` are sorted ticker lists; nothing is loaded) and `fetch_failed`
(unreachable, HTTP error, truncated, HTML or a wrong structure instead of a workbook, an as-of date
after today or before the vintage's first snapshot; `detail.reason` says which). `detail` also holds
the as-of date and text, counts, the source URL, the response headers and the raw sha256. If the
holdings are dated before the vintage's last change but inside its coverage, they are compared with
the snapshot in force on their own date (a match then sets no `known_through` beyond the
vintage's coverage end). `status` shows the last matched check, `known through` and the age.

**Upstream check, status and diff (slice M4).**

```bash
# Fetch fja05680/sp500 at its newest commit, compare with the latest vintage, record one check row
ggt db membership check-upstream     # exit 0 equal, 2 differs (reported, not loaded), 1 failure

# Latest vintage, coverage, last verification, recent checks, and the live gate's verdict
ggt db membership status [--asof 2026-11-29]

# Two stored vintages: dates only on one side, changed dates, historical / tail / rename split
ggt db membership diff --from 1 --to 2 [--json diff.json]
```

`check-upstream` appends exactly one row to `universe.checks` and never touches `loads` or
`membership`; a differing upstream is reported and is **never loaded automatically** (loading is
`ggt db load-sp500`). Since M4b the check is informational: the upstream feed is dead and the
gate no longer counts it. Outcomes: `unchanged` (the raw bytes, or failing that the parsed content,
equal the latest vintage; it no longer resets any clock), `refused`
(upstream differs; `detail.result = "differs"` carries the upstream coverage end, the dates only on
each side and the historical / tail / rename split) and `fetch_failed` (unreachable, HTTP error,
a content type that is not text, a body shorter than its `Content-Length` or not ending in a
newline, or a file that fails the loader's validation). The file is read at the exact commit the
GitHub commits API names, so bytes and commit always match. `status` and `diff` are read-only.

**Reading membership from code (`ggTrader.data.membership`, slice M2).** Pick a source by name;
there is no default and no fallback:

| Spec string | Reads |
|---|---|
| `db` | The latest vintage in the database, resolved to one load id when the source is built. |
| `db:<load_id>` | That exact vintage (what a cited result should pin). |
| `fixture:<name>` | A frozen file listed in `data/universe/membership/MANIFEST.json`, hash-checked. |

`membership_source(spec)` takes the string, else the environment variable `GGT_MEMBERSHIP`, else
raises `MembershipSourceNotSelected`. If `db` cannot be read it raises `MembershipUnavailable`
(connection details redacted); it never reads a fixture instead. The reader login is
`GGT_MEMBERSHIP_DB_URL` when set, else the project's usual connection, over a read-only session.
A source offers `history()` (the shape of `load_sp500_history()`), `members_asof`,
`all_members_between`, `snapshot_asof`, `vintage()` and `provenance()` (vintage number or fixture
name plus content checksum, for result JSON). `check_membership_age(vintage, asof, ...)` compares
a vintage's age with warn and refuse thresholds you pass; the live gate (`membership_gate`,
slices M4 and M4b) has its own inclusive lag thresholds and is the only caller that enforces them.

**Every reader now takes the source (slice M3).** `universe_members_asof`, `universe_all_between`,
`sp500_members_asof`, `all_members_between`, `eligible_at`, `equity_universe_between`,
`pit_mask`, `membership_mask`, `check_constituent_age`, `universe_mask`, `load_bars(universe=
'sp500', membership=...)` (required there), `generate_signals` and `compute_sleeve_curve` have a
`membership=` argument. Passing `None` means "the source named by `GGT_MEMBERSHIP`", and raises
`MembershipSourceNotSelected` if that is unset; no function falls back to a file.
`load_sp500_history()` with no argument selects a source the same way, and `load_sp500_history(path)`
stays a plain file reader for frozen studies that name their file. `download_sp500_history()` is
gone: nothing in `src/` writes a membership file. The process caches the source once per spec and
the history frame once per content hash. The root `conftest.py` sets
`GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02` for every test, so no test needs the database.

**Drivers (`--membership SPEC`, `--expect-membership-sha SHA256`).** The oracle drivers
`scripts/p5_wfo_validation.py`, `p7_live_golden.py`, `p3_pit_parity.py` and
`make_tape_snapshot.py` default to `fixture:sp500_v1_2026-06-02`; `equity_backfill.py` (needed for
`--universe sp500`) and `google_trends_backfill.py` have no default, as `ggt lab`.
`legacy_reference_runs.py` runs the legacy code, which reads its own copy of the frozen file, so it
accepts only `fixture:<name>` and checks that the file the legacy code reads has that fixture's
hash. Each driver records the vintage in its JSON: in the `provenance` block as
`membership_sha256` and `membership` (`kind`, `load_id` or `fixture`, `raw_sha256`, coverage,
`retrieved_at`, `verified_at`, `age_since_verified_days`), never in the payload, so the guarded
comparison with the committed evidence is unaffected (`p3_pit_parity.py` and
`make_tape_snapshot.py` write plain JSON and add the two keys at the top level). `provenance.compare`
treats a difference in `membership_sha256` as `MEMBERSHIP_DIFF` (not comparable); a record without
it is "unknown", no flag.

The only registered fixture is `sp500_v1_2026-06-02`, the existing file
`data/universe/sp500_constituents_history.csv.gz`, which stays where it is and is never edited
(file sha256 `b30739a3...`, content sha256 `256e022d...`; it is vintage 1, load 1 on Encom).

| `load-sp500` flag | Meaning |
|---|---|
| `--from-file PATH` | Required. CSV or CSV.GZ (`date,tickers`); the upstream fetch (`--from-upstream`) is not built yet. |
| `--retrieved-at ISO` | Required. When the bytes were fetched; needs a zone (`...Z`). |
| `--source` | `fixture-import` (default) or `fja05680/sp500`. `--source-url`, `--upstream-commit`, `--upstream-etag`, `--note` are stored with the load. |
| `--expect-sha256` | sha256 of the uncompressed bytes; the load is refused if it differs. |
| `--parent N` / `none` | Pin the latest vintage seen in the dry run; refused if another load got in first. |
| `--commit` | Write. Without it the run is a dry run. |
| `--force --reason TEXT` | Needed when the diff against the parent is suspicious (coverage shrinks, more than 5 historical rewrites, 25 tail changes or 3 renames). |
| `--report-json PATH` | Save the diff report. |

Exit status: 0 success, 2 refused (validation, pin, hash or suspicious diff), 1 `verify` found a
difference. The connection is the project's usual one; `GGT_MEMBERSHIP_LOADER_URL` overrides it
for the loader only. Tests that need a real server are marked `integration` and `db`, run only on
request (`GGT_TEST_DB_URL`), and use a throwaway database.

---

## 5. Machine Learning Signal Pre-Screen Script (archived)

`scripts/ml_signal_screen.py` ran on the legacy signal classes and was archived in P5c to
`research/programme/vectorbt-engine/legacy-scripts/` (it runs only at git tag
`legacy-engine-vbt0285` through `scripts/legacy_env.sh`). The closed ML-gate findings are in
`research/RESEARCH_SNAPSHOT.md`.

---

*See the [Architecture Guide](architecture.md) for how these components fit together, and [Developer Guidelines](../AGENTS.md) for coding standards.*

*Back to [README.md](../README.md).*

## Event-timing audit research utility

`.venv/bin/python research/audits/event-timing/code/run.py --help` describes the standalone
read-only evidence audit. It defaults to offline operation; `--smoke` caps each
stratum at two and writes partial JSON/Markdown under `/tmp`. `--evidence` and
`--detector-flags` ingest checksummed evidence. `--resume` preserves budgets and
frozen inputs. DB SELECTs and external collection require separate explicit
`--read-db` / `--network` flags; `.env` is never loaded. Collection remains
paused after the recorded HTTP 403; the already-authorized contact does not
clear that stop. The approved offline amendment still needs fresh code review.
See the
[implementation handoff](../research/audits/event-timing/event-timing-audit-implementation.md). A completed bounded
`--network` run can explicitly select `--output research/audits/event-timing/_event_timing_audit.json`
to publish the registered JSON/Markdown, including honest UNRESOLVED results.
Smoke, offline and runtime-interrupted work remain partial. Byte references alone
do not validate semantic facts; see the field-review contract in the handoff.

Every new assessment also reports `insider_filing_date_only`, alongside the
original `insider_cluster` and unchanged `pead`. No new CLI option is needed.
Each has its own requirement matrix, annual coverage, reconstructed events and
disposition. The new timing PASS is date-only convention compliance, with
observed dissemination UNVERIFIED. Amendment identity/hash binds both progress
and result metadata; old registrations or changed code/policy cannot resume.
This does not authorize collection, a budget reset or cache migration.
