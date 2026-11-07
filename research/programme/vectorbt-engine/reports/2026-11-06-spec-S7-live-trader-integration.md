# S7: Moving the live paper trader onto the vectorbt 1.2.0 + Rust stack

Date 2026-11-06. Repo read-only at `6774b4d3`. Nothing in the repo, the broker, Docker or Telegram was touched.
Evidence tags: MEASURED (I ran it, files in this folder), CITED (file:line), ESTIMATE, UNVERIFIED.
Companion docs: S1 (architecture), S2 (upgradeability), S5 (validation protocol, section 4 shadow). I align with them and deepen the live side.

## 0. Findings that change the plan

1. **The live signal path calls no vectorbt function today.** `generate_core_signals` -> `generate_signals` uses pandas-only
   indicators (`indicators.py` rolling/ewm). vectorbt is imported only because `overlay.py:12` imports `lab.simulate`
   (`import vectorbt as vbt`, `simulate.py:9`). `simulate_signals` is called only in `overlay.compute_sleeve_curve` (blend path, not run
   by `generate_core_signals`). So "live on vectorbt" means *new code*: vbt indicators replace the pandas ones. That is a signal
   definition change, not an engine swap, and it needs the re-baselined strategy as the reference (section 2.1).
2. **Live cannot share a stack with 0.28.5.** vbt 1.2.0 needs pandas>=3.0.3, numpy>=2.4.6, numba>=0.66 (B section 6). A dry-run resolve of the
   whole `pyproject.toml` dependency list (minus dead `pandas-ta`, plus `vectorbt==1.2.0 vectorbt-rust==1.2.0`) on cp312 **succeeds
   with wheels** today (MEASURED, `dry.log`, `req_new.txt`): pandas 3.1.1, numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, plotly 7.1.0,
   scikit-learn 1.9.1, pyarrow 26.0.1, scipy 1.18.1, **SQLAlchemy 2.1.8, yfinance 1.7.0, alpaca-py 0.45.0** (host venv: 2.0.48 / 1.2.0 / 0.43.4).
   The last three are live-path libraries (`persist.py`, the loader, the broker) that would jump with an unlocked rebuild. `ta` is sdist-only.
3. **Rust removes the cold start; numba cannot be the live fallback.** MEASURED on a loaded box (load avg 12), fresh process, 500 symbols x 252 bars,
   vbt 1.2.0 / pandas 3.1.1 / numba 0.68.1 (`coldstart.py`, `numba_cold.txt`, `numba_warm.txt`):
   Rust: `RSI/BBANDS/MA/MACD.run` 0.01-0.02 s each, 0 numba dispatchers compiled; tiny `from_signals` 0.5 s (3 dispatchers: `get_group_lens_nb`, `impl`);
   whole script 3.3 s incl. 2.3 s import (twice). numba, empty cache: 190 s total (RSI 33 s, BBANDS 18 s, MACD 19 s, `from_signals` 57 s).
   numba, warm on-disk cache: still 45 s (`from_signals` 28 s). Absolute numbers are inflated by load; the ratio is the point.
   A numba fallback inside a 15-minute pre-close window is therefore not acceptable. **The live fallback is the legacy pandas path.**
4. **Order selection is alphabetical and the cap is 3.** `buys = sorted(...)` (`signal_runner.py:100`), `MAX_POSITIONS=3` in live
   (`risk.py:8-9`), buy loop in list order (`trader.py:880`). One extra or missing early-alphabet buy changes which names are bought.
   The shadow must compare *would-be orders after the cap*, not only raw signals.
5. **The live bar is partial.** `trader.py:104-131` docstring: `signals["sells"]` sees "today's partial bar" at 12:45 PT. A shadow run
   at 13:05 PT would read a different (closed) bar. See 2.1 for how to keep inputs identical.

## 1. Live path map and frozen surface

Cron `45 12 * * 1-5 scripts/paper_trade.sh` (`paper_trade.sh:3-6`, `crontab -l` shows the entry). The script does
`docker compose run --rm ggtrader_live python -u ggt.py paper --live` (`paper_trade.sh:55`): a **fresh container per run** from the local
image `ghcr.io/garykuepper/ggtrader:latest`; `.env` is re-read via `env_file` (`docker-compose.yaml:7`). Run time 33-93 s (logs 11-02..11-05, MEASURED).
An `EXIT` trap pages Telegram on any non-zero exit (`paper_trade.sh:25-49`).

```
cron 12:45 PT -> paper_trade.sh -> docker compose run --rm ggtrader_live (fresh container, image :latest, TZ=America/Los_Angeles, 16G/2cpu)
  ggt.py paper --live -> cli -> paper.trader.run_paper_trading (trader.py:1101)
    AlpacaBroker.get_account (multiplier>1 => RuntimeError)            [alpaca-py]
    PaperTrader.run (trader.py:493)
      init_paper_schema ; _reconcile_pending_orders ; broker.get_clock (closed => skip)
      generate_core_signals (signal_runner.py:175)  <== THE SEAM
        generate_signals("sp500", lookback_days=365) (signal_runner.py:60)
          data.core.index_constituents.universe_members_asof / normalize_yf_ticker   (PIT SP500, static csv.gz)
          lab.data.fetch_stock_ohlcv -> CachedYFinanceLoader (TimescaleDB ohlcv, naive-UTC) -> yfinance gap fetch (+_cache_to_db WRITE) -> Tiingo
          lab.strategies.ensemble.EnsembleSignal(LabConfig(min_history_bars=60)).select / .to_targets
            -> lab.strategies.indicators: bb/rsi/ema/macd/volume_bb signals, eligibility_mask, extract_close/volume
          last_bar entries/exits -> buys, sells, last_exit{sym: last exit date}, as_of, universe_size
          paper.feature_gate.FeatureGate (ML_GATE_ENABLED, default off)
      refresh_benchmark_tape (SPY,TLT,GLD,DBC,IEF 30d upsert, WRITE to ohlcv, never raises)
      freshness gate (as_of <= 4d, trader.py:76,562) ; RiskGuard (drawdown halt, daily loss) ; untradable (AVB) ; split/dividend checks
      _missed_exits (last_exit vs get_last_buy_dates) ; catastrophe_stop (flag) ; cash_sweep (CASH_SWEEP_ENABLED, SPY)
      sells -> _poll_orders -> buys (alphabetical, MAX_POSITIONS=3) -> _poll_orders -> sweep buy
      persist: log_trade / log_pending_order / log_snapshot / save_peak_value  (paper_* tables via lab.persist.get_engine)
      Telegram: trade_alert, daily_summary, risk line
```

**What persistence depends on signal details.** `paper_trades(run_date, side, symbol, amount, order_id, reason)` and `paper_pending_orders`
are keyed by the symbols the signals produced. `paper_snapshots(run_date, portfolio_value, cash, positions JSONB)` depends on signals only
through fills. `paper_rebalance_state` is blend-only (unused by core). `paper_risk_state`, split and dividend tables are signal-independent.
`last_exit` dates feed `_missed_exits` via `get_last_buy_dates()`; a wrong `last_exit` causes spurious or missed sells (warm-up study: 7.0% wrong at 120 d, 0.0% at 365 d,
`research/paper-system/indicator-warmup/2026-09-28-live-indicator-warmup.md`). A23 SSO sleeve and the two-symbol sweep are **not** in the live path
(advisory helper and design only, `next_steps.md:19-24`); only the SPY cash sweep is (`cash_sweep.py`, pure sizing, no lab imports).

**Frozen surface: lab/data symbols the live path imports** (a change to any is a live-signal change):

| Symbol | Imported at | Used live? |
|---|---|---|
| `lab.strategies.ensemble.EnsembleSignal` (`select`, `to_targets`, `_generate_signals`, `_apply_time_stop`, defaults `FIVE_VOTERS`, `min_agree=2`, `bb 20/2.0`, `rsi 14/30/50`, `ema 20/50`, `macd 12/26/9 div 20`, `vol 20/2.0`) | `signal_runner.py:11`, `overlay.py:13` | yes |
| `lab.strategies.indicators`: `bb_signals`, `rsi_signals`, `ema_signals`, `macd_signals`, `volume_bb_signals`, `eligible_symbols`, `eligibility_mask`, `extract_close`, `extract_volume` (+ `mtf_*`, `*_strength` imported by `ensemble.py:10-27`) | transitive via `ensemble.py` | the first nine yes; mtf and strength fns imported but unused (MTF not in default voters) |
| `lab.strategy.LabConfig`, `Plan`, `SignalTargets` | `signal_runner.py:12`, `overlay.py:14` | yes |
| `lab.data.fetch_stock_ohlcv` | `signal_runner.py:10`, `overlay.py:11` | yes (lookback 365d, DB-first, gap fetch writes the tape) |
| `lab.data.STOCK_BASE_CONFIG` | `overlay.py:11` | import only (blend path) |
| `lab.simulate.simulate_signals` (imports `vectorbt` at module level) | `overlay.py:12` | **import only**; this is the sole reason `import vectorbt` runs live (09-23 crash) |
| `lab.allocation.inverse_vol_weights`, `target_vol_scale`, `trailing_realized_vol` | `overlay.py:82` lazy | blend only, not called |
| `lab.persist.get_engine` | `paper/persist.py:10` | yes (all paper_* DB access) |
| non-lab: `data.live.cached_yfinance_loader`, `yfinance_loader`, `tiingo_loader`, `data.core.index_constituents`, `utils.config` | loader chain | yes |

Consequence for the refactor: a lab restructure (S1's `ggTrader.engine`) must keep these import paths alive or move the paper
imports in the same commit, and `overlay.py:12` should stop importing `lab.simulate` before cutover (decouples the 09-23 failure mode).

## 2. Cutover design

### 2.1 What is being validated
Owner decision: use vbt indicator definitions in live. vbt defaults are not textbook (RSI and MACD use simple MAs, BBANDS ddof=0, first-valid-bar differs;
F section b). Two cases, pre-registered before the shadow starts:
* **Parity port** (explicit flags: `RSI window=2n-1 ewm=True`, `BBANDS ddof=1`, `MACD macd_ewm=signal_ewm=True`): expect 0 last-bar diffs after 365 d warm-up.
* **Definition change** (vbt defaults, or any re-tuned parameter): the signals differ by construction; the live reference is the
  *re-baselined lab strategy*. The invariant is then **lab == live on the same frame**, with legacy-vs-new differences measured and reported, not zeroed.
The shadow below reports both comparisons every day, so the owner sees which case applies. Which one is intended is an owner decision I cannot make.

### 2.2 Shadow mode (no orders; a separate job)
**Placement.** Not inside `ggtrader_live` (frozen image, frozen code, and the pandas-3 stack cannot coexist). A host script
`scripts/shadow_vbt.sh` at **12:50 PT** (after the live run, which ends by ~12:47), own `.env.shadow` (DB + Telegram only, **no `APCA_*` keys**, so it physically cannot
reach the broker). It runs three containers in sequence from two immutable images:
* **O** = the frozen production image by digest `ghcr.io/garykuepper/ggtrader@sha256:0f2403f1...27cbf37c7` (legacy code, old stack). `docker manifest inspect` returns
  the manifest today (MEASURED 2026-11-06; 12 layers, 527 MB compressed; tag `291e6dd` also resolves). Shadow script mounted read-only with `-v`.
* **O'** = new image, legacy pandas indicators (isolates the *stack* effect: pandas 3, new loader).
* **N** = new image, vbt indicators (isolates the *indicator* effect). This 3-way split is the addition over S5: without O' a diff cannot be attributed.
Step 1 (O) fetches the 365-day frame once, writes `panel.parquet` + sha256, plus its own signals. Steps 2-3 run O' and N both on **O's parquet** (engine diff) and on a frame
**their own loader** produced (data-path diff; hash of the two canonicalised frames must match). The compare step runs in the new image.

**Inputs equal to live.** The shadow reads with `use_db_cache` only, no network, no `_cache_to_db`: the loader's DB-first path should return the bar the 12:45 run cached
(UNVERIFIED that the DB holds the partial 12:45 bar for all 503 symbols; day-1 check: compare O-vs-live agreement). Defence in depth: DB role `ggt_shadow_ro` (SELECT only on `ohlcv` and
`paper_*`) so a pandas-3 timestamp bug can never write to the tape (CLAUDE.md: whole-tape corruption history; changelog.md:19 "no-writes rule on the price table").
Post-lift, in-process dual compute inside the live run removes the 12:45-vs-12:50 drift entirely.

**Data model (files, not tables).** Per session `results/shadow/<YYYY-MM-DD>/`: `meta.json` (image digests, lock hash, vbt/rust/pandas/numpy/numba versions, engine probe, runtime, maxrss,
panel sha256, as_of, universe_size), `panel.parquet` (close+volume, ~1-2 MB, ESTIMATE; 60 days ~100 MB), `signals_{O,Oprime,N}.json`
(`buys`, `sells`, `last_exit`, per-voter vote counts, indicator value and distance to threshold for every symbol whose vote is within 1% of a threshold),
`orders_wouldbe_{O,N}.json` (output of a pure function applying the same cap, alphabetical order, sweep-free sizing to the day's `paper_snapshots.positions` read-only),
`diff.json`. A DB table is deferred (a new table means `init_paper_schema` changes in frozen code); JSON also feeds the forward-evidence plan
(`next_steps.md:6-11`: preserve signals, inputs, configuration, outcomes).

**Comparison report** (markdown + JSON, Telegram-free except failures): per symbol and day, rows for `entry`, `exit`, `last_exit date`, `would-be BUY/SELL`:
O vs N, O vs O', O' vs N, and each vs the actual `paper_trades` of the day. Columns: symbol, votes by voter, indicator value, threshold distance, class (A
float-noise flip |diff|<1e-9 at threshold; B warm-up/definition; C unexplained). Plus the data-path hash result and a runtime/RSS line.

**Duration.** At least 20 consecutive trading sessions **and** at least 30 shadow buy+sell events (S5:186), spanning one month-end, one session with
exits and entries on held names, and ideally one half-day or holiday skip. Offline first: replay O vs N over ~1,000 sessions of the stored tape (about 500k symbol-days)
to bound flip frequency before any forward day (this closes F open item 2). Pre-registered counts: ~8 entries and ~47 exits per day at 503 symbols (warm-up study: 1,026 entries / 5,881 exits over 125 sessions),
so 20 sessions give ~160 entries and ~940 exits (ESTIMATE from that study).

**Pass / fail** (fixed before day 1):
* P1 O' vs O: 0 diffs in frame hash, buys, sells, last_exit (the stack change is invisible). Fail => do not proceed to indicators.
* P2 Parity port: O vs N 0 diffs on every symbol with >=200 bars (S5:187). Definition change: N == lab output on the same frame (0 diffs) and a signed count of
  legacy-vs-N differences published; owner accepts or rejects.
* P3 would-be orders after the cap identical on every day for a parity port; for a definition change, owner reviews each differing day.
* P4 O vs actual orders: every actual SELL in O.sells union missed-exit list; every actual BUY in O.buys; mismatches each explained (intraday drift, AVB, caps).
* P5 100% of runs completed, p95 wall <= 120 s, maxrss <= 2 GB (ESTIMATE ceiling), engine probe `rust` on every run, zero `UserWarning` about version mismatch.
* P6 Any class C diff, any crash, any panel-hash mismatch: stop, root-cause, **reset the day count**.

### 2.3 The switch: feature flag and rollback ladder
All reads happen in `signal_runner`, not `trader.py` (so no new `trader.py` persist import):
* `SIGNAL_ENGINE=legacy|vbt` (default `legacy`; unknown value raises => no trade). `.env` is re-read each cron run, so a flip takes effect at the next 12:45 with no rebuild.
* `SIGNAL_CROSSCHECK=1`: compute both; if buys or sells or last_exit differ, raise `SignalDisagreement` => notifier + exit non-zero => **no orders** (missed exits self-heal via `_missed_exits`; a wrong order does not).
* `SIGNAL_ENGINE_FALLBACK=legacy|none`: on *infrastructure* failure of the vbt path (selfcheck fails, import error, Rust missing, exception) run the legacy path and page with "FALLBACK USED".
  Allowed only during probation; after probation set `none` (fail closed). Disagreement is never auto-resolved.
* Compose image becomes `ghcr.io/garykuepper/ggtrader:${GGTRADER_TAG:?}`, tag set in `.env` (compose interpolates the project `.env`; CITED compose docs, UNVERIFIED on this host).

Rollback ladder, fastest first, none needs a rebuild:
* R1 (seconds): `.env` `SIGNAL_ENGINE=legacy` (legacy code on the new stack, validated by P1).
* R2 (seconds, image local): `.env` `GGTRADER_TAG=<previous immutable tag>`; next cron run uses it.
* R3 (minutes): `docker pull ghcr.io/garykuepper/ggtrader@sha256:0f2403f1...` then R2. Pull-test it once before cutover (not done; I only read the manifest).
Never rely on re-tagging `:latest`; `docker image prune` or a `compose pull` orphans the old image ID (only the `:latest` tag exists locally, MEASURED `docker images`).
Keep a local alias tag `ggtrader:legacy-291e6dd` on the old image so it survives prune.

### 2.4 Cutover sequence (post-lift) changes one thing at a time
D1 stack only: deploy new image with `SIGNAL_ENGINE=legacy`, 5 sessions. D2 indicators: `SIGNAL_ENGINE=vbt`, `SIGNAL_CROSSCHECK=1`, `FALLBACK=legacy`, 20 sessions. D3: crosscheck off,
fallback `none`; keep the legacy module one quarter. Each step needs a green daily check and owner OK.

### 2.5 What is allowed before the freeze ends (verify wording)
Wording: `next_steps.md:30-31` "The production paper trader remains frozen. The observation window's qualifying start and any reset are unresolved."; `next_steps.md:12-13`
"A calendar date alone does not lift the production freeze"; `working_task_list.md:26` "A plan or design does not authorize a return test, installation, scheduling,
container change or order... elapsed calendar time does not establish readiness"; readiness doc line 3 and 54: ~2026-12-29 is the *earliest possible review*, "not an established clearance date";
readiness doc 42-47: do not pull or recreate production during the freeze. changelog.md:268 records "the owner decision to retire stock picks around 2026-12-29, leaving only the SPY sweep";
I make no recommendation on it, but it decides whether the ensemble signal path needs porting at all, so confirm scope with the owner before spending the live-side hours.
* **Allowed without a new decision (my reading, owner to confirm):** work on a branch (never merge to `main`; S5:171-173), docs, tests that touch no DB, offline replay with read-only SELECTs
  on the host venv (precedent `--read-db` convention, `cli_reference.md:352`), reading GHCR manifests.
* **Needs explicit owner authorization first (A1):** scheduling the shadow cron, building/running the shadow image, creating the read-only DB role, a second env file.
  The literal list "installation, scheduling, container change" makes these unauthorized by default.
* **Not allowed until the readiness decision (A3):** any change under `src/ggTrader/paper/`, the signal chain files (S5:170-172), `pyproject.toml` deps, `Dockerfile`, `docker-compose.yaml`, `.env`,
  and anything that moves `:latest` that a pull would then deliver. Optional A2 (compose tag pin, no behaviour change, S2 step 0) is a "container change": ask.
* The 12-29 date is therefore only the earliest review; the forward shadow can finish before it, the cutover cannot be scheduled by calendar.

## 3. Deployment mechanics for the new stack

* **Base/versions (image).** Today: `python:3.12-slim` unpinned (image has 3.12.14, MEASURED via `docker image inspect`), `pip install -e ".[test]"` after `COPY . .` (`Dockerfile:12-21`):
  every src edit re-resolves all dependencies (S2 point 1). New: `python:3.12-slim@sha256:<digest>` (digest UNVERIFIED, look up at implementation), lockfile-driven install
  (`pip install -c constraints.lock -e ".[test]"` or S2's uv sync), `pandas-ta` removed, pins: pandas 3.1.1, numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, `vectorbt[rust]==1.2.0` (extra pins `vectorbt-rust==1.2.0`),
  plotly 7.1.0 (the `<7` cap in `pyproject.toml` is for 0.28.5 only). Wheel: `vectorbt_rust-1.2.0-cp312-cp312-manylinux_2_17_x86_64` (917 KB, B 4.4), resolved from PyPI in my dry run, no compiler.
* **Build time / size (ESTIMATE).** Rust wheel +1 MB, build +5-15 s. Size 1.65 GB now (MEASURED); new stack about the same, +/-5% (scikit-learn, pyarrow, numba/llvmlite are the bulk; pandas 3 not larger).
  Real change: dependency layer cached by lock hash, so a src-only push drops from ~2.5 min to ~1 min (ESTIMATE). Drop `build-essential` if all wheels resolve (only `ta` is sdist, pure Python). Compose `build: .` stays.
* **Numba cold start: recommendation = rely on Rust, no baked/volume cache.** Reasons: MEASURED 0.01 s vs 33 s per indicator cold; baked caches are CPU-specific (S2, numba docs, UNVERIFIED here) and CI CPUs differ from Encom's;
  warm numba cache still took 45 s on my box. Set `NUMBA_CACHE_DIR=/tmp/numba` (writable by `appuser`) only to silence cache warnings. If the owner ever sets `GGT_VBT_ENGINE=numba` for research,
  use a named volume there, not in the live service.
* **Smoke test (extends `Dockerfile:25`).** Add `ggTrader/engine/selfcheck.py` (S1 prototype `s1_engine_selfcheck.py`) and run it **twice**: once as root after install, once after `USER appuser` (proves write perms). Assertions:
  `vbt.settings["engine"]` is `rust` not `auto`; `vectorbt._engine.is_rust_available()`; vectorbt and vectorbt-rust share `major.minor`; the mismatch `UserWarning` is an error
  (`warnings.simplefilter("error")`); a 2-column trailing-stop `from_signals` increments the wrapped `simulate_from_signals_rs` counter exactly once; `RSI.run(window=27, ewm=True)` equals the pandas Wilder RSI(14) on a fixed 300-bar fixture within 1e-9
  after bar 100 (definition-drift canary); whole check under 10 s with an empty numba cache; `pip check`; `import ggTrader.paper.trader`; `ggt.py paper --help`.
  Image labels: lock sha256, git sha, version tuple.
* **Workflow (`docker-build.yml`).** Today it builds and pushes with no test (S2 L4). Change: `build-push-action` with `load: true`, `docker run` the smoke, then push (same cache). Tags: keep `type=sha,prefix=` (immutable short sha, already
  published, line 35) and add `type=raw,value=vbt-1.2.0-<sha>`; `latest` stays default-branch-only. A branch build via `workflow_dispatch` produces only the sha tag, so the shadow image can come from CI without moving `:latest`;
  for the shadow phase prefer a local `docker build -t ggtrader-vbt:<sha>` (no registry, no `:latest` risk). Record the pushed digest in the job summary and in `docs/changelog.md`.
  `ci.yml:30` installs unlocked too; point it at the lock so green CI tests what ships.
* **Deploy sequence** `push main -> build+smoke -> push -> set GGTRADER_TAG=<sha> in .env -> docker compose pull ggtrader_live -> up -d -> in-container selfcheck -> next-run checklist`. The
  tag edit, not `:latest`, selects the image, so an unrelated docs push cannot change what cron runs (the hazard in readiness doc 42-47). Never deploy 12:30-13:15 PT (`ggtrader-deploy` rule).

## 4. Failure modes of the live run on the new stack

| Failure | Detection | Fail-safe |
|---|---|---|
| Rust wheel missing or minor-mismatched, engine silently numba (B 4.3: `auto` falls back with one `UserWarning`) | selfcheck at the start of `generate_signals` (before data fetch): engine setting, `is_rust_available`, counter probe | `engine="rust"` explicit never `auto`. Probation: legacy fallback + page. After: raise before any order, existing handler sends "signal generation error" (`trader.py:519-523`) and the shell trap pages |
| Callback or int64 input silently on numba even with `engine=rust` (B 4.3) | house rule float thresholds; probe counts Rust calls per live call; dispatcher-count/wall budget in selfcheck | wall watchdog (below) |
| Long first-call compile (numba path, 33-190 s MEASURED) | wall time of signal stage logged; `timeout` around `docker compose run` | **late-run gate:** if signals are ready after 12:55 PT (15:55 ET) place no orders and page (DAY orders after the 13:00 PT close queue to the next open, the exact slippage `paper_trade.sh:5-7` avoids). Add `timeout 600` to the compose run so a hang cannot overlap the next cron |
| pandas 3 changes: datetime64 unit/`str` dtype/CoW read-only arrays (B 6.2); `pd.read_sql` on a raw psycopg2 connection already warns (`cached_yfinance_loader.py:193`, log 11-04) | P1 frame-hash compare in shadow; scratch-Postgres round-trip test of read and write paths (B: only sqlite tested, UNVERIFIED for Postgres); `-W error::FutureWarning` lane on 2.3.3 first | write paths (`_cache_to_db`, `refresh_benchmark_tape`) must pass a unit-aware timestamp test before cutover; shadow uses RO role. Live-path `pct_change` sites are in blend/gate code, not in `generate_core_signals` (grep: `overlay.py:88`, `feature_gate.py:50`), but fix with the other 44 |
| Data gaps: symbol missing, stale bar, duplicate day rows (SPY had 826 dups; EA "possibly delisted" in every log) | existing freshness gate (as_of <= 4 d); add **input sanity gate** before trading: >=240 rows, universe_size >= 450 (live 503), last-bar NaN share < 5%, no duplicate index days, no weekend bars, buys <= 40 and sells <= 150 (a degenerate all-True frame would otherwise place orders alphabetically) | gate failure => no orders, Telegram with the failed check. Bounds are ESTIMATES from the warm-up study (about 8 buys, 47 sells per day); tune from shadow data |
| Wrong but plausible signals (definition drift, off-by-one warm-up) | shadow P2/P3; probation `SIGNAL_CROSSCHECK` | disagreement => no orders (never auto-resolved) |
| Memory | `maxrss` into `meta.json` and the log line; signal stage handles ~503 x 252 frames, vbt indicator outputs are same-sized (ESTIMATE < 1 GB); container cap 16G (`docker-compose.yaml:19`) | alert if maxrss > 2 GB; 137 exit hits the existing trap |
| Half-run (crash between sells and buys) | pre-existing: pending-order table and `_reconcile_pending_orders`; signals are computed in full before the first order, so the new engine adds no new half-run window | unchanged |
| Silent library drift on rebuild (09-23 plotly) | lock + `pip check` + smoke at build; image labels; daily check compares running tag to expected `GGTRADER_TAG` | build fails, nothing deployed |
| Cron or Docker never fires | the shell trap only covers a failed run, not a missing one. UNVERIFIED whether FlynnRX watches this | add a 13:10 PT heartbeat: if no `paper_snapshots` row for today's ET date, Telegram |

## 5. Skills and tests

**`ggtrader-deploy` (`.claude/skills/ggtrader-deploy/SKILL.md`).** (a) Step 0: read the freeze status from `docs/next_steps.md` and refuse without a recorded owner authorization id (A1/A2/A3). (b) Classification table: add "stack change (pandas/numpy/vbt major, base image)" requiring
shadow report + owner sign-off; add "`SIGNAL_ENGINE` / `GGTRADER_TAG` in `.env` only" (no rebuild, still needs steps 5b-7). (c) Step 4: record previous **tag and digest** (not only image ID), ensure the local alias tag exists, deploy by editing `GGTRADER_TAG`, not by trusting `:latest`. (d) Step 5a: replace the
vectorbt import line with `python -m ggTrader.engine.selfcheck` and print the version tuple. (e) Rollback section: replace "re-tag `:latest`" with ladder R1-R3; warn that `compose pull` and `image prune` orphan the old image. (f) `verify_live_path.py`: print engine provenance and run legacy vs vbt on today's frame (still no orders, mocked notifier). (g) Step 1 pre-flight: forbid `docker image prune` before the 20-session probation ends.
**`daily-trader-check`.** Step 1: compare running tag/digest with `.env` `GGTRADER_TAG` and with the last deploy record; print version tuple. Step 2: parse the new provenance line (`ENGINE=rust vbt=1.2.0 rust=1.2.0 signal_engine=vbt wall=..s maxrss=..`); flag `FALLBACK USED`, `SignalDisagreement`, sanity-gate and late-run messages, and run wall time over 120 s (log start and complete lines already exist). Step 6 tape check add duplicate-day and datetime-unit sanity. New step: shadow summary (days passed, class C diffs, P1-P6 status) while the shadow runs, then probation counter. Output block gains one `Engine:` line.

**Tests (all via `scripts/run_tests.sh`).**
* `tests/paper/test_signal_runner.py`: flag dispatch (default legacy; `vbt`; unknown raises); fallback only on infrastructure exceptions, never on disagreement; crosscheck raises on any buys/sells/last_exit difference and passes on equal; sanity-gate cases (small universe, NaN last bar, duplicate days, weekend bar, absurd buy count); selfcheck failure path with a monkeypatched probe; lookback 365 guard (existing `:349` test) extended to vbt path.
* `tests/paper/test_trader.py`: disagreement and sanity-gate failures call `notifier.send` and never `submit_buy`/`submit_sell`; late-run gate with a patched clock; **rule check:** keep new provenance and gate output in `signal_runner` and logs so `trader.py` gains no persist import. If a DB provenance write is ever added, add its `patch("ggTrader.paper.trader.<fn>")` to the autouse `_stub_write_path_db` fixture (`test_trader.py:27-45`) in the same commit or CI dies in `get_db_connection_string()` (CLAUDE.md rule; 42 tests died once).
* `tests/engine/`: dispatch (fails unless `rust`), defaults canary (S1 section 5), golden live frame (S5 line 142: 120 sessions of 365-day buys/sells, exact), parity of vbt-definition indicators vs pandas references on a fixed fixture, run with `-W error`.
* Loader tests on a scratch Postgres: round-trip of naive-UTC timestamps under pandas 3 (read, `_cache_to_db`, `refresh_benchmark_tape`), asserting no unit change and no tz shift.
* Image smoke script run by CI (section 3). No test may touch the real `.env`, DB or network (`conftest.py` makes `load_dotenv` a no-op).

## 6. Risk register

| # | Risk | P | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Unrelated push moves `:latest`, a pull ships the new stack to live early (happened twice, CLAUDE.md) | M | H | Migration stays on a branch; `GGTRADER_TAG` pin (A2); build smoke; ggtrader-deploy step 0 |
| 2 | vbt indicator definitions change signals, mistaken for parity | H if defaults used | H | Section 2.1 pre-registration; explicit flags; shadow reports O vs N and N vs lab |
| 3 | Rust silently off, numba cold start eats the window | L | H | Explicit `engine="rust"`, selfcheck, late-run gate, `timeout`; fallback is pandas not numba |
| 4 | pandas 3 corrupts a write path (timestamps, units) on the shared tape | M (UNVERIFIED on Postgres) | Very H (tape history) | RO role for shadow; scratch-DB write tests; `refresh_benchmark_tape` and gap-fetch tests before D1 |
| 5 | Alphabetical 3-slot selection amplifies a one-symbol diff | M | M | Compare would-be orders after cap; sanity bounds |
| 6 | Library drift on rebuild (SQLAlchemy 2.1, yfinance 1.7, alpaca-py 0.45) | H without lock | M-H | Lock + `pip check`; D1 isolates stack from indicators (O vs O') |
| 7 | Shadow reads a different partial bar than live | H | L (noise) | DB-only read at 12:50, O-vs-live baseline, primary test is O vs N on one frozen frame; in-process dual compute after lift |
| 8 | Freeze breach by well-meant shadow scheduling | M | M | A1 explicit authorization; no `APCA_*` in shadow env; no writes |
| 9 | Rollback image unavailable | L-M (retention untested) | H | Digest + `291e6dd` tag resolve today (MEASURED); keep local alias; pull-test before D1 |
| 10 | Silent no-run (cron/docker down) | L | M | 13:10 PT heartbeat |
| 11 | Young Rust path parity bugs (fixed through 1.1.x, B 4.5) | L | M | Live uses only indicators and last-bar signals; golden test both engines; `GGT_VBT_ENGINE=numba` research escape only |
| 12 | Scope: stock-pick sleeve may change before the port pays off (`changelog.md:268`) | UNVERIFIED | wasted hours | Confirm scope with owner before section 8 hours beyond P0 |
| 13 | Probation fallback hides a persistent engine fault | L | M | FALLBACK page on every use; auto-expire fallback after 20 sessions |

## 7. Timeline relative to the freeze and effort (ESTIMATE, agent-hours)

Today is 2026-11-06 (about 36 sessions to 12-29; 11-26 closed, 11-27 half-day, 12-25 closed).

| When | Step | Needs | Hours |
|---|---|---|---|
| 11-06 to 11-18 | **P0 branch work, no production touch:** lock + pins (shared with S2) 1.5; `signal_runner` flag/selfcheck/sanity/crosscheck module on branch 5; shadow harness (3-way, parquet freeze, compare, report) 6; offline O-vs-N replay over ~1,000 sessions 3; tests 4; skills/docs 2.5 | none | 22 |
| 11-18 | Ask owner for A1 (shadow scheduling, local image build, RO role, `.env.shadow`) and confirm scope (risk 12) | owner | 0.5 |
| 11-20 to 12-15 | **P1 forward shadow** 20+ sessions (~0.15 h/day review, class-C triage, scratch-DB write tests 2, report 2) | A1 | 8 |
| 12-15 to 12-29 | Report, pull-test the rollback digest, readiness review inputs; A2 (compose tag pin) if approved | owner | 2 |
| After readiness decision (date unknown) | **D0** workflow smoke + tag changes + Dockerfile/compose edits 3; **D1** stack cutover, `SIGNAL_ENGINE=legacy`, 5 sessions 2.5; **D2** flip to `vbt` with crosscheck, 20 sessions 4; **D3** crosscheck off, cleanup, docs, skills 1.5 | A3 per step | 11 |

Total about 43.5 agent-hours (ESTIMATE), of which 32.5 can be done before the freeze question is settled if A1 is granted, 22 without any authorization.
Excluded: the pandas-3 port of lab code, re-baseline runs (other workstreams). Earliest live indicator flip: D1 + 5 sessions + D2 start, i.e. about 8 weeks after the readiness decision lands if no step resets.

## Evidence files (this folder)
`coldstart.py`, `numba_cold.txt`, `numba_warm.txt` (cold-start probe); `req_new.txt`, `dry.log`, `report.json` (full-dependency dry-run resolve). Not verified: any real-tape signal flip (F open item 2), a Postgres write under pandas 3, the 12:45 partial-bar contents of the DB cache.
