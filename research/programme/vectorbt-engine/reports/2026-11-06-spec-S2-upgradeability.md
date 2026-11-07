# S2: Upgradeability, making vectorbt updates routine and safe

Date 2026-11-06. Author: S2 research agent (read-only on the repo). Prototypes and raw outputs: `.../scratchpad/vbt-spec/s2/` (`contract_fingerprint.py`, `fp_0285.json`, `fp_120.json`, `proto/uv.lock`, `proto/req.lock`, `ruffproto/`, `test_boundary_proto.py`, `cold.py`). Prior research: `vbt-research/F-synthesis-and-verdict.md`, `B-vectorbt-1x-and-rust.md`. Labels: MEASURED (I ran it), CITED (URL/file:line), ESTIMATE, UNVERIFIED.

## 0. Findings that change the design

1. **Production drift exists today.** `Dockerfile:21` runs `pip install -e ".[test]"` after `COPY . .`, so every source change re-resolves every unpinned dependency (numpy, pandas, numba, yfinance, alpaca-py...). `ci.yml:30` does the same, so a green CI does not test what ships. The 2026-09-23 crash (commit 833125e5) was one instance; the live image already resolved plotly 6.9.0 while the native venv has 6.6.0 (commit message of 833125e5). A lockfile is the first deliverable, before any engine work.
2. **Only 5 files import vectorbt** (MEASURED, AST walk): `src/ggTrader/lab/simulate.py:9`, `src/ggTrader/lab/strategies/signals.py:37`, and 3 scripts (`position_sizing_regimes.py`, `position_sizing_wfo.py`, `leveraged_trend_research.py`). The boundary is cheap to build now.
3. **Live cron pays the numba cold start on every run.** `compose run --rm` creates a fresh container each run, so numba's on-disk cache (`cache=True` is used 78 times in `vectorbt/portfolio/nb.py` and 85 times in `generic/nb.py`, MEASURED on 1.2.0) is lost each time. MEASURED today on 1.2.0 (empty cache dir, loaded box, one run): first `from_signals` + `value` + `trades.count` = **36.7 s numba cold, 5.4 s numba warm cache, 0.4 s Rust**. B measured 4-14 s. This is the one concrete operational benefit of Rust for the live path. It is not a throughput benefit (B: ~0).
4. **`.vbt` is a global pandas accessor.** Once `ggTrader.engine` imports vectorbt, any module can write `df.vbt.x` with no import, so an import ban alone leaks. The boundary test must also ban the attribute `.vbt` (section 3).
5. `uv lock` resolves the full dependency set with `vectorbt[rust]==1.2.0`, pandas 3.1.1, numpy 2.5.4, numba 0.68.1, plotly 7.1.0 in 1.7 s (MEASURED, 147 packages, `proto/uv.lock`, 2,327 lines with `requires-python=">=3.12,<3.13"`; 5,154 lines/151 packages with `>=3.11`). With `pandas-ta` kept it fails: "all versions of pandas-ta depend on numba==0.61.2 and your project depends on numba>=0.66" (MEASURED, `proto_pta`). `pandas-ta` has 0 imports (F, B8), so deleting `pyproject.toml:22` unblocks the stack.

## 1. Upstream cadence and breakage risk

PyPI upload dates (CITED https://pypi.org/pypi/vectorbt/json, `.../vectorbt-rust/json`; release bodies https://api.github.com/repos/polakowo/vectorbt/releases):

| Release | Date | Gap (days) | vectorbt-rust | Dependency floor / result change (release notes) |
|---|---|---|---|---|
| 0.28.4 | 2026-01-26 | | none | pandas `<3,>=2`, numba `>=0.60` |
| 0.28.5 | 2026-03-26 | 59 | none | same deps. `gross_exposure` abs fix for grouped portfolios (#837) |
| 1.0.0 | 2026-04-22 | 27 | 1.0.0 same hour | same deps + `[rust]` extra. New engine, new static signal simulator. Our `from_*` outputs bit-identical (B, 102/102 fuzz) |
| 1.1.0 | 2026-07-05 | 74 | 1.1.0 | **minor release raises floors: numpy>=2.4.6, pandas>=3.0.3,<4, numba>=0.66, Python>=3.11** (PyPI metadata). Empty-metric NaN change |
| 1.1.1 | 2026-09-26 | 83 | 1.1.1 | "**Results change**": deflated Sharpe, rolling std (Welford, 1.96e-9), expanding NaN |
| 1.1.2 | 2026-10-09 | 13 | 1.1.2 | `cond_value_at_risk` NaN handling, datetime64 conversion, NaN parity |
| 1.2.0 | 2026-10-19 | 10 | 1.2.0 | EWM std ddof fix (max abs diff 3.16, 29% rel, NaN mask changed, B), breakeven trades in expectancy, drawdown/OBV/unstack fixes |

- **Cadence:** 7 releases in 266 days (mean gap 44 d, median 27 d); 8 releases in the last 12 months. They cluster: 1.1.1, 1.1.2 and 1.2.0 landed within 23 days. Earlier 2025 gaps were 45-118 days (PyPI dates above). No 0.28.3 exists. Expect roughly 6-9 releases a year, in bursts.
- **Semver is not respected.** 1.1.0 (a minor) dropped pandas 2 and numpy <2.4.6. 1.2.0 (a minor) changed EWM-std numbers with no deprecation cycle. Nothing is labelled breaking; only 1.1.1 says "Results change". Every release since 0.28.5 contains at least one output-affecting item (ESTIMATE from reading the bodies above; only 1.0.0's simulator effect is verified neutral for our calls). **Treat every release as result-changing until the contract suite says otherwise.**
- **Rust tracks vectorbt in lockstep:** same version numbers, uploaded within ~1 hour; `vectorbt[rust]` requires `vectorbt-rust==<same>` (PyPI metadata for 1.2.0). The runtime check compares major.minor only and on mismatch only warns and falls back to numba (`vectorbt/_engine.py` `is_rust_available`, installed 1.2.0). Explicit `engine="rust"` then raises a misleading `ImportError` (B 4.3).
- **Maintenance risk:** contributor-driven (five first-time contributors in 1.1.2 alone), parity bugs found through 1.1.x (B 4.5), docs moved to Zensical, and the PRO edition holds the agent tooling. Licence is Apache-2.0 with Commons Clause (B 4.1). The 1.0.0 -> 1.2.0 numba kernel is 2.6x slower on numba 0.68 (B 3.2), so version choice has real performance consequences.
- **Realistic policy:** adopt every release, but never the day it ships. Research track: detect within 24 h, 3-day cooldown (Dependabot's default cooldown is 3 days, CITED GitHub options reference), PR with a diff report within 1 day after that, owner/agent decision within 7 days. Skipping is allowed only by recording a reason in the PR. A burst (two releases in <14 days) means evaluate only the newest. Live-image track: promote only after the research-track PR is merged, the live golden is identical, and a 7-day soak is complete. Never during the 12:30-13:15 PT window (`ggtrader-deploy` rule) and not before the freeze ends (~2026-12-29).
- **Pair rule:** the pair is atomic, because `vectorbt[rust]==X` pins the matching rust version. Never let a bot bump either one alone.

## 2. Dependency policy

**Tool: uv, with `uv.lock` committed.** Evaluated:

| Option | Verdict |
|---|---|
| uv (`uv lock`, `uv sync --frozen`) | **Choose.** Universal lock, hash-verified install, 1.7 s resolve (MEASURED), `uv lock --upgrade-package`, Dependabot supports the `uv` ecosystem (v0.11, CITED GitHub supported-ecosystems table). Already on the host (`~/.local/bin/uv` 0.11.25). Risk: single vendor, mitigated by `uv export --format requirements-txt` giving a hashed pip file (MEASURED: 139 hashed lines) as an escape hatch |
| pip-tools `pip-compile --generate-hashes` | Works but per-platform, slower, no Dependabot-native `uv`-style lock handling in this repo's setup. Second choice |
| Plain `pip -c constraints.txt` | Rejected: hand-maintained, no resolution check; this is today's failure mode |

**pyproject.toml (intent) vs uv.lock (truth).**
- Engine stack: `vectorbt[rust]==1.2.0` (exact; the one line the upgrade workflow edits), plus explicit bounded ranges that mirror vectorbt's own floors: `numpy>=2.4.6,<2.6`, `pandas>=3.0.3,<3.2`, `numba>=0.66,<0.69`, `plotly>=7,<8`. They are bounded so a transitive cap can never silently pick the version again (today `numba`, `numpy`, `pandas` are bare, `pyproject.toml:17-20`).
- Everything else: compatible ranges (`>=x,<next-major`), exact versions only in the lock. `ruff==0.15.12` stays exact.
- Delete `pandas-ta` (`pyproject.toml:22`) and the `plotly<7` cap comment (`:23-25`; vectorbt >=1.1.1 fixed the Plotly 7 template, B 2.1). Replace `vectorbt>=0.28.5,<0.29` (`:39`).
- `requires-python = ">=3.12,<3.13"` (matches `FROM python:3.12-slim`, `Dockerfile:2`; vectorbt 1.2.0 allows 3.11-3.14). It shrinks the lock by 55% (MEASURED). A Python bump is its own upgrade event, taken at most yearly, one minor behind the newest.
- A test (`tests/engine/test_lock_consistency.py`) parses `uv.lock` and asserts that installed `importlib.metadata` versions of the engine stack equal the lock, so a stale venv fails loudly.

**Docker consumption.** Replace `Dockerfile:18-21` (`COPY . .` then `pip install -e`) with: copy `pyproject.toml` + `uv.lock` first; `uv sync --frozen --no-install-project --extra test` into `/opt/venv` (uv copied from a pinned `ghcr.io/astral-sh/uv:<version>` image); then copy source and `uv sync --frozen`; pin the base image by digest (`python:3.12-slim@sha256:...`, digest UNVERIFIED, Dependabot `docker` ecosystem can bump it). Dependency layer then changes only when the lock changes. Keep `RUN python -c "import ggTrader.paper.trader"` (`Dockerfile:25`) and add `pip check` and `python -c "from ggTrader.engine import selfcheck; selfcheck()"`. Label the image with lock sha256 and the version tuple. UNVERIFIED: I did not build this Dockerfile.

**CI.** `ci.yml:30` becomes `uv sync --frozen --extra test --extra dev`, so CI tests the shipped set.

**Rollback.** (a) Image: `docker-build.yml` already pushes a `type=sha` tag per commit, so every image stays in GHCR. Make `docker-compose.yaml` reference `ghcr.io/garykuepper/ggtrader:<sha>` (or `@sha256:`), never `:latest`; rollback is one edit plus `docker compose up -d`. This also means a merge to main can never change the trader until the compose pin is edited. **Step 0 (no behaviour change):** pin compose to the digest that runs today (digest not inspected by me). (b) Code: `git revert` the lock-bump commit; `uv sync --frozen`. (c) Results: provenance (section 6) keeps old and new numbers separable.

## 3. Adapter boundary: `ggTrader.engine`

**One package, one rule: nothing outside `src/ggTrader/engine/` touches `vectorbt`, `vectorbt_rust`, or `.vbt`.** Public functions take and return pandas/NumPy and frozen dataclasses (for example `SimResult(value, trade_count, orders, per_column_returns)`). No vbt object crosses the boundary, so `"vbt.Portfolio"` in `simulate.py:88` goes away, and a rename affects exactly one file.

Enforcement, three layers:
1. **Ruff TID251** (MEASURED on ruff 0.15.12, `ruffproto/`): add `"TID"` to `select` in `ruff.toml`, `[lint.flake8-tidy-imports.banned-api] "vectorbt".msg=... "vectorbt_rust".msg=...`, and `[lint.per-file-ignores] "src/ggTrader/engine/**" = ["TID251"]`. It flagged `import vectorbt`, `from vectorbt.portfolio import x`, `import vectorbt.indicators`, `from vectorbt import RSI` and a function-local `import vectorbt`, and allowed the same lines inside `engine/`. It does not catch dynamic imports or `.vbt`. The existing Claude `PostToolUse` ruff hook and `.githooks/pre-commit` therefore enforce it on every edit.
2. **AST test** `tests/engine/test_boundary.py` (prototype `test_boundary_proto.py`, run on the repo: 5 violating files): walks `src/`, `scripts/`, `tests/`, `ggt.py` and fails on imports of the two modules, the attribute `.vbt`, and string constants `"vectorbt"`/`"vectorbt_rust"`. `tests/engine/` itself is allowed (contract tests compare the adapter against raw vbt).
3. Migration: the 3 legacy scripts move to the adapter; no per-file ignores remain at the end.

**Minimal API surface** (derived from F's primitive table and S1-style needs; explicit arguments only, no reliance on vbt defaults):

| Adapter function | vbt call | Pinned semantics |
|---|---|---|
| `simulate_orders`, `simulate_signals`, `hold` | `Portfolio.from_orders / from_signals / from_holding` | `cash_sharing`, `group_by`, `call_seq="auto"`, `size_type`, `sl_stop/sl_trail/tp_stop`, `freq`, `fees`, `slippage`; engine passed explicitly |
| `value`, `trade_count`, `returns` | `pf.value()`, `pf.trades.count()`, `pf.returns()` | no use of `trades.expectancy/profit_factor` (changed in 1.2.0) |
| `ma`, `mstd`, `rsi`, `bbands`, `macd`, `atr`, `stoch`, `obv` | `vbt.MA/MSTD/RSI/BBANDS/MACD/ATR/STOCH/OBV.run` | every semantic flag spelled out: `ewm`, `ddof`, window; RSI/MACD defaults are simple-MA (B 3.3), BBANDS default ddof=0 (F) |
| `sharpe`, `sortino`, `calmar`, `max_drawdown`, `ann_return`, `ann_vol`, `dsr` | `.vbt.returns(freq, year_freq=...)` accessors | `year_freq` always explicit; flat curve inf -> NaN guard; DSR not used until the contract test passes against an independent formula (1.1.1 changed it) |
| `crossed_above/below`, `and_/or_`, `stop_exit` | `.vbt.signals` accessors | |
| `rolling_folds` | `Splitter`/`.vbt.rolling_split` | only if S1 adopts it; bar-count based and drops the index (F), wrapper must restore dates |
| `configure`, `selfcheck`, `provenance`, `versions` | `vbt.settings["engine"]`, `_engine.is_rust_available/clear_engine_cache` | private `_engine` access allowed only here; `selfcheck` also fails if `vbt.settings["engine"]=="auto"` |

**Shim rules.** Version-conditional code is allowed only in `engine/_compat.py`, one function per upstream change, each with `# since vectorbt X.Y.Z: reason (release URL)` and a strict test that fails when the shim becomes unnecessary (so shims are deleted, not accumulated). Use `inspect.signature` gating for renamed kwargs, never `if version >` on semantics. Deprecation visibility: the engine contract tests run with `-W error::DeprecationWarning -W error::FutureWarning` (filtered to `vectorbt`/`pandas` modules). The adapter wraps vendor exceptions in `EngineError` (numba raises `ValueError`, Rust `RejectedOrderError` for the same fault, B 3.1).

## 4. Test strategy for upgrades

| Layer | What | Where | Cost (ESTIMATE) |
|---|---|---|---|
| L0 static | ruff TID251, `test_boundary`, `test_lock_consistency`, `pip check` | every PR | <15 s |
| L1 contract | Each adapter function vs an **independent reference** written in the test with plain NumPy/pandas (not vbt): SMA, EMA, Wilder RSI (first valid bar, zero-loss behaviour), BB ddof=1, MACD with both EWM flags, ATR (first-bar rule), Sharpe ddof=1 x sqrt(252), Sortino, CAGR on calendar days, max drawdown, flat-series NaN, DSR vs Bailey-Lopez de Prado formula. Tolerance `rtol=1e-9` | every PR | ~5 s |
| L1b fingerprints | Frozen JSON of raw vbt outputs we do **not** re-implement (portfolio value arrays, trade counts) | every PR | ~5 s |
| L2 golden | 3 seeded synthetic panels (60 symbols x 600 bars, PIT mask, NaN holes, a price gap), run through `simulate_signals`, `simulate_orders`, a 6-combo sweep and a 2-fold mini WFO using the real harness functions without the DB. Stored: equity arrays (npz, `rtol=1e-9`), trade counts (exact), fold winners (exact), metrics. Updated only via `pytest --update-golden`, which writes a provenance block and requires a changelog line | every PR, both engines | 1-2 min |
| L3 engine self-check | Wrap `vectorbt_rust.portfolio.simulate_from_signals_rs` with a counter (S1 prototype `s1_engine_selfcheck.py`, B `install_rust_counters`); assert calls>0 for `rust`, ==0 for `numba`; turn the mismatch `UserWarning` into an error; assert `settings["engine"]!="auto"`; run at start of `ggt paper`, `ggt lab` and in the image build | PR, image build, every live run (0.4 s with Rust) | <1 s |
| L3b parity | Golden under numba vs rust: simulators bit-identical (B), indicators `rtol=1e-9` | PR | 30 s |
| L4 image smoke | Build, `docker run image python -c "import ggTrader.paper.trader; selfcheck()"` as `appuser`, then push (today `docker-build.yml` pushes without testing) | docker-build | +1 min |
| L5 oracle (manual/agent) | F's Phase 0 pinned-window artefacts: 17-fold WFO on the real tape and the 60-day live golden. Needs the local tape, so it cannot run on GitHub runners; the runbook runs it on Encom before any merge that changes a number | upgrade runbook | ~1 h |

**Prototype evidence that contract tests earn their keep** (MEASURED, `contract_fingerprint.py` on 16 quantities, 400 seeded bars): running it on vectorbt 0.28.5/pandas 2.3.3 and on 1.2.0/pandas 3.1.1 gave 13 identical, 2 different at 1e-12 relative (Bollinger sums, float noise) and 1 real definition change (DSR 0.019057 -> 0.019276, the documented 1.1.1 fix). Lessons: exact equality is too strict (use `rtol=1e-9`), and fingerprints alone would simply be regenerated after a change, so L1 compares against independent formulas and golden regeneration is gated.

**PR CI** = L0, L1, L1b, L2, L3, L3b plus the existing suite (about +2-4 min, ESTIMATE). **Nightly** = latest-version job (section 5), `-W error` lane over the whole suite, B's 102-config fuzz parity (`fuzz_parity.py`, numba vs rust) and a from-scratch Docker build of main's lock (catches base-image and apt drift).

## 5. Automation

**`.github/workflows/vbt-upgrade-watch.yml`** (draft; actions pinned to majors like the repo's existing workflows):
```yaml
name: vbt-upgrade-watch
on:
  schedule: [{cron: "17 6 * * *"}]
  workflow_dispatch:
    inputs: {version: {description: "force a version", required: false}, cooldown_days: {default: "3"}}
permissions: {contents: write, pull-requests: write, issues: write}
concurrency: {group: vbt-upgrade, cancel-in-progress: false}
jobs:
  detect:
    runs-on: ubuntu-latest
    outputs: {version: "${{ steps.d.outputs.version }}"}
    steps:
      - uses: actions/checkout@v6
      - id: d   # newest vectorbt older than cooldown AND matching vectorbt-rust exists AND != locked pin
        run: python scripts/vbt_latest.py --cooldown "${{ inputs.cooldown_days || 3 }}" --force "${{ inputs.version }}" >> "$GITHUB_OUTPUT"
  try:
    needs: detect
    if: needs.detect.outputs.version != ''
    runs-on: ubuntu-latest
    timeout-minutes: 45
    strategy: {fail-fast: false, matrix: {engine: [numba, rust]}}
    env: {GGT_VBT_ENGINE: "${{ matrix.engine }}", VER: "${{ needs.detect.outputs.version }}"}
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7        # version pinned to the repo's uv
      - name: bump pair and re-lock
        run: |
          sed -i -E "s/vectorbt\[rust\]==[0-9.]+/vectorbt[rust]==${VER}/" pyproject.toml
          uv lock 2>&1 | tee lock.log || { echo RESOLVE_FAILED > status; exit 0; }
          uv sync --frozen --extra test --extra dev
      - name: suites (never fail the job; the report decides)
        run: |
          pytest tests/engine -q --junitxml=engine.xml || echo ENGINE_RED >> status
          pytest -q --junitxml=full.xml                || echo FULL_RED   >> status
          python scripts/vbt_fingerprint_diff.py --old origin/main --out report-${{ matrix.engine }}.md
      - uses: actions/upload-artifact@v5
        with: {name: "report-${{ matrix.engine }}", path: "*.md\n*.xml\nlock.log\nstatus\nuv.lock"}
  report:
    needs: [detect, try]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/download-artifact@v5
      - name: PR if resolvable, issue otherwise (no auto-merge, ever)
        env: {GH_TOKEN: "${{ secrets.GITHUB_TOKEN }}", VER: "${{ needs.detect.outputs.version }}"}
        run: python scripts/vbt_open_pr.py --version "$VER"   # body = release notes (gh api repos/polakowo/vectorbt/releases/tags/v$VER) + dependency delta + contract/golden failures + fingerprint diff; labels vbt-upgrade, needs-rebaseline if golden/fingerprint moved
```
Notes. (1) PRs created with `GITHUB_TOKEN` do not trigger other workflows (known GitHub behaviour, UNVERIFIED here), which is why the job runs the suites itself before opening the PR. (2) The repo setting "allow Actions to create PRs" must be enabled. (3) Merging to main triggers `docker-build.yml`, which is harmless once compose is pinned by sha. (4) Failure to resolve (a new transitive cap conflict, like `pandas-ta` today) opens an issue with `lock.log`, not a PR.

**`.github/dependabot.yml`** (draft): `version: 2`; `package-ecosystem: uv` weekly with `cooldown: {default-days: 7}`, `groups: {minor-patch: {update-types: [minor, patch]}}`, and `ignore:` for `vectorbt`, `vectorbt-rust`, `numpy`, `pandas`, `numba`, `llvmlite`, `plotly` (the watch workflow owns the engine stack so the pair is never split); also `github-actions` (monthly) and `docker` (monthly, base-image digest). Dependabot updates `uv.lock` only, so no separate exported requirements file is kept (the Dockerfile uses `uv sync --frozen`). Renovate is the alternative (richer grouping); not needed.

**Agent runbook** `.claude/skills/vbt-upgrade/SKILL.md` (outline; Sonnet orchestrates, Opus judges per CLAUDE.md delegation rules):
```
---
name: vbt-upgrade
description: Take a new vectorbt (+vectorbt-rust) release into ggTrader: read release notes, bump the pair, run contract/golden/oracle suites, triage diffs, decide ACCEPT / RE-BASELINE / REJECT. Use for a vbt-upgrade PR or issue, "new vectorbt version", "bump vectorbt".
---
0 Preconditions: not between 12:30-13:15 PT; freeze status read from docs/next_steps.md; no live files edited.
1 Read: gh api releases for every version between locked and target; list each item touching an adapter function (engine/*.py symbols) -> table "item | adapter function | expected effect".
2 Bump: edit ONLY the `vectorbt[rust]==X` line (+ floors if the new release raised them); `uv lock`; if unresolvable, name the conflicting package and stop (issue, do not loosen caps blindly).
3 Run: scripts/run_tests.sh tests/engine; full suite; scripts/vbt_fingerprint_diff.py; both engines.
4 Triage (decision tree): all green + fingerprints identical -> ACCEPT. Contract failure -> read the definition change; fix inside engine/_compat.py ONLY if the old definition is recoverable via explicit flags, else RE-BASELINE decision (owner). Golden moved but contract green -> inspect per-fold winners/gates; if any fold winner or gate flips -> escalate. Engine parity red -> pin numba, file upstream issue.
5 Oracle (Encom only): scripts/vbt_oracle.py on the pinned 17-fold window and 60-day live golden; diff against stored artefacts; record in research/.../oracle-<ver>.json with provenance.
6 Decide: ACCEPT (numbers identical within tolerance) | RE-BASELINE (new baseline epoch, regenerate goldens via --update-golden, write research note, snapshot refresh) | REJECT (cap `<next`, record reason + upstream issue in docs/vbt_upgrade_log.md).
7 Record: docs/changelog.md entry (what/why/results), docs/vbt_upgrade_log.md row (version, date, decision, items), bump `baseline_epoch` if re-baselined.
8 Promote to live ONLY via ggtrader-deploy (sha-pinned compose edit, selfcheck in container, next-run checklist). Never merge+pull in the same hour.
Red lines: never widen a version cap to make CI pass; never regenerate goldens without a written reason; never edit src outside engine/ for a pure upgrade.
```

## 6. Results provenance

**Fields** (`provenance`, schema 1): `vectorbt`, `vectorbt_rust` (or null), `engine_requested`, `engine_effective` (from `selfcheck`, not from settings), `rust_available`, `pandas`, `numpy`, `numba`, `llvmlite`, `scipy`, `python`, `lock_sha256` (of `uv.lock`), `git_sha`, `git_dirty`, `contract_version` (integer bumped when an adapter definition changes: ann factor, ddof, Sortino, CAGR convention), `baseline_epoch` (id of the current re-baseline, e.g. `E1`), `created_utc`. Built by `ggTrader.engine.provenance.provenance()` from `importlib.metadata` (never from hand-written strings).

**Where written.**
1. `lab_runs`: new nullable `provenance JSONB` via `ALTER TABLE lab_runs ADD COLUMN IF NOT EXISTS provenance JSONB` in `init_schema()` (`persist.py:101`, because `CREATE TABLE IF NOT EXISTS` in `_SCHEMA` will not add it). Additive only. `harness.py:69` passes it to `start_run`. Kept out of `params` so nothing that compares params changes (UNVERIFIED: I did not check every reader of `params`).
2. Research JSON artefacts: a helper `write_result_json(path, payload)` in `src/ggTrader/utils/results.py` injects a top-level `"provenance"`; `scripts/*_wfo.py` drivers switch to it (CLAUDE.md already requires JSON for any cited number). A test fails if a driver under `scripts/` writes JSON without it (grep for `json.dump` not routed through the helper).
3. Golden files and oracle artefacts carry the provenance of the run that produced them.
4. Live trader: one log line per run plus a Telegram alert when `engine_effective != engine_requested` or the version tuple differs from the previous run (needs a small state record; DB table choice left to the spec; the paper path stays read-only in S2).

**Comparison flags.** `compare(a, b)` returns the worst of: `SAME_STACK`; `ENGINE_DIFF` (numba vs rust, informational, differences ~1e-11); `PATCH_DIFF` (same vbt minor, other libs patch-level); `VBT_VERSION_DIFF` (different vectorbt version: comparable only if the same `contract_version` and the golden suite passed on both); `PANDAS_MAJOR_DIFF` / `CONTRACT_DIFF` / `EPOCH_DIFF` (**not comparable**: number deltas are the re-baseline, not an effect). `research-report-review` and the report template gain a "Comparable?" line, and any table quoting runs with different `baseline_epoch` must say so. Legacy results with no block read as `unknown (assumed vectorbt 0.28.5 / pandas 2.3.3, F)`; they are never silently mixed with epoch E1.

## 7. Failure modes and guardrails

| Failure | Detection | Mitigation |
|---|---|---|
| Numba cold start (4-14 s B; 36.7 s MEASURED once, loaded box) | wall time logged per live run; alert > 60 s | Live uses Rust (0.4 s). Fallback numba with `NUMBA_CACHE_DIR` on a named compose volume (a `run --rm` container's own filesystem is discarded). Do not bake the cache into the image: numba caches are CPU-specific (numba docs, UNVERIFIED here) and CI CPUs differ from Encom's i5-7500 |
| Numba cache dir not writable in Docker | `vectorbt` is installed root-owned, container runs as uid 1000 (`Dockerfile:36-38`); numba falls back to a user-wide dir (UNVERIFIED) | Set `NUMBA_CACHE_DIR=/app/.cache/numba` explicitly, `chown` it, smoke-test as `appuser` |
| Silent Rust fallback (`auto`; version mismatch; int64 close; callbacks, B 4.3) | `selfcheck` counters; mismatch `UserWarning` -> error; `settings["engine"]=="auto"` -> error | Never `auto`: adapter always passes `engine=`; no callback-based APIs in the surface; `float64` coercion in the adapter; lock forces the pair |
| Rust and numba disagree | L3b parity test and nightly fuzz | Differences expected ~1e-11 (B); numba is the reference engine, Rust the production engine only while parity is green; flip `GGT_VBT_ENGINE=numba` is a one-line rollback |
| pandas 3 silent changes (`pct_change()` default no longer pads; `agg(np.std)` ddof; epoch units; string dtype; CoW; B 6.2, F B6-B8) | `-W error::FutureWarning` lane on pandas 2.3 beforehand; golden diffs; 44 un-pinned `pct_change` sites listed by F | Make fill behaviour explicit in the adapter and call sites; unit tests per site; DB write path tested against a scratch Postgres (B: only sqlite tested) before any live use |
| Transitive cap conflict (`pandas-ta`, plotly, numba/numpy) | `uv lock` fails in the watch job -> issue with the resolver message | Resolver message names the culprit (MEASURED above); fix by removing or replacing the offender, never by loosening engine bounds |
| Import-time crash from a dependency (2026-09-23) | image-build import check (`Dockerfile:25`), now followed by `selfcheck` | Lock + sha-pinned compose: the trader can only change when the owner edits one line |
| Upstream yank or bad release | watch job sees it, or install fails | Exact pin + cooldown; REJECT path caps `<next` with reason |
| Lock/CI drift | `test_lock_consistency`; CI uses `--frozen` | Same lock everywhere |
| Rust wheel unavailable for a platform/Python | `selfcheck` raises | Wheels exist for CPython 3.11-3.14 on Linux x86_64/aarch64, macOS, Windows (B 4.4); sdist needs a Rust toolchain: stay on 3.12 manylinux |
| Numba kernel slowness on newer numba (2.6x, B 3.2) | benchmark smoke in nightly (`bench_small_calls`-style, threshold 1.5x) | Prefer Rust; record timing in provenance-adjacent nightly report |

## 8. Files and effort

**Create**
- `src/ggTrader/engine/__init__.py`: public API re-exports.
- `src/ggTrader/engine/portfolio.py`: `simulate_orders/signals/hold`, `SimResult`.
- `src/ggTrader/engine/indicators.py`: indicator wrappers with explicit flags.
- `src/ggTrader/engine/metrics.py`: returns/risk wrappers (explicit `year_freq`, inf guard).
- `src/ggTrader/engine/signals.py`, `splits.py`: signal helpers and folds (scope per S1).
- `src/ggTrader/engine/runtime.py`: `configure`, `selfcheck`, dispatch counters.
- `src/ggTrader/engine/provenance.py`: `provenance()`, `compare()`.
- `src/ggTrader/engine/_compat.py`: version-gated shims with strict removal tests.
- `src/ggTrader/utils/results.py`: `write_result_json` that stamps provenance.
- `tests/engine/test_boundary.py`, `test_lock_consistency.py`, `test_selfcheck.py`, `test_provenance.py`, `test_engine_parity.py`.
- `tests/engine/contract/` (one file per indicator/metric group), `tests/engine/golden/` (fixtures, `expected_*.npz/json`), `tests/engine/conftest.py` (`--update-golden`, engine parametrisation, `-W` filters).
- `scripts/vbt_latest.py`, `scripts/vbt_open_pr.py`, `scripts/vbt_fingerprint_diff.py`, `scripts/vbt_oracle.py`: watch-job helpers and the Encom oracle.
- `uv.lock`: the lock.
- `.github/workflows/vbt-upgrade-watch.yml`, `.github/dependabot.yml`.
- `.claude/skills/vbt-upgrade/SKILL.md` (+ `references/release-triage.md`).
- `docs/vbt_upgrade_policy.md`, `docs/vbt_upgrade_log.md`: policy, cadence data, decision log.

**Modify**
- `pyproject.toml`: drop `pandas-ta`, pin the engine pair, bounded numpy/pandas/numba/plotly, `requires-python`.
- `ruff.toml`: `TID` rules, banned-api, per-file ignore for `engine/`.
- `Dockerfile`: digest-pinned base, `uv sync --frozen`, dependency layer before source, smoke steps, `NUMBA_CACHE_DIR`.
- `docker-compose.yaml`: image pinned by sha/digest, `GGT_VBT_ENGINE`, numba cache volume (after the freeze, except the no-behaviour-change pin).
- `.github/workflows/ci.yml`: `uv sync --frozen`, engine matrix, boundary and `-W` lanes.
- `.github/workflows/docker-build.yml`: build, load, smoke, then push; image labels.
- `src/ggTrader/lab/simulate.py`, `src/ggTrader/lab/strategies/signals.py`: consume the adapter (remove direct `vectorbt`).
- `src/ggTrader/lab/persist.py`, `src/ggTrader/lab/harness.py`: provenance column and write.
- `scripts/position_sizing_regimes.py`, `position_sizing_wfo.py`, `leveraged_trend_research.py`, `scripts/*_wfo.py`: adapter and stamped JSON.
- `src/ggTrader/cli/` (`ggt paper`, `ggt lab` entry): call `selfcheck()` and log provenance.
- `tests/conftest.py`: engine pin fixture.
- `.claude/skills/ggtrader-deploy/SKILL.md` (sha-pinned promotion, selfcheck), `.claude/skills/research-report-review/SKILL.md` ("Comparable?" check), `CLAUDE.md` (engine rules), `docs/architecture.md`, `docs/changelog.md`.

**Effort (ESTIMATE, agent-hours, excluding the pandas-3 port of the 44 `pct_change` sites and the re-baseline runs, which belong to other workstreams):** lock + pyproject + `pandas-ta` removal 1.5; engine package + boundary lint + migration of the 5 files 4; contract tests 4; golden + parity 4; selfcheck + provenance + persist + JSON helper 3; CI, Docker and compose changes 3; watch workflow + helper scripts + Dependabot 3.5; runbook skill + docs 2; shake-out of the first dry-run upgrade 3. **Total about 28 agent-hours**, of which ~9 can land before the freeze ends if kept off the live path (lock, ruff boundary on scripts, watch workflow, Dependabot, tests, skill). Anything touching `lab/simulate.py`, `Dockerfile` or `docker-compose.yaml` is held for the post-2026-12-29 window with the first re-baseline; the compose sha-pin (step 0) is the only live-adjacent change worth doing now, and it changes no behaviour.

## Open uncertainties
- The Dockerfile described in section 2 was not built; digest pinning and the `uv` image copy are UNVERIFIED.
- Behaviour of numba's cache fallback under uid 1000 and CPU-specific cache keys is from numba documentation knowledge, not tested in a container.
- The 36.7 s cold start is one run on a loaded 4-core box; treat the range 5-37 s as the plausible band.
- GitHub token/PR-trigger behaviour and the Actions-create-PR setting are from memory, not re-checked.
- Whether Dependabot's `uv` ecosystem handles the `extras`/`[rust]` pin edit cleanly is UNVERIFIED; the design avoids needing it by ignoring the engine stack.
