# Refactor phases 0 to 2 (safety nets, dead code, shared plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put in place the safety nets every later refactor depends on, then make the low-risk
clean-ups (dead code, stale text, duplicated plumbing, logging, annotation style). Neither step
may change what the trader trades or what any research number comes out as.

**Architecture:** Tests come first. Phase 0 adds tests that only observe today's behavior:
- a fail-closed guard against real database and network connections;
- a check that every `ggTrader` name a study imports still resolves;
- an import-direction check between packages;
- a transcript test that records every outside call the paper trader makes over 38 scenarios;
- a characterization test of the price-loader chain.

Phases 1 and 2 then change code. After each change, the transcripts and the oracle results must
stay identical, unless the task names a deliberate behavior change.

**Tech Stack:** Python 3.12 (uv-managed `.venv`), pytest through `scripts/run_tests.sh`, ruff
0.17.4, SQLAlchemy and psycopg2, typer, pandas.

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md),
the consolidated design. "Spec 0.4" and similar are the slice labels of the original code review, which is not kept;
the design's phase table (section 7) lists the same work by task number.

**Revision 4 (2026-11-17)** applies the GPT-6.2 Astra independent review
(a working file that is not kept; design section 9 records it).
Each item was checked against the code before it was applied. The changes:
- **Oracles:** compared with a baseline taken from the untouched base commit, not with
  "IDENTICAL" verdicts, because 16 of the 26 committed p5 reports say `DIFFERENCES`.
- **Offline wrapper:** a scrubbed-environment wrapper, plus a check that no database address
  can be built. The host has no process-level network block.
- **Guard:** it fails on attempts that the code swallows.
- **Test outcomes:** a node-by-node check replaces skip-list matching.
- **Transcript harness:** per-call broker replies, orders that already exist before the run,
  expected-outcome docstrings, and four new scenarios (38 in all).
- **Narrowed tasks:**
  - Task 7 keeps both config dicts.
  - Task 10 moves only the engine cache, because the URL helpers differ: first match against
    every match.
  - Task 12 leaves `make_tape_snapshot.py` alone, because the legacy tag lacks
    `utils/manifest.py`.
- **Logging:** no claims about secrets in logs, and no changes to third-party loggers.
- **Task 14:** corrected bugbear hits, and the CI pin change moved here from Task 10.

**Revision 3 (2026-11-17)** applies the verified items of the Gemini 4 Argon review
(a working file that is not kept; design section 9 records it):
the p5 compare file's top-level `peak_rss_mb` and empty folders (Task 0), the run-lock
context manager, the membership token, notifier argument recording and the order-status
cursor (Task 4), hostname lookups and `Curl.perform` (Task 1), a stderr-following log handler
(Task 13), and loggers for the two modules without one (Task 14). Not applied, because the
code shows otherwise:
- I-A4: the trader does not call `load_ohlcv` or `compute_sleeve_curve`. They are called in
  `signal_runner.py:318-320`, behind the patched `generate_core_signals`.
- I-A6 cites tasks, methods and scenario names that do not exist.
- B-A6's 135 test and research hits are already covered by the planned per-file ignores.
- B-A5's `+psycopg` form is unchanged by all three helpers, so they agree.

**Revision 2 (2026-11-17)** applies the Fable plan review
(a working file that is not kept; design section 9 records it):
blocking items B1 to B8, important items I1 to I26, and the minor items. Fable also wrote the
spec, so its fidelity check is not independent; the pre-merge Sol review repeats that check.

## Scope

**In this plan:** spec slices 0.1 to 0.5, 1.1 to 1.3, 1.5 and 2.1 to 2.5, plus the bugbear and
try-except-pass lint rules from design section 5.4.

**Left out on purpose, each to a later plan or the owner:**
- Spec Phases 3 to 6. Their designs depend on the transcripts and the characterization test
  built here, so each gets its own plan once Phase 0 lands.
- Slice 1.4, the unused database writers (owner decision O3).
- From finding 17: the `max_stocks` duplicate, the hard-coded bar field names, `subset_bars`,
  and the four hand-rolled repo roots in `src/`. The roots differ from `utils.paths` outside an
  editable install (`utils/paths.py:13-29`), so changing them is not behavior-neutral.
- From finding 6: the two crypto-era tables in `_db_enable_compression`, and printing row counts
  before a wipe. This plan prints the wipe's conditions as text, which needs no database
  connection.
- From finding 11: a test that a dry run prints its "DRY RUN buy" line on stderr. That needs
  `run_paper_trading`, which the Phase 4 plan covers.
- From finding 20: the nine hand-written annotations, which move to Phase 4 with the paper
  trader's types.
- `run_paper_trading` (`trader.py:1561-1605`) stays outside the transcript harness. The Phase 4
  plan adds scenarios through it.
- Merging `canonical_hash` and `git_sha`. The duplicates behave differently
  (`verify_pins.py:172-176` against `utils/json_safe.py:24-27`; `universe_db.py:234-256`
  against `utils/binding.py:27-30`).
- Any edit to the oracle, legacy and standard-library-only scripts: `p5_wfo_validation.py`,
  `p7_live_golden.py`, `legacy_reference_runs.py`, `legacy_reference_compare.py`,
  `verify_pins.py`, `verdict_register.py`, `study.py` and `vbt_latest.py`.

## Owner decisions this plan needs before execution

A default is not an approval. Tasks 6 and 9 do not start until the owner's answer is recorded
here, with the date.

**Recorded 2026-11-17:** O1 approved (delete). O2 approved (require the flags). O3 keep.
O4 both (2026-11-18): show the percentage against the day's starting value and against
starting capital; a separate slice after Task 5, out of this plan. Execution: not yet. Nothing in this
plan runs until the owner says so.

- **O1, the dead "untradable" branch (Task 6).** The trader has code meant to skip a stock the
  broker says it cannot trade and carry on, but that code is never called. Today a rejected
  order stops the whole run, and the tests pin that behavior.
  *Default: delete the dead code; behavior is unchanged.*
- **O2, the wipe commands (Task 9).** `ggt db truncate` empties the 5.7-million-bar price table
  and `ggt db clean` deletes rows, both without asking.
  *Default: require an explicit flag; without it, print what would be deleted and exit with no
  database connection.* This is a deliberate behavior change.
- **O3, unused database writers (not in this plan).** *Default: keep them for now.*
- **O4, the Telegram profit percentage (not in this plan).** `paper/notifier.py:151` divides
  total profit by the day's starting value, not by starting capital. Owner answer: show both
  percentages. The change alters the Telegram text, so it needs its own slice with the
  transcript goldens regenerated and reviewed.

## Global Constraints

- **Branch and worktree.** Work on branch `refactor/phase-0-2`, created from the `main` commit
  recorded in Task 0 step 1 as `BASE`, in a worktree at `/home/flynn/ggTrader-refactor`. Its
  environment is built with `uv sync --frozen --extra test --extra dev` inside the worktree.
  Every standalone command uses the worktree's own `.venv/bin/python` and `.venv/bin/ruff` by
  absolute path. Never commit to `main`, push or merge without the owner's say-so.
- **Offline by construction.** The real boundary is that no process in this work can build a
  database address or hold a service key. `utils/config.py:17-41` raises when no `DB_*` or
  `POSTGRES_CONNECTION_STRING` is set; the code has no default address. A process-level network
  block is not available on this host: unprivileged user namespaces are restricted, and
  `IPAddressDeny` has no effect on user scopes (both checked 2026-11-17). So:
  - Never copy `.env` into the worktree.
  - Every test, oracle and script command runs with a scrubbed environment through the wrapper
    `/home/flynn/ggTrader-refactor/offline.sh` (Task 0). It runs
    `env -i HOME="$HOME" PATH="$PATH" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LANG=C.UTF-8 TZ=UTC GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02 "$@"`,
    `XDG_RUNTIME_DIR` is kept because `run_tests.sh` and `run_capped.sh` start their cgroup with
    `systemd-run --user --scope`, which fails without it (checked 2026-11-18: exit 1 with it
    missing, exit 0 with it set). It is not a credential. So no database, Alpaca, Telegram,
    Tiingo or test-override variables (`DB_*`,
    `POSTGRES_CONNECTION_STRING`, `GGTRADER_DB_URL`, `GGT_TASK2_TEST_DSN`, `APCA_*`,
    `ALPACA_*`, `TELEGRAM_*`, `TIINGO_*`, `PYTEST_ADDOPTS`, `GGT_REGEN_TRANSCRIPTS`,
    `GGT_TEST_*`) can leak in.
  - The `integration` marker stays deselected (the default of every tier).
  - Task 1's in-process guard is a second layer for the new tests. It is not the boundary.
- **Tests.** Run tests only through the worktree's `scripts/run_tests.sh` (fast core),
  `scripts/run_tests.sh --full` (merge gate) or `scripts/run_tests.sh <paths>`. Never a bare
  `pytest`.
  - A `137` exit means a hang. Look for a real DB, network or clock access.
  - A `--full` gate passes only through the test-outcome check below.
- **No real connections from tests.** No test may reach the real database, Alpaca, Telegram,
  yfinance, Tiingo or the network. Never run `ggt paper` with `--live`, never touch cron, never
  read `.env`.
- **Research code.** Never edit anything under `research/`: study code is bound to its results
  by a code hash. Every `ggTrader` name a study imports stays importable, re-exported if it
  moves.
- **Fixed rules.** Only `src/ggTrader/engine/` imports vectorbt, and nothing in `src/` imports a
  study. `ohlcv.timestamp` is naive UTC; never introduce a tz-aware write.
- **Code standards.** `ruff check .` and `ruff format --check .` stay clean after every task.
  New code follows `CLAUDE.md` section 4: type hints, Google docstrings, a module docstring,
  constants at module top.
- **Commits** end with the attribution line from the session instructions.
- **The oracle check.** Wherever a task says "oracles identical". The comparison is against
  the baseline taken from the untouched `BASE` commit, not against "IDENTICAL" verdicts. The
  p5 comparison reports compare the new engine with the pre-rebuild engine, and 16 of the 26
  committed reports say `DIFFERENCES - classify`. That is the accepted state (see
  `research/programme/vectorbt-engine/P5-completion.md`), and it must stay exactly as it is.
  1. Run both drivers through `scripts/run_capped.sh` and `offline.sh`, each into a new empty
     folder, with `--force`:
     ```
     scripts/run_capped.sh .venv/bin/python scripts/p5_wfo_validation.py --runs all --force --out-dir /tmp/ggtrader_refactor/<label>/p5
     scripts/run_capped.sh .venv/bin/python scripts/p7_live_golden.py --out /tmp/ggtrader_refactor/<label>/p7.json
     ```
  2. Both must exit 0.
  3. The 13 run result files `_p5_<run>.json` (not the `_p5_compare_*` reports) must each have
     `"ok": true` as a JSON boolean. Each run's comparison report and `*_oos_equity.csv` must
     be present. The expected file list is recorded once in Task 0.
  4. p7 must report the same 120 sessions as the baseline, with `sessions_different == 0`.
  5. `offline.sh .venv/bin/python scripts/compare_evidence_json.py /tmp/ggtrader_refactor/baseline /tmp/ggtrader_refactor/<label>`
     must exit 0. This requires every comparison report's verdict and details to equal the
     baseline's, including the `DIFFERENCES` ones.

  The committed evidence shows about 100 seconds for p5 and 31 seconds for p7, so run the check
  after every task that changes `src/` or a script the oracles load (Tasks 6, 7, 10, 11, 12,
  13 and 14), not only at the phase gates.
- **The test-outcome check.** Matching skip lists are not enough: a removed or deselected test
  produces no skip. `run_tests.sh --full`
  starts two pytest processes (`tests/`, then the study folders), and each one overwrites the
  same `--junitxml` file, so a single `--full -rA --junitxml=...` records only the study half
  (found 2026-11-18: the first baseline held 1144 study tests and no core tests). Each gate
  therefore runs two commands, one file each:
  ```
  offline.sh scripts/run_tests.sh --full -rA --junitxml=/tmp/ggtrader_refactor/<label>-core.xml tests
  offline.sh scripts/run_tests.sh --studies -rA --junitxml=/tmp/ggtrader_refactor/<label>-studies.xml
  ```
  (a path argument makes the first a single run of `tests/` in the full tier; the second ends
  with the study folders, whose run is the one that writes the file). The checker
  `scripts/compare_test_outcomes.py` (Task 0) takes comma-separated files per side:
  `compare_test_outcomes.py baseline-core.xml,baseline-studies.xml <label>-core.xml,<label>-studies.xml`.
  It compares the two sets as follows:
  - every node that passed in the baseline still passes;
  - every skip is in the baseline with the same reason, with line numbers stripped;
  - the named real-tape cases pass rather than skip (`tests/paper/test_live_golden.py` and
    the tests marked by `tests/lab/wfo/helpers.py:110-116`);
  - the tests each task adds appear and pass.

  Before every gate, check that `run_tests.sh` actually ran pytest: the XML file exists and
  holds at least the baseline's node count.
- **The verify-pins check.** `scripts/verify_pins.py --expect <counts in .github/workflows/ci.yml:147>`
  exits 0, and the per-pin listing (path, digest, resolution) is unchanged except for pins a
  task deliberately touches. A task that edits a pinned file updates that CI line in the same
  commit. Today the only `MATCH` pin is `src/ggTrader/data/live/cached_yfinance_loader.py`,
  which Task 14 edits.

## Review Focus

1. **A transcript that cannot fail.** Arguments may be dropped, notifier text not rendered, or a
   raise not recorded. Task 4 adds a mutation check and records the rendered Telegram text and
   any raise.
2. **Nondeterminism in transcripts.** The sources are `uuid4`, the trader's `time` name,
   `datetime.now` at `trader.py:379`, set iteration order across processes (`trader.py:576`),
   NaN and infinity values, and object reprs. Task 4 fixes each one and checks determinism in a
   second process with another `PYTHONHASHSEED`.
3. **Real connections.** The boundary is a scrubbed environment with no credentials (`offline.sh`),
   checked in Task 0. Task 1's guard blocks psycopg2, curl_cffi and Python sockets as a second
   layer, and fails even when the code swallows the error.
4. **Study imports broken by a move.** Examples are `lab.persist.get_engine`, the
   `paper.persist._get_engine` alias that tests patch 18 times,
   `universe_db.to_psycopg_url`, and `live_signals.MIN_HISTORY_BARS` and
   `live_ensemble_frames`. Task 2 checks imported names, and Task 10 keeps every alias.
5. **A wipe command run without its flag from a script.** Task 9's tests show no engine is
   created and the exit is immediate, with no prompt.

---

### Task 0: Worktree, frozen tape, plan files, comparer and oracle baseline

**Files:**
- Create: `scripts/compare_evidence_json.py`
- Create: `scripts/compare_test_outcomes.py`
- Test: `tests/scripts/test_compare_evidence_json.py`, `tests/scripts/test_compare_test_outcomes.py`
- Create, outside the repository: `/home/flynn/ggTrader-refactor/offline.sh` (git-ignored by
  living at the worktree root under a name in `.git/info/exclude`)
- Add to the branch: this plan and the spec (both untracked in the main checkout today).

**Interfaces:**
- Produces: `compare(before: Path, after: Path) -> list[str]`, the list of differences (empty
  means identical). CLI: `compare_evidence_json.py BEFORE AFTER` exits 0 if identical and 1
  otherwise, printing each difference.
  - BEFORE and AFTER are folders, matched by relative path. A file on one side only is a
    difference.
  - `.json` files are compared as JSON with exact types (`true`, `1` and `1.0` all differ),
    ignoring only an explicit list of volatile paths, each named with its source line:
    - `provenance.created_utc`, `provenance.git_sha`, `provenance.git_dirty`
      (`engine/provenance.py:141-166`);
    - `wall_seconds`, `diagnostics.peak_rss_mb`, `diagnostics.timings_s`, and the top-level
      `peak_rss_mb.new` of each `_p5_compare_*.json` (`p5_wfo_validation.py:322`);
    - any other path found volatile in step 9, added with its source line.

    Nothing is ignored by key name at any depth.
  - Any JSON file containing `"ok": false` is itself a difference.
  - An empty folder, or a missing folder, on either side is a difference. p7's JSON has no
    `ok` key; the oracle check tests its `sessions_different` and its session list
    separately.
- Produces: `compare_test_outcomes.py BASELINE.xml NEW.xml [--must-pass NODEID ...]`, which
  exits 0 when the rules of the test-outcome check in Global Constraints hold, and 1
  otherwise, listing each breach.
  - Every other file (the `*_oos_equity.csv` files) is compared byte for byte.

- [ ] **Step 1: Worktree and environment.**
  ```
  git -C /home/flynn/ggTrader worktree add /home/flynn/ggTrader-refactor -b refactor/phase-0-2 main
  cd /home/flynn/ggTrader-refactor && uv sync --frozen --extra test --extra dev
  .venv/bin/python -c "import ggTrader; print(ggTrader.__file__)"
  ```
  The last command must print a path under `/home/flynn/ggTrader-refactor/src/`. Record
  `BASE=$(git rev-parse main)` in the plan's hand-back notes. The pre-commit hook comes from the
  shared repository config; check it with `git config --get core.hooksPath` rather than setting
  it.
- [ ] **Step 1b: Offline wrapper and boundary check.** Write `offline.sh` as given in Global
  Constraints, `chmod +x` it, and add its name to `.git/info/exclude`. Then run
  `./offline.sh .venv/bin/python -c "from ggTrader.utils.config import get_db_connection_string as g; g()"`.
  Expected: it raises `ValueError: Database configuration not found`. If it returns an
  address, stop: the worktree environment can reach the database.
- [ ] **Step 2: Frozen tape.**
  ```
  cp /home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet data/snapshots/
  sha256sum data/snapshots/ohlcv_20261106.parquet
  ```
  The hash must equal the one in `data/snapshots/MANIFEST.json` (`6573e840…789b2e`), and
  `git status --short data/` must be empty (the tape is git-ignored).
- [ ] **Step 3: Plan files.** Copy this plan and the spec from the main checkout into the same
  paths in the worktree and commit them:
  `docs: refactor design (Fable code review) and phase 0-2 plan`.
- [ ] **Step 4: Write the failing tests.**
  - `test_identical_folders_compare_equal`
  - `test_provenance_git_sha_ignored`
  - `test_provenance_tape_sha_not_ignored`
  - `test_unlisted_seconds_key_not_ignored`
  - `test_bool_and_int_differ`: `true` against `1` is a difference
  - `test_differences_verdict_equal_to_baseline_passes`
  - `test_outcomes_lost_pass_fails`, `test_outcomes_new_skip_fails`,
    `test_outcomes_must_pass_skipped_fails`, `test_outcomes_missing_node_fails`
  - `test_nested_value_difference_is_reported`: `{"a": {"b": [1, 2]}}` against
    `{"a": {"b": [1, 3]}}` gives `["x.json: a.b[1]"]`.
  - `test_float_compared_exactly`: `0.1` against `0.10000000000000002` differ.
  - `test_ok_false_is_a_difference`
  - `test_csv_compared_bytewise`
  - `test_missing_file_reported`
  - `test_empty_folder_is_a_difference`
  - `test_top_level_peak_rss_ignored`
  - `test_cli_exit_codes`
- [ ] **Step 5:** Run `scripts/run_tests.sh tests/scripts/test_compare_evidence_json.py`. Expected: FAIL.
- [ ] **Step 6:** Implement the script with `main()` and `argparse`; `parse_args` runs before
  any other work, which `tests/scripts/test_scripts_have_help.py` requires.
- [ ] **Step 7:** Run the new test and `tests/scripts/test_scripts_have_help.py`. Expected: PASS.
- [ ] **Step 8: Baseline.** On `BASE`, before any code change, run the oracle check's
  drivers with label `baseline`. Record the expected file list: the 13 named run results, their
  comparison reports and equity files, and `p7.json` with its 120 sessions.
  - Check every `ok` is `true`.
  - Check each comparison report's verdict against its committed copy in
    `research/programme/vectorbt-engine/p5-evidence/`.
  - Check that p7 matches `research/programme/vectorbt-engine/p7-evidence/p7_live_golden.json`
    on `sessions_different` and the session list.

  A mismatch with the committed evidence is a finding to report before any refactor starts.
- [ ] **Step 9: Volatile fields.** Run the drivers a second time into `baseline2` and compare
  the two with the comparer. This only discovers volatile paths; it is not a pass criterion.
  - A path that differs between the two runs may be added to the ignore list only if it is a
    time, a memory reading or provenance, and only with its source line.
  - Any differing number in a result is a finding to report.
- [ ] **Step 10:** Run the two gate commands of the test-outcome check with label `baseline`,
  on the untouched `BASE` tree. Keep `baseline-core.xml` and `baseline-studies.xml` as the
  test-outcome baseline. Expected:
  - the real-tape and live-golden tests pass rather than skip;
  - no test reaches the network. The environment has no credentials, so a test that tries
    fails loudly instead.
- [ ] **Step 11:** Commit: `test: evidence and test-outcome comparers for refactor checks`.

### Task 1: Fail-closed connection guard (spec 0.1)

**Files:**
- Create: `tests/netguard.py`, a plain module so tests can import it.
- Create: `tests/paper/conftest.py`, which applies the guard to every paper test.
- Test: `tests/test_netguard.py`

**Interfaces:**
- Produces, in `tests/netguard.py`:
  - `class RealConnectionAttempted(RuntimeError)`
  - `EXEMPT_NODEIDS: frozenset[str] = frozenset()`. The opt-in real-database test
    (`test_paper_persist.py:610`) is not exempted. It skips because `offline.sh` removes
    `GGT_TASK2_TEST_DSN`, and if the variable ever leaked, the guard would fail it.
  - `guard_applies(nodeid: str, markers: set[str]) -> bool`, which is `False` only for tests
    marked `integration` (deselected in every tier anyway) and `True` otherwise. A `db`
    marker does not exempt.
  - `blocked_connections() -> ContextManager[None]`, which patches:
    - `socket.socket.connect` and `socket.socket.connect_ex`, for `AF_INET` and `AF_INET6`
      only;
    - `psycopg2.connect`;
    - `socket.getaddrinfo` and `socket.gethostbyname`, which block hostname lookups such as
      the one in `utils/config.py:43-50`;
    - curl_cffi's lowest common call, `curl_cffi.Curl.perform` (yfinance uses a
      `curl_cffi.requests.Session`), so session, module-level and direct calls are all
      covered.

    Each patch first records the attempt (call name and destination) in a list owned by the
    context manager, then raises `RealConnectionAttempted`. On exit, the context manager
    raises `RealConnectionAttempted` listing every recorded attempt. That raise happens
    outside the application code, so a caller that swallows the first error, such as
    `TelegramNotifier.send` (`notifier.py:32-43`) or the broad excepts in `lab/data.py`,
    still fails the test. The guard is entered before any collaborator is constructed.
    Python sockets are covered on the paths Requests and urllib3 use (Alpaca, Telegram,
    Tiingo), psycopg2 on fresh connections, and curl_cffi on synchronous `Curl.perform`. The
    async curl_cffi path is not used by this code base and is out of scope. The process
    boundary remains the missing credentials.
  - The fixture `no_real_connections(request)`, which applies `blocked_connections()` when
    `guard_applies(request.node.nodeid, {m.name for m in request.node.iter_markers()})`.

- [ ] **Step 1: Write the failing tests.**
  - `test_inet_socket_connect_raises`: connect to `("127.0.0.1", 1)`.
  - `test_psycopg2_connect_raises`: `psycopg2.connect("postgresql://u:p@127.0.0.1:1/x")`
    raises `RealConnectionAttempted` inside the guard. Port 1 never reaches the real
    database, even if the guard failed.
  - `test_sqlalchemy_postgres_engine_raises`: the same address through
    `create_engine(...).connect()`, where the cause chain contains `RealConnectionAttempted`.
  - `test_sqlite_engine_allowed`: `create_engine("sqlite://").connect()` works.
  - `test_unix_socketpair_allowed`
  - `test_curl_cffi_request_raises`: both `curl_cffi.requests.get(...)` and a
    `curl_cffi.requests.Session().get(...)` raise.
  - `test_hostname_lookup_raises`: `socket.gethostbyname("example.invalid")` raises.
  - `test_swallowed_attempt_still_fails`: code that catches the guard's error and carries on
    still makes the guard's exit raise, listing the attempt.
  - `test_guard_applies_decision`: a plain node and a node marked `{"db"}` give `True` (a
    `db` marker does not exempt); `{"integration"}` gives `False`; `EXEMPT_NODEIDS` is empty.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/test_netguard.py`. Expected: FAIL.
- [ ] **Step 3:** Implement `tests/netguard.py` and `tests/paper/conftest.py`.
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/paper/ tests/test_netguard.py`. Expected: PASS.
  A paper test failing with `RealConnectionAttempted` is a real finding: stub it in that test
  and name it in the commit message.
- [ ] **Step 5:** Commit: `test: paper tests fail closed on any real database or network connection`.

### Task 2: Study-facing import surface test (spec 0.2)

**Files:**
- Create: `tests/study_surface.py`, holding the scanner.
- Create: `tests/utils/test_study_surface.py`

**Interfaces:**
- Produces: `study_imports(root: Path) -> list[StudyImport]`, where `StudyImport` is the frozen
  dataclass `(path: str, line: int, module: str, name: str | None)`.
  - It scans every `import ggTrader…` and `from ggTrader… import …`, including imports inside
    functions.
  - It covers the files matched by `tests/tiers.py` `STUDY_TEST_GLOBS` folder roots, with
    `code` and `_lib` widened to all `.py` files beneath them:
    `research/*/*/code/**/*.py` and `research/*/_lib/**/*.py`.
  - `KNOWN_UNRESOLVED: frozenset[tuple[str, str]] = frozenset()`. The review found all 53
    distinct pairs resolve today.

- [ ] **Step 1: Write the tests.**
  - `test_every_study_import_resolves`
  - `test_known_unresolved_is_still_unresolved`
  - `test_scan_finds_imports`: at least 150 imports, including
    `("ggTrader.lab.persist", "get_engine")`.

  The module docstring says the test checks imported names only, not `module.attr` uses; none
  exist today.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/utils/test_study_surface.py`. Expected: PASS.
  If a pair fails on the untouched tree, stop and report it rather than listing it.
- [ ] **Step 3:** Commit: `test: every ggTrader name a study imports must keep resolving`.

### Task 3: Import-direction rules (spec 0.3)

**Files:**
- Modify: `tests/utils/test_import_rules.py`. Add `LAYER_RULES`, `LAYER_EXCEPTIONS` and
  `test_layer_rules`; the existing tests stay unchanged.

**Interfaces:**
- Produces:
  - `LAYER_RULES = {"utils": ("lab", "paper", "cli"), "data": ("lab", "paper", "cli"), "engine": ("lab", "paper", "cli"), "lab": ("paper", "cli")}`
  - `LAYER_EXCEPTIONS: frozenset[tuple[str, int, str]]`, a set of
    `(file relative to src/ggTrader, line, imported module)`. Use the module form `a.b`, not
    the `a.b.name` form that `_imported_modules` also yields; deduplicate per line.
  - The review measured today's exceptions as `utils/fred_data.py` lines 92, 118 and 136, and
    `utils/significance.py:49`.

- [ ] **Step 1:** Write `test_layer_rules`. It fails on any violation not in
  `LAYER_EXCEPTIONS` and on any exception that no longer occurs.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/utils/test_import_rules.py`. Expected: PASS
  with exactly the four exceptions above. A different list means the scan is wrong; fix the
  scan, not the list.
- [ ] **Step 3:** Commit: `test: import-direction rules between ggTrader packages`.

### Task 4: Paper-trader transcript harness and goldens (spec 0.4)

**Files:**
- Create: `tests/paper/transcript/__init__.py`
- Create: `tests/paper/transcript/harness.py`
- Create: `tests/paper/transcript/scenarios.py`
- Create: `tests/paper/test_trader_transcripts.py`
- Create: `tests/paper/golden/trader_transcripts/<scenario>.json`

**Interfaces:**
- Produces, in `harness.py`:
  - **`Scenario`**, a frozen dataclass with these fields:
    - `name`, `dry_run: bool`, `read_only: bool | None`, `membership: str`, `env: dict[str, str]`;
    - `today_et: date`, `now_pt: datetime`;
    - `replies: dict[str, list[object]]`, the ordered replies or exceptions per broker
      method (`get_clock`, `get_account`, `get_positions`, `get_untradable`,
      `get_split_evidence`, `get_dividend_corrections`). Each call takes the next entry, and
      the last entry repeats once the list runs out. This covers, for example, a first
      account read that succeeds, a sell, and then a failing post-trade read;
    - `existing_orders: list[dict]`: orders the broker already holds before this run, each
      with `id`, `client_order_id`, `symbol`, `side`, `qty` or `notional` and its status
      sequence. These serve the legacy pending-order poll and the startup intent lookup
      (`trader.py:216-252`, `:374-398`);
    - `signals: dict | Exception`;
    - `persist: dict[str, object]`, the read return values or exceptions, keyed by persist
      function name;
    - `intents: list[dict]`, the starting rows of the in-memory intent store;
    - `fills: dict[str, list[dict]]`, the per-symbol status sequence for orders submitted
      in this run. Each entry carries `status`, `filled_qty` and `filled_avg_price`;
    - `untradable: set[str]`, `splits: dict`, `dividends: dict`.

    Live scenarios use `membership="db"`, so the real `require_live_membership_spec` guard
    runs. Dry-run scenarios use the fixture spec the root `conftest.py` sets.
  - **`FakeBroker`**, which implements every public method of `AlpacaBroker`
    (`alpaca_broker.py:36-240`), records each call, and answers from the scenario.
    - `submit_buy` and `submit_sell` return order IDs `ord-1`, `ord-2`, … in submission
      order. The broker remembers each order's symbol, side, amount and client order ID.
    - `get_order(order_id)` keeps one cursor per order ID: the nth poll of an order returns
      the nth entry of its status sequence. After the list runs out it repeats the last entry,
      and an empty list means filled at once.
    - `get_order_by_client_id(client_id)` returns the last state already returned for that
      order without advancing the cursor, or the first entry if it has not been polled. The returned dict carries the `id`, `client_order_id`,
      `symbol`, `side` and `notional` or `qty` that `_reconcile_order_intents`
      (`trader.py:219-265`) checks, taken from the remembered submission.
  - **`RecordingNotifier(TelegramNotifier)`** records the arguments of `trade_alert` and
    `daily_summary` and then calls the parent method, so the real formatting runs. It
    overrides `send` to record the rendered text and return `True` without any network call.
    After the parent `__init__`, it sets `_enabled = True`, so formatting is never skipped
    for lack of credentials. The parent `__init__` calls `_load_env()`, which the root
    `conftest.py` already makes a no-op.
  - **`FakeRunLock`**, with a `verify()` method that records each call. `paper_run_lock` is
    patched with a context manager that yields one `FakeRunLock` and records enter and exit.
    The trader enters it with `with paper_run_lock() as lock:` (`trader.py:722`), and
    `_submit_order` calls `lock.verify()` (`:194`).
  - **`live_membership_source`** is patched to return a frozen `MembershipToken(spec: str)`.
    `to_json` writes it as `{"membership": spec}`, because the trader passes it straight to
    `generate_core_signals` (`trader.py:761-763`).
  - **Recording rules.** Each run starts from `copy.deepcopy` of the scenario. Each recorded
    call snapshots its arguments with `copy.deepcopy` at call time, because the trader mutates
    some persist replies. Finite floats stay JSON numbers. Special values are encoded as
    `{"float": "NaN"}` (and likewise for the infinities), so they cannot collide with a real
    string. A harness error (a missing reply, an unserializable value) raises
    `HarnessError` and fails the test outright. It is never recorded as a trading outcome.
  - **`IntentStore`**, an in-memory stand-in for the intent ledger. It implements
    `create_order_intent`, `attach_broker_order`, `get_unresolved_order_intents`,
    `settle_order_intent` and `settle_legacy_pending_order`, modeled on `_intent_store` in
    `tests/paper/test_trader.py:153-200`, and records every call.
  - **`run_scenario(s: Scenario) -> dict`**, which returns
    `{"calls": [{"target", "args", "kwargs", "raised"}], "result": dict | None, "raised": str | None, "skipped_writes": list[str]}`.
    A run that raises records `"raised": "<Type>: <message>"` and `"result": None`.
  - **`to_json(t: dict) -> str`** writes canonical JSON:
    - sorted keys and a two-space indent;
    - floats via `repr`; NaN and infinity as the strings `"NaN"`, `"Infinity"` and
      `"-Infinity"`;
    - dates ISO, sets sorted, `Decimal` as a string;
    - any other unknown type raises `TypeError`, so no memory address can reach a golden.
- **Patching rules.** All patches are on the `ggTrader.paper.trader` module namespace only:
  - **Persist functions:** every name the module imported from `ggTrader.paper.persist`,
    found by inspection. The intent-ledger names route to `IntentStore`. `paper_run_lock`
    becomes the `FakeRunLock` context manager, not a value-returning wrapper. Every wrapper keeps
    the real function's `__name__`, because `_write` records `fn.__name__` (`trader.py:175`).
  - **Other collaborators:** `generate_core_signals`, `refresh_benchmark_tape`,
    `live_membership_source`, `_today_et` and `_now_pt`.
  - **IDs:** `uuid4`, returning `UUID(int=n)` from a counter.
  - **Clock:** replace the module's `time` name with a fake object (fake `sleep`, `time` and
    `monotonic` on one fake clock) using `patch("ggTrader.paper.trader.time", fake)`. Each
    `sleep` advances the fake clock by its argument, and the harness raises `HarnessError`
    after 200 polls in one run. Never patch the global `time.time`; that pattern caused the
    2026-09-22 hang.
  - **Environment:** clear every variable read by `src/ggTrader/paper/*.py` (find them by
    grepping `os.environ` and `getenv`), then apply `env`.
- **Determinism rules:**
  - Scenarios with pending orders set `created_at=None`, because `trader.py:379` reads
    `datetime.now(timezone.utc)` unpinned.
  - Split scenarios use one applied symbol, because `trader.py:576` iterates a set.

- [ ] **Step 1: Write `scenarios.py`** with these 34 scenarios (38 after the additions below):
  - **Ordinary runs:** `plain_buys_and_sells`, `market_closed`, `late_run_live`, `late_run_dry`.
  - **Signal timestamps:** `stale_as_of`, `future_as_of`, `unparseable_as_of`.
  - **Untradable names:** `untradable_holding`, `untradable_candidate`.
  - **Corporate actions:** `split_correction`, `new_dividend_accrual`.
  - **Risk halts:** `drawdown_halt`, `daily_loss_halt`, `daily_loss_halt_with_split`,
    `peak_unreadable`, `peak_invalid`.
  - **Catastrophe stop:** `catastrophe_stop`, `catastrophe_stop_pending_order`,
    `catastrophe_stop_unassessable`, `catastrophe_stop_invalid_qty`.
  - **Cash sweep:** `sweep_funding_sell`, `sweep_buy`.
  - **Fills and pending orders:** `sell_partial_fill`, `pending_symbol_buy`,
    `pending_symbol_sell`.
  - **Position limits:** `global_slot_cap`, `sleeve_slot_cap`, `concentration_limit`,
    `missed_exit_catch_up`.
  - **Dry runs:** `dry_run_read_only`, `dry_run_with_writes`.
  - **Failures:** `submit_error`, `post_trade_read_failure`, `snapshot_write_failure`.

  Each scenario's inputs come from the existing test of that behavior in
  `tests/paper/test_trader.py`, cited in a comment. `plain_buys_and_sells`,
  `sell_partial_fill` and `sweep_buy` must exercise the `IntentStore` round trip
  (create, attach, read back, settle).

  Each scenario also states, in a docstring, its expected terminal outcome: the exact halt or
  error text, or a successful result; the discriminating calls in order with their amounts;
  and the final settled and unresolved intent rows. Specific requirements:
  - `sell_partial_fill` splits into `sell_partial_fill_working`, where the order is still
    `partially_filled` at the deadline, blocks the run and stays unresolved, and
    `sell_partial_fill_canceled`, where the order is canceled after a partial fill, settles,
    and blocks replacement buys. That makes 35 scenarios.
  - Add `sweep_sell_and_strategy_sell`, with a sweep funding sell and at least one strategy
    sell, so swapping the two blocks changes the transcript (36 scenarios).
  - Add `sell_rejected_by_broker`: the broker accepts the sell, then reports the order as
    `rejected`. It goes through normal settlement and a second, unrelated exit is still
    submitted (`trader.py:252-282`). Task 6 deletes the code next to this path, so the
    behavior must be pinned first (37 scenarios).
  - Add `strategy_exit_invalid_qty`: a strategy exit whose held quantity is invalid
    (`trader.py:1157-1160`) blocks all buys and the sweep buy (`:1360-1361`). The existing
    `catastrophe_stop_invalid_qty` covers only the stop path (`:1106-1109`) (38 scenarios).
  - `global_slot_cap`, `sleeve_slot_cap` and `concentration_limit` each include a candidate
    that would trade if that limit alone were lifted. They use real `RiskConfig` values, not
    a mocked decision.
  - A strategy sell's quantity equals the held quantity, and the dividend scenario scripts
    both the dividend total read before the write and the one read after.

    The seven rules in design section 5.1 each need a scenario whose golden shows the rule, and
  a `test_golden_shows_scenario_behavior` assertion for it:
  - stopped symbol not sold twice or re-bought: `catastrophe_stop`;
  - slot arithmetic counts regular sells only: `sweep_sell_and_strategy_sell`, with a buy that
    fits only when the sweep sell is not counted;
  - a sell that does not fully fill blocks all buys: `sell_partial_fill_working`;
  - an invalid exit quantity blocks buys and the sweep buy: `strategy_exit_invalid_qty` and
    `catastrophe_stop_invalid_qty`;
  - orders use the broker's uncorrected quantity: `split_correction`, where the golden's sell
    quantity equals the broker's held quantity and differs from the corrected one;
  - a read-only run writes nothing: `dry_run_read_only`;
  - a submit exception ends the run, a `rejected` status settles: `submit_error` and
    `sell_rejected_by_broker`.

  The scenario count elsewhere in this task is 38.
- [ ] **Step 2: Write `test_trader_transcripts.py`** with these tests:
  - `test_transcript_matches_golden[<scenario>]`, which shows a unified diff on failure;
  - `test_transcript_is_deterministic[<scenario>]`;
  - `test_transcripts_stable_across_hash_seeds`, which runs every scenario in two
    subprocesses, with `PYTHONHASHSEED=1` and `PYTHONHASHSEED=2`, and compares each with the
    goldens. Each subprocess runs inside `blocked_connections()`, with `GGT_MEMBERSHIP` set to
    the fixture spec, `GGT_REGEN_TRANSCRIPTS` unset, and `load_dotenv` patched to a no-op,
    because pytest's monkeypatches and the root fixture do not reach a child process;
  - `test_every_scenario_has_a_golden` and `test_no_orphan_goldens`;
  - `test_golden_shows_scenario_behavior[<scenario>]`, with one assertion per scenario that
    the golden shows what its name claims. Examples: `drawdown_halt` has no `submit_buy`;
    `dry_run_read_only` has no persist write and a non-empty `skipped_writes`; `submit_error`
    has `raised` set, after a recorded `create_order_intent`.

  With `GGT_REGEN_TRANSCRIPTS=1` set, the golden test writes the file and then fails with
  "regenerated, review and rerun".
- [ ] **Step 3:** Run `GGT_REGEN_TRANSCRIPTS=1 scripts/run_tests.sh tests/paper/test_trader_transcripts.py`
  (expected to fail, regenerating the goldens), then run the same command without the variable.
  Expected: PASS.
- [ ] **Step 4: Mutation check (not committed).** Make each change below, rerun the transcript
  tests and confirm at least one fails, then revert:
  - change one character of the drawdown-halt notice in `trader.py`;
  - swap the sweep sell and the strategy sells;
  - change one character of a line in `notifier.py` `daily_summary`.

  Afterwards, `git status` must show both files clean.
- [ ] **Step 5:** Run `scripts/run_tests.sh tests/paper/`. Expected: PASS, with no
  `RealConnectionAttempted`.
- [ ] **Step 6:** Commit: `test: transcript goldens for the paper trader over 38 scenarios`.

### Task 5: Price-loader chain characterization test (spec 0.5)

**Files:**
- Create: `tests/lab/test_fetch_stock_ohlcv_chain.py`

**Interfaces:**
- Consumes: `ggTrader.lab.data.fetch_stock_ohlcv` (`src/ggTrader/lab/data.py:58`), unchanged;
  `tests/netguard.blocked_connections` (Task 1).
- Produces: an autouse fixture that applies `blocked_connections()` and, for every test,
  patches all four seams with fakes:
  - `ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader`;
  - `ggTrader.data.live.yfinance_loader.YFinanceDataLoader`;
  - `ggTrader.data.live.tiingo_loader.TiingoDataLoader`;
  - `ggTrader.lab.negative_cache.load_skip_symbols` and `record_no_data`.

  A test changes a fake's behavior, never removes the patch. Tickers are made up (`ZZTEST1`,
  `ZZTEST2`, …). `frame(symbols, days)` builds the `(symbol, field)` frame.

- [ ] **Step 1: Write the tests.** Each asserts the returned symbols and rows, the recorded
  persist calls and the captured stdout lines:
  - `test_cache_hit_returns_cache_only`
  - `test_cache_constructor_fails_falls_back_to_yfinance_without_writer`
  - `test_cache_empty_falls_back_to_yfinance_and_keeps_writer`
  - `test_yfinance_empty_raises_value_error`
  - `test_gap_filled_by_yfinance_is_persisted`
  - `test_gap_persist_failure_is_reported_and_ignored`
  - `test_gap_filled_by_tiingo_is_persisted`
  - `test_tiingo_unavailable_is_reported`
  - `test_negative_cache_skips_and_records`
  - `test_min_coverage_drops_sparse_symbols`
  - `test_window_trimmed_to_start_and_end`
  Two cache routes are distinct and each needs its own test:
  - a cache that constructs and returns an empty frame keeps its writer attached, so later
    gap fills are persisted;
  - a cache whose constructor or fetch raises clears the writer, so nothing is persisted.

  The plain yfinance fallback requests all tickers, including negative-cache names, and the
  negative cache affects only the later gap lists (`lab/data.py:79-184`). Each test states the
  exact requests, the persist payloads and their order, and the output rows. One test covers
  a symbol obtained only from the second provider request.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/lab/test_fetch_stock_ohlcv_chain.py`.
  Expected: PASS on today's code. If a test fails, inspect the observed call sequence and
  record the discrepancy in the test's docstring before changing the expectation. The code is
  not changed in this task.
- [ ] **Step 3:** Commit: `test: characterize the fetch_stock_ohlcv fallback chain`.

**Phase 0 gate:** the test-outcome check passes against the baseline, with the Task 1 to 5
tests present and passing. The orchestrator reads every golden transcript once, and an Opus reviewer checks
that each golden shows what its scenario claims.

---

### Task 6: Remove the dead untradable-skip branch (spec 1.1; gated on O1)

**Files:**
- Modify: `src/ggTrader/paper/trader.py`. Remove `_order_failed` (`:697-707`), the
  `untradable_hit` set (`:920`) and its alert (`:1422-1427`). At `:1556` the result becomes
  `"untradable": sorted(untradable_seen)`. Remove the `is_untradable_error` import from
  `trader.py` only.
- Modify: `src/ggTrader/paper/untradable.py:10-12`. The docstring should say:
  - untradable names are filtered before ordering;
  - an exception raised by the broker on submit is re-raised and ends the run
    (`trader.py:199-210`);
  - an order the broker later reports as `rejected` goes through normal settlement, which can
    leave other exits eligible (`trader.py:252-282`).
- Keep: `is_untradable_error` itself; `tests/paper/test_untradable.py:38-56` uses it.

- [ ] **Step 1:** Make the removal.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/paper/ tests/utils/test_study_surface.py`.
  Expected: PASS, with the goldens unchanged and not regenerated.
- [ ] **Step 3:** Oracles identical (label `t6`).
- [ ] **Step 4:** Commit: `refactor(paper): drop the never-called untradable-skip branch; a rejected order still aborts the run`.

### Task 7: The unused `names` parameter, and documenting unread config keys (spec 1.2, narrowed)

**Files:**
- Modify: `src/ggTrader/lab/data.py:19-52`
- Modify: `src/ggTrader/lab/wfo/run.py:527-533` and its one caller at `:446`

Both config dicts stay exactly as they are. Current command paths copy them into downstream
code (`lab/cli.py:142-155,178`), so removing a key is not provably neutral. The unused keys are
documented instead.
- [ ] **Step 1:** Remove the unused final parameter `names` of `_select_live_params`
  (`run.py:527-533`) and its one argument at `:446`. Keep the other `names` uses in `run.py`.
- [ ] **Step 2:** Add a comment above each dict listing the keys nothing in `src/` reads, found by
  searching each quoted key across `src`, `scripts` and `tests`. Say that they are kept because
  the dicts are copied into run records and the legacy runner.
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/lab/ tests/utils/test_study_surface.py`.
  Expected: PASS. Then check that the oracles are identical (label `t7`).
- [ ] **Step 4:** Commit: `refactor(lab): drop an unused parameter; document unread config keys`.

### Task 8: Documentation fixes (spec 1.3)

**Files:**
- Modify: `docs/architecture.md:246-299`. Replace the stale strategy list and the `Strategy`
  protocol with the real contract, citing `lab/strategies/contract.py`.
- Modify: the stale comments and help text at `src/ggTrader/lab/wfo/cli_support.py:1-6`,
  `src/ggTrader/paper/trader.py:352`, `src/ggTrader/paper/trader.py:1532`,
  `src/ggTrader/lab/data.py:25` and `src/ggTrader/cli/main.py:8-9`.
- Modify: `docs/changelog.md`, with one entry for Tasks 0 to 8.

- [ ] **Step 1:** Make the text edits. Only comments, docstrings and the typer `help=` string
  change.
- [ ] **Step 2:** Run `scripts/run_tests.sh` and `scripts/run_tests.sh tests/paper/test_trader_transcripts.py`.
  Expected: PASS.
- [ ] **Step 3:** Commit: `docs: correct stale architecture, help and comment text`.

### Task 9: Confirmation flags for the wipe commands (spec 1.5; gated on O2)

A deliberate behavior change.

**Files:**
- Modify: `src/ggTrader/cli/cmd_db.py`. `truncate` gains `--yes-wipe-ohlcv` and `clean` gains
  `--yes`, both `store_true`. Without its flag, `run_db` prints one line naming what would be
  deleted ("all rows of ohlcv" for truncate; the two `DELETE` conditions verbatim for clean)
  and the flag that allows it, then calls `sys.exit(2)` before any engine is created. With the
  flag, behavior is unchanged.
- Modify: `docs/cli_reference.md`, documenting both flags.
- Test: `tests/scripts/test_db_wipe_flags.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_truncate_without_flag_exits_2_and_creates_no_engine`: patch
    `ggTrader.cli.cmd_db.create_db_engine_or_exit` with a mock; assert `SystemExit` code 2 and
    that the mock is never called.
  - `test_clean_without_flag_exits_2_and_creates_no_engine`
  - `test_truncate_with_flag_runs_truncate`: the mock engine's `begin()` context gets exactly
    one `execute` with the SQL `TRUNCATE TABLE ohlcv;`.
  - `test_clean_with_flag_runs_both_deletes`
  - `test_help_mentions_flags`
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/scripts/test_db_wipe_flags.py`. Expected: FAIL.
- [ ] **Step 3:** Implement the flags.
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/scripts/`. Expected: PASS.
- [ ] **Step 5:** Commit: `fix(cli): ggt db truncate and clean require an explicit flag`.

**Phase 1 gate:** the test-outcome check passes, the oracles are identical (label `phase1`),
and the verify-pins check passes.

---

### Task 10: One home for the cached engine (spec 2.1, narrowed)

The URL-rewrite consolidation is dropped. The three helpers differ today:
`universe_db.to_psycopg_url` replaces the first occurrence (`universe_db.py:216`), while
`data/bars.py:185` and `cached_yfinance_loader.py:55` replace every occurrence. Each keeps its
own behavior. Only the engine cache moves.

**Files:**
- Create: `src/ggTrader/utils/db.py`, which holds the module-level cache `_ENGINE` and
  `get_engine() -> sqlalchemy.Engine`, both moved unchanged from `lab/persist.py:15,102-106`.
- Modify: `src/ggTrader/lab/persist.py`, which imports `get_engine` from `utils.db` so the
  name stays in its namespace. Its functions keep calling the module-level name, so a test
  that sets `persist.get_engine` by attribute (`tests/lab/test_persist_provenance.py:33`)
  still works.
- Modify: `src/ggTrader/paper/persist.py:13`, which becomes
  `from ggTrader.utils.db import get_engine as _get_engine`. The alias `_get_engine` stays,
  because `tests/paper/test_paper_persist.py` patches it 18 times.
- Modify: `src/ggTrader/utils/fred_data.py` (lines 92, 118 and 136), which imports the engine
  from `utils.db`.
- Modify: `tests/utils/test_import_rules.py`. Drop the three `fred_data.py` exceptions;
  `significance.py:49` stays as the one remaining exception.
- Leave unchanged: `utils/significance.py` (it imports `curve_summary`, not the engine),
  `utils/db_engine.py` (an uncached per-call engine), every URL rewrite, and
  `scripts/make_tape_snapshot.py:72`. No pinned file changes in this task, so the CI pin
  counts stay as they are.
- Test: `tests/utils/test_db_helpers.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_get_engine_is_cached`: with `create_engine` and `get_db_connection_string` patched,
    two calls return the same object. The test saves and restores `utils.db._ENGINE` around
    itself, so no later caller receives its mock.
  - `test_old_names_are_the_new_objects`: `lab.persist.get_engine is utils.db.get_engine`
    and `paper.persist._get_engine is utils.db.get_engine`.
  - `test_attribute_patch_on_lab_persist_still_applies`: setting `lab.persist.get_engine` to
    a fake, as `tests/lab/test_persist_provenance.py:33` does, makes `lab.persist.init_schema`
    use the fake.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/utils/test_db_helpers.py`. Expected: FAIL.
- [ ] **Step 3:** Implement the move and the aliases.
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/utils/ tests/paper/ tests/data/ tests/lab/`.
  Expected: PASS, transcripts unchanged and the study surface green. Then run the verify-pins
  check (counts unchanged) and confirm the oracles are identical (label `t10`).
- [ ] **Step 5:** Commit: `refactor: one home for the cached DB engine`.

### Task 11: Shared paper constants (spec 2.2, partial)

**Files:**
- Create: `src/ggTrader/paper/constants.py`, holding `TERMINAL_STATUSES: tuple[str, ...]` (today's order),
  `ET = ZoneInfo("America/New_York")` and `PT = ZoneInfo("America/Los_Angeles")`.
- Modify: `src/ggTrader/paper/trader.py` (`:79-80`, `:252`, `:391`, `:449`),
  `src/ggTrader/paper/persist.py:108`, `src/ggTrader/paper/signal_runner.py:51` and
  `src/ggTrader/paper/gates.py:39`, all to import from `constants`. Names that tests or studies
  import (`gates.PT`, and any other the study-surface test lists) stay as module attributes.

- [ ] **Step 1:** Write `tests/paper/test_constants.py::test_terminal_statuses_equal_each_legacy_set`
  and `test_time_zones_match_legacy`, with today's literals copied in and their source lines
  cited. The review found the four status sets equal as sets.
- [ ] **Step 2:** Make the replacements. `TERMINAL_STATUSES` is a `tuple` in today's order,
  not a `frozenset`, so a non-hashable status value from a broker still compares the way it
  does today. `ET` and `PT` stay bound at their existing module names (`trader._ET`,
  `trader._PT`, `gates.PT` and the rest), re-bound from `constants`.
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/paper/ tests/utils/`. Expected: PASS,
  transcripts unchanged. Then check that the oracles are identical (label `t11`).
- [ ] **Step 4:** Commit: `refactor(paper): one definition of terminal order states and time zones`.

### Task 12: Scripts clean-up (spec 2.3, narrowed)

**Files:**
- Modify: delete the `sys.path.insert(0, "src")` line in `scripts/equity_backfill.py:20`,
  `scripts/backfill_missing_members.py:38` and `scripts/fetch_midcap400_snapshot.py:16`.
- Modify: `src/ggTrader/data/snapshot.py:63` and `src/ggTrader/data/membership_export.py:96`.
  Each delegates to `ggTrader.utils.manifest.sha256_file` and keeps its own name as an alias.
- Leave unchanged: `scripts/make_tape_snapshot.py:216`. The legacy reference runner executes
  this script by path on top of the tagged `legacy-engine-vbt0285` package
  (`legacy_reference_runs.py:144-153,757-762`), and that tag has no `utils/manifest.py`.
- Leave unchanged: `scripts/study.py:274` and every script in the "Left out on purpose" list.
  Leave the `Path(__file__).resolve().parents[1]` roots alone in all scripts: a script's own
  root is always its own checkout, while `utils.paths.REPO_ROOT` follows the installed package
  (`utils/paths.py:28-29`).
- Test: `tests/scripts/test_hash_helpers_agree.py`

- [ ] **Step 1:** Write `test_file_hash_helpers_agree`: every file-hash function above,
  `utils.manifest.sha256_file` and `scripts/study.py:274` (imported by path, not changed) give
  the same hex digest on a temporary file of 0 bytes, 1 byte and 3 MiB. Run it. Expected: PASS
  before any change; if not, stop and report.
- [ ] **Step 2:** Make the replacements.
- [ ] **Step 3:** Run `scripts/run_tests.sh tests/scripts/ tests/utils/ tests/data/`, then run
  each of the three changed scripts with `--help`. Run them with the worktree interpreter
  from `/tmp`, through `offline.sh`, with no `PYTHONPATH`, so an injected source path cannot
  hide a broken import. Then run the verify-pins check and the oracle check (label `t12`).
  Expected: PASS and identical.
- [ ] **Step 4:** Commit: `refactor(scripts): drop relative sys.path edits; one file-hash helper`.

### Task 13: Logging set-up (spec 2.4)

**Files:**
- Create: `src/ggTrader/utils/logging_setup.py` with `configure(level: str | None = None) -> None`.
  - It attaches one handler to the `ggTrader` logger, not the root. The handler is a small
    `StreamHandler` subclass whose stream is looked up as `sys.stderr` at each emit. A handler
    that keeps the first stream would hold a closed buffer after typer's `CliRunner` swaps
    stderr in tests.
  - The format is `%(asctime)s %(levelname)s %(name)s: %(message)s`.
  - The level comes from `level`, then the environment variable `GGT_LOG_LEVEL`, then `INFO`.
  - It is idempotent.
  - The `ggTrader` logger gets `propagate = False`, so a root handler configured elsewhere
    cannot print each line twice.
  - Third-party loggers are not touched. Today urllib3's retry warnings already reach stderr
    through Python's last-resort handler, and may contain the Telegram URL with its token
    (`notifier.py:13,36-41`; urllib3 `connectionpool.py:872-884`). This task does not change
    that and makes no claim that logs are free of secrets. Redacting that path is a
    follow-up for the owner.
  - An invalid `GGT_LOG_LEVEL` falls back to `INFO` and logs one warning naming the bad value.
    It never aborts a run.
- Modify: `src/ggTrader/cli/main.py`, adding a typer `@app.callback()` that calls `configure()`.
- Modify: `src/ggTrader/data/live/yfinance_loader.py:20,55` and
  `src/ggTrader/data/live/tiingo_loader.py:44`. Each uses `logging.getLogger(__name__)` with
  %-style arguments and the same message text, so their lines fall under the `ggTrader`
  logger.
- Test: `tests/utils/test_logging_setup.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_configure_routes_ggtrader_info_to_stderr`
  - `test_no_duplicate_with_root_handler`: with a root handler already installed, one
    `ggTrader` line appears exactly once.
  - `test_third_party_levels_untouched`: the levels of `urllib3` and `yfinance` are the same
    before and after `configure()`.
  - `test_invalid_level_falls_back`: with `GGT_LOG_LEVEL=LOUD`, the level is INFO and a
    warning is emitted.
  - `test_configure_is_idempotent`
  - `test_handler_follows_stderr_swap`: configure, swap `sys.stderr` for a new buffer, log,
    and the line lands in the new buffer.
  - `test_env_level_respected`
  - `test_cli_callback_configures`: `CliRunner` runs `ggt db --help` with `configure`
    patched.

  Each test restores the logger state.
- [ ] **Step 2:** Run `scripts/run_tests.sh tests/utils/test_logging_setup.py`. Expected: FAIL.
- [ ] **Step 3:** Implement the logging set-up.
- [ ] **Step 4:** Run `scripts/run_tests.sh tests/utils/ tests/paper/ tests/scripts/`.
  Expected: PASS, transcripts unchanged. Then check that the oracles are identical (label
  `t13`); the oracle scripts do not go through `cli/main.py`.
- [ ] **Step 5:** Commit: `feat(cli): configure ggTrader logging once so the trader's INFO lines reach the cron log`.
  The commit message notes that `scripts/paper_trade.sh` sends the last 8 log lines on
  failure, and those will now often be log lines.

### Task 14: Annotation style and the first locked-in lint rules (spec 2.5 and design section 5.4)

**Files:**
- Modify: `ruff.toml`. Append `UP006`, `UP035`, `UP045`, `B` and `S110` to the existing
  selection (`ruff.toml:15-22`). Add these per-file ignores for all five rules, merging into an
  existing entry where a file already has one (`ruff.toml:39-49`):
  - `research/**`, which is hash-bound;
  - `tests/**`, which has 35 hits. A test clean-up is a later task;
  - the scripts left out on purpose: `scripts/p5_wfo_validation.py`, `scripts/p7_live_golden.py`,
    `scripts/legacy_reference_runs.py`, `scripts/legacy_reference_compare.py`,
    `scripts/verify_pins.py`, `scripts/verdict_register.py`, `scripts/study.py` and
    `scripts/vbt_latest.py`, each with a comment saying why.
- Modify: the files the automatic fix touches under `src/`, and under `scripts/` except the
  excluded ones.
  - Hand fixes for the 5 bugbear hits in non-excluded files. These are `zip` without
    `strict` at `cli/cmd_db.py:94`, `data/bars.py:126` and
    `data/live/cached_yfinance_loader.py:241`, and unused loop variables at
    `paper/trader.py:1403` and `paper/overlay.py:116` (`label`). The two other hits are in
    `scripts/legacy_reference_compare.py:43` and `scripts/study.py:1021`, which are excluded.
  - `cached_yfinance_loader.py` is the one `MATCH` pin. This task changes
    `.github/workflows/ci.yml:147` to `MATCH=0,MOVED-MATCH=4,HISTORY-MATCH=18,FAIL=0` in the
    same commit as its edit, and notes it in the changelog.
  - A log line at each of the 3 try-except-pass sites: `trader.py:436`, `lab/persist.py:123`,
    `cached_yfinance_loader.py:280`. `lab/persist.py` and `cached_yfinance_loader.py` have
    no logger yet, so each gains `_log = logging.getLogger(__name__)` at module top
    (`trader.py` already has `_log`, `:65`).

- [ ] **Step 1: Mechanical fix.** Run:
  ```
  ruff check src --select UP006,UP035,UP045,F401,I --fix
  ruff format src
  ```
  Use the worktree's `.venv/bin/ruff`, with safe fixes only. Then run the same two commands on
  each non-excluded `scripts/` file. Read the diff for removed imports that are compatibility
  re-exports or have side effects, such as `trader.py:20,36-38` and `engine/signals.py:19`.
  Restore any such import with a `noqa: F401` and its reason. Commit this mechanical diff
  alone: `style: modern type annotations (ruff UP006/UP035/UP045 autofix)`.
- [ ] **Step 1b:** Run `ggt --help`, `ggt paper --help` and `ggt db truncate --help` through
  the installed CLI under `offline.sh`. Each must print its help and exit 0, which proves that
  typer still parses the rewritten annotations.
- [ ] **Step 2:** Run the test-outcome check and the oracle check (label `t14a`). Expected:
  both pass. If typer fails on a rewritten `cli/` annotation,
  revert that file's annotation changes and add a per-file ignore with the reason.
- [ ] **Step 3: Hand fixes.**
  - The three flagged `zip` calls get an explicit `strict=False`, which is today's behavior.
    Existing `strict=True` calls elsewhere stay as they are.
  - Both unused loop variables are renamed with a leading underscore.
  - At each of the three sites, the bare `except Exception:` becomes
    `except Exception as exc:` followed by `_log.debug("…: %s", type(exc).__name__)`.
- [ ] **Step 4:** Edit `ruff.toml`, then run `ruff check .` and `ruff format --check .`.
  Expected: both clean.
- [ ] **Step 5:** Run the test-outcome check, the oracle check (label `t14b`) and the
  verify-pins check with the new counts. Expected: all pass.
- [ ] **Step 6:** Commit: `style: enable bugbear and try-except-pass rules for src and scripts`.

**Phase 2 gate and hand-back:**
- The test-outcome check passes.
- The oracles are identical (label `phase2`).
- The verify-pins check passes.
- `ruff check .` and `ruff format --check .` are clean.
- Codex Sol and Claude Opus each review the whole branch in a fresh context. Sol also repeats
  the check of this plan against the spec.
- Report to the owner. Merge only on their say-so.

## Execution roles (`CLAUDE.md`, delegation preference of 2026-11-08)

The owner reported ample quota on 2026-11-17. Usage is reported after each phase so the owner
can stop early.

- **Orchestrator:** Claude Sonnet. It re-verifies each task itself, by reading the diff and
  rerunning the task's commands.
- **Builders:**
  - Claude Sonnet: Tasks 4 and 5, which need judgment about the trader's behavior.
  - Codex Terra or Luna, or Claude Sonnet: the other tasks.
  - Claude Haiku: the searches in Tasks 7 and 12, with every result checked.
- **Reviewers, never from the builder's own family:**
  - Codex Sol reviews Claude-built tasks; Claude Opus reviews Codex-built tasks.
  - Task 4's goldens get an Opus read.
  - The whole branch gets both Sol and Opus.
