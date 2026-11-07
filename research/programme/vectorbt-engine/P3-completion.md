# P3 completion note: read-only data layer and vectorised point-in-time universe

**Date:** 2026-11-06. **Phase:** P3 of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md) (section 9). **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed).
Terms: **PIT** = point-in-time. **OD** = open decision in the spec. **Tape** = the daily bars table.

## Verdict against the exit criteria

| Criterion | Result |
|---|---|
| PIT mask: 0 mismatches against `eligible_at` on the sampled dates | **0 symbol mismatches** on 456 sampled bars for each of 4 configs (1,824 bar-config checks), frozen tape, 605 symbols x 1,929 bars. Sample = bars at -1/0/+1 around every constituent-change date inside the tape plus 160 random bars (seed 7). |
| No research write to `ohlcv` | No write path exists: a test scans `bars.py` and `snapshot.py` for INSERT/CREATE/UPDATE/DELETE/DROP/ALTER/TRUNCATE; the DB source opens `default_transaction_read_only=on` and refuses if the server says otherwise; the only statements issued are SHOW and one SELECT. The one manual live read (below) ran on that path. |
| Tests green through `scripts/run_tests.sh` | 79 new tests (13 snapshot, 22 bars, 44 universe); `tests/data` plus `tests/lab/test_universe.py`: 112 passed in 154 s under the 8 GB cap. The full suite was not re-run (no existing file was edited). |
| Ruff clean | `ruff check` and `ruff format --check` clean on all new files. |

## Mismatch counts, timings (evidence: `p3-evidence/p3_pit_parity.json`, driver `scripts/p3_pit_parity.py`)

| Config | Dates | Mismatches | Mean eligible | Mask build (best of 5) | `eligible_at` loop |
|---|---:|---:|---:|---:|---:|
| min history 400 (lab default) | 456 | **0** | 392.5 | 0.030 s | 77.8 s (0.171 s/date) |
| min history 60 (live) | 456 | **0** | 481.4 | 0.031 s | 77.9 s |
| min history 60, `max_stocks=50` | 456 | **0** | 49.2 | 0.040 s | 80.0 s |
| min history 60, cap off (`None`) | 456 | **0** | 481.4 | 0.031 s | 78.6 s |

Mask memory 1.17 MB (bool, 1,929 x 605); tracemalloc peak during a build 21-23 MB (the int64 history cumsum); whole script peak RSS 468 MB. S6 measured 0.05 s and 1.2 MB; this is faster on a quieter box. Full tape through the old per-bar path would take about 0.171 s x 1,929 = 330 s. The in-suite real-tape test runs about 200 dates x 4 configs (150 random plus ~17 change neighbourhoods) and takes ~2.5 min; an earlier, larger version of that test (about 500 dates x 4) also passed with 0 mismatches. 15 synthetic parity cases (3 seeds x 5 configs, random late starts, early ends, gaps, churning membership) also match `eligible_at` exactly and are not vacuous (eligible counts vary 0 to 24).

## What was built

- `src/ggTrader/data/snapshot.py`: `load_snapshot_long` verifies file sha256, row count, date range, symbol count (and, with `deep=True`, the writer-independent content hash, confirmed equal to the P0 manifest value) before returning rows; raises `SnapshotIntegrityError` on any mismatch; `default_snapshot_path()` (last manifest entry, or `$GGT_SNAPSHOT`).
- `src/ggTrader/data/bars.py`: `load_bars(symbols | universe, start, end, source='snapshot'|'db')` returning a frozen `Bars` (per-field frames, `requested`, `missing_symbols`, `last_valid_bar`, `provenance`, `summary()`, `early_ending()`, `last_bar_coverage()`, `legacy_frame()`); `to_legacy_frame`; `validate_long` (unique, naive, midnight, weekday). Module docstring records why `vbt.Data` is not adopted. `collapse_daily_duplicates` is untouched.
- `src/ggTrader/lab/universe.py`: `membership_mask`, `static_membership_mask`, `pit_mask`, `check_constituent_age` (warn/raise/ignore, default 45 days), `removal_exits` (OD6), `terminal_exits` (OD7; returns exits, entries with the same-bar entry removed, and a metadata dict with `optimistic: true` and the dead-ticker list), `coverage_by_year`.
- Tests: `tests/data/test_snapshot.py`, `tests/data/test_bars.py`, `tests/lab/test_universe.py`. Docs: `docs/changelog.md` entry, `docs/architecture.md` (tree plus a data-layer section). Evidence: `p3-evidence/p3_pit_parity.{json,log}`, `p3-evidence/p3_db_smoke.json`.

## Findings worth knowing

1. **The tape's ragged end is really 51 dead names.** Final-bar coverage on the frozen tape is 554 of 605 (91.6%); `terminal_exits` flags exactly those 51 as dead (default threshold: one trailing NaN bar). The evidence run supplied no entry signals, so same-bar entry suppression is covered by unit tests only; the S6 toy RSI run found 2 such cases (NBL, EQR). 101 symbols leave the index inside the bars window (OD6 events).
2. **Constituent file is stale on this tape:** last snapshot 2026-06-02, 150 days before the last bar (limit 45). Every `pit_mask` call on real data therefore warns unless `on_stale='ignore'`. Refresh stays a separate P6 run (OD5). The age is the last snapshot row's date, so a quiet stretch with no index changes also ages it; it is a prompt, not proof of error.
3. **Member-day price coverage** (requested symbols with no bars count as uncovered): 96.3% (2019), 96.2%, 96.4%, 97.7%, 98.8%, 99.7%, 99.8% (2025), 99.7% (2026). 12 requested symbols have no bars at all.
4. **Live DB read works** (one manual run, evidence `p3_db_smoke.json`): 5 symbols over 2021-01-31 to 2026-04-29, 1.0 s, identical to the snapshot in all five fields. The connection string came from the production config in a subshell and was never printed or stored.
5. P0's reader `load_snapshot_wide` and `to_legacy_frame(load_bars(...))` are bit-for-bit identical (`assert_frame_equal(check_exact=True)`, 7 symbols including a deliberately absent one), once the end-day convention is aligned: P0's frame excludes the bar stamped on `end` (database session-zone quirk), `load_bars` is inclusive. A caller reproducing a legacy window ending on day D passes the previous weekday.

## Deviations, omissions, uncertainties

- `pit_mask` takes the `close` frame (index and symbols come from it, because the history rule needs the closes) instead of the task's `(index, symbols, membership_history, ...)` list; this follows spec 5.7. `membership_mask(index, symbols, history)` is the literal index/symbols form.
- **Sector cap (`max_sector_count`) is not implemented.** The sweep mask path never applied it (`EnsembleSignal.select` does); S6 verified it on a prototype only. Add it if P4 needs it on the mask path.
- Date bounds are inclusive on both ends, which differs from the legacy loader's effective end-exclusive rule (finding 5); validation of stored rows covers the whole snapshot file, but for the DB source only the requested window.
- `terminal_exits` default treats any series missing on the final bar as dead; drop a ragged last bar or raise `min_trailing_nan_bars` when the tape end is in progress (live-like data). Not exercised on real signals here (that is a P6 run, OD7).
- `content_sha256` is duplicated from `scripts/make_tape_snapshot.py` (a script cannot be imported from `src`); equality is guarded by the real-tape deep-hash test against the manifest value. `scripts/make_tape_snapshot.py` still uses the legacy `collapse_daily_duplicates`.
- The DB source is tested against a mocked psycopg2 plus the one manual live read, not an automated integration test.
- The real-tape parity test is slow (~2.5 min of the 154 s run) because `eligible_at` is O(members x history) per call; it skips with a clear reason when the git-ignored snapshot is absent.
- Not done by design: `vbt.Data` adoption, any change to `lab.data`, `sweep.py`, `wfo.py` (callers move to `pit_mask` in P5), `refresh_bars`, the PIT file refresh.
