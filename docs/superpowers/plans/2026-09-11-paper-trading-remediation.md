# Paper-Trading Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute `docs/next_steps.md`'s ACTIVE STEP (2026-09-10) in order:
persistent MNST split-state fix → catastrophe-stop enable → core revert
(blend → standalone SP500) → cash-sweep hysteresis → the queued research
re-runs.

**Architecture:** Five independent, sequentially-deployed changes to
`src/ggTrader/paper/`, each behind the project's existing
"container isn't updated until CI builds + `docker compose pull`" deploy
loop. Each task lands as its own commit/PR-sized unit so attribution stays
clean (standing project lesson — see `AGENTS.md`). Tasks 1-2 are a paired
code+data fix (persistent split state); Task 3 is a pure ops flag flip
gated on Task 2; Task 4 is a signal-source swap that reuses the existing
sleeve-iteration code path via a thin adapter; Task 5 adds a hysteresis
threshold to the cash-sweep sizing function; Task 6 is CLI-driven lab
research, not application code.

**Tech Stack:** Python 3.12, pytest + `unittest.mock`, SQLAlchemy/
TimescaleDB (`ggTrader.lab.persist.get_engine`), Alpaca paper broker,
`ggt.py lab` CLI (vectorbt-based WFO harness).

**Spec:** `docs/next_steps.md` (ACTIVE STEP, 2026-09-10) and
`docs/research/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`
§7 (operational roadmap) and §5 (ranked research candidates); corrected
baseline numbers: `docs/research/_rebaseline_corrected_tape_20260822.json`;
live-ops findings: `docs/research/artifacts-2026-09-09-perf-review/ggtrader_paper_review_20260909.md`.

## Global Constraints

- Ruff (`ruff check .` / `ruff format .`) is a hard gate — `line-length = 100`,
  rules `E`/`F`/`W`/`I` — enforced by the `.claude/hooks/ruff-fix.sh`
  PostToolUse hook. Run it before every commit in this plan.
- Absolute imports rooted at `src` (`from ggTrader.<package> import ...`).
- Module/function docstrings explain *why*, not *what* — match the existing
  style in `src/ggTrader/paper/*.py` (see `split_check.py`, `cash_sweep.py`
  for the house voice).
- Every DB call in `src/ggTrader/paper/` fails soft (`try/except` → log
  warning → safe default), never raises out of the run loop — match this in
  every new persist function's call site.
- `src/ggTrader/paper/` is **not** bind-mounted into the `ggtrader_live`
  container — a code change has zero live effect until: push to `main` →
  `.github/workflows/docker-build.yml` publishes `ghcr.io/garykuepper/ggtrader:latest`
  (~2.5 min) → `docker compose pull && docker compose up -d` in
  `ggtrader_live`'s compose directory. Every task below that touches
  `src/ggTrader/paper/` ends with this deploy step, not just a commit.
- Never touch `data/` or database config outside the explicit one-off
  migration script in Task 2.
- Test DB calls are always mocked (`@patch("...persist._get_engine")` /
  equivalent) — no test in this plan hits a real database. Match the
  pattern already used throughout `tests/paper/test_paper_persist.py`.

---

### Task 1: `paper_split_state` persistence layer

**Files:**
- Modify: `src/ggTrader/paper/persist.py` (schema `_SCHEMA` string, new
  functions)
- Test: `tests/paper/test_paper_persist.py`

**Interfaces:**
- Produces: `get_open_split_corrections() -> dict[str, float]`,
  `save_split_correction(symbol: str, ex_date: str, factor: float) -> None`,
  `clear_split_correction(symbol: str) -> None` — consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Add to `tests/paper/test_paper_persist.py` (follow the existing
`@patch("ggTrader.paper.persist._get_engine")` + `MagicMock()` pattern used
by the neighboring `TestPeakValue`-style classes in that file):

```python
class TestSplitState:
    @patch("ggTrader.paper.persist._get_engine")
    def test_creates_split_state_table(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        init_paper_schema()

        executed_sql = " ".join(str(call[0][0]) for call in mock_conn.execute.call_args_list)
        assert "paper_split_state" in executed_sql

    @patch("ggTrader.paper.persist._get_engine")
    def test_get_open_split_corrections_returns_persisted_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [("MNST", 2.0), ("XYZ", 0.5)]

        result = get_open_split_corrections()

        assert result == {"MNST": 2.0, "XYZ": 0.5}

    @patch("ggTrader.paper.persist._get_engine")
    def test_get_open_split_corrections_empty_when_no_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        assert get_open_split_corrections() == {}

    @patch("ggTrader.paper.persist._get_engine")
    def test_save_split_correction_upserts(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        save_split_correction("MNST", "2026-08-11", 2.0)

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "INSERT INTO paper_split_state" in sql_str
        assert "ON CONFLICT (symbol)" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params == {"symbol": "MNST", "ex_date": "2026-08-11", "factor": 2.0}
        mock_conn.commit.assert_called_once()

    @patch("ggTrader.paper.persist._get_engine")
    def test_clear_split_correction_deletes_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        clear_split_correction("MNST")

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "DELETE FROM paper_split_state" in sql_str
        assert mock_conn.execute.call_args[0][1] == {"symbol": "MNST"}
        mock_conn.commit.assert_called_once()
```

Add the three new names to that test file's existing
`from ggTrader.paper.persist import (...)` block:
`clear_split_correction, get_open_split_corrections, init_paper_schema,
save_split_correction` (keep the block alphabetically sorted, matching the
rest of the file).

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_paper_persist.py::TestSplitState -v`
Expected: FAIL — `ImportError: cannot import name 'get_open_split_corrections'`.

- [ ] **Step 3: Add the schema and functions**

In `src/ggTrader/paper/persist.py`, add to the `_SCHEMA` string (after the
existing `paper_dividend_accruals` table, before the closing `"""`):

```python
CREATE TABLE IF NOT EXISTS paper_split_state (
    symbol TEXT PRIMARY KEY,
    ex_date DATE NOT NULL,
    factor DOUBLE PRECISION NOT NULL,
    first_detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Then add these three functions (near `get_peak_value`/`save_peak_value`,
which they mirror):

```python
def get_open_split_corrections() -> dict[str, float]:
    """Return `{symbol: factor}` for every split this account has detected
    as broker-unapplied and not yet confirmed applied.

    Persists past `trader._SPLIT_LOOKBACK_DAYS` (the rolling window used to
    detect *new* splits from the broker's corporate-actions feed) so a
    long-held unapplied split keeps being corrected instead of the
    correction silently expiring while the position is still held -- the
    MNST incident, unapplied since 2026-08-11, correction expired
    2026-08-25 while still held. See `split_check.py`'s module docstring.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT symbol, factor FROM paper_split_state")).all()
    return {symbol: float(factor) for symbol, factor in rows}


def save_split_correction(symbol: str, ex_date: str, factor: float) -> None:
    """Upsert a detected-unapplied split for `symbol`. Idempotent -- calling
    again for the same symbol just refreshes `factor`/`updated_at`."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_split_state (symbol, ex_date, factor) "
                "VALUES (:symbol, :ex_date, :factor) "
                "ON CONFLICT (symbol) DO UPDATE SET "
                "factor = EXCLUDED.factor, updated_at = now()"
            ),
            {"symbol": symbol, "ex_date": ex_date, "factor": factor},
        )
        conn.commit()


def clear_split_correction(symbol: str) -> None:
    """Remove a symbol's persisted split-correction state. Call once the
    broker's own snapshot history confirms the split was actually applied
    (qty jumped by ~factor) or the position is fully closed -- see
    `trader._compute_split_corrections`."""
    with _get_engine().connect() as conn:
        conn.execute(
            text("DELETE FROM paper_split_state WHERE symbol = :symbol"), {"symbol": symbol}
        )
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_paper_persist.py -v`
Expected: PASS, all tests in the file (not just the new class).

- [ ] **Step 5: Lint and commit**

```bash
ruff check --fix src/ggTrader/paper/persist.py tests/paper/test_paper_persist.py
ruff format src/ggTrader/paper/persist.py tests/paper/test_paper_persist.py
git add src/ggTrader/paper/persist.py tests/paper/test_paper_persist.py
git commit -m "feat(paper): add persistent paper_split_state table

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire persistent split state into `PaperTrader`, restate NAV history

**Files:**
- Modify: `src/ggTrader/paper/trader.py:238-289` (`_compute_split_corrections`),
  its import block (`~line 14-31`)
- Test: `tests/paper/test_trader.py` (split-correction test class)
- Create (one-off, not shipped): `/tmp/restate_mnst_snapshots.py` (run once
  against the live DB, not committed to the repo)

**Interfaces:**
- Consumes: `get_open_split_corrections`, `save_split_correction`,
  `clear_split_correction` from Task 1.
- Produces: `_compute_split_corrections` now returns corrections for
  long-held unapplied splits too — no new public interface, same signature
  and return shape as before (`dict[str, float]`).

- [ ] **Step 1: Write the failing tests**

Find the existing split-correction test class in `tests/paper/test_trader.py`
(search for `_compute_split_corrections` or `class.*Split`) and add these
cases alongside it, following that class's existing fixture/mock style
(construct a `PaperTrader` with a mocked `_broker`, call
`trader._compute_split_corrections(positions, split_since, today)`
directly):

```python
def test_split_correction_persists_past_lookback_window(self):
    """A split whose ex_date is now outside the broker-feed lookback
    window (corp_splits comes back empty) must still be corrected if it
    was previously detected and persisted -- the MNST bug."""
    trader = self._make_trader()
    trader._broker.get_split_evidence.return_value = {
        "corp_splits": {},
        "activity_applied": set(),
    }
    positions = {"MNST": {"qty": 20.8041, "cost_basis": 1887.11, "market_value": 986.67}}

    with patch(
        "ggTrader.paper.trader.get_open_split_corrections",
        return_value={"MNST": 2.0},
    ):
        result = trader._compute_split_corrections(
            positions, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
        )

    assert result == {"MNST": 2.0}

def test_split_correction_clears_persisted_state_once_applied(self):
    """Once snapshot history confirms the broker applied the split (qty
    jumped ~factor), persisted state for that symbol is cleared."""
    trader = self._make_trader()
    trader._broker.get_split_evidence.return_value = {
        "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
        "activity_applied": set(),
    }
    positions = {"MNST": {"qty": 41.6082, "cost_basis": 1887.11, "market_value": 1973.5}}

    with (
        patch(
            "ggTrader.paper.trader.get_open_split_corrections",
            return_value={"MNST": 2.0},
        ),
        patch("ggTrader.paper.trader.get_snapshot_history", return_value=[]),
        patch("ggTrader.paper.trader.get_trade_history_dates", return_value=[]),
        patch(
            "ggTrader.paper.trader.find_split_applied_symbols",
            return_value=({"MNST"}, set()),
        ) as mock_find,
        patch("ggTrader.paper.trader.clear_split_correction") as mock_clear,
    ):
        result = trader._compute_split_corrections(
            positions, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
        )

    assert mock_find.called
    mock_clear.assert_called_once_with("MNST")
    assert result == {}

def test_split_correction_drops_persisted_state_for_closed_position(self):
    """A symbol no longer held has nothing left to correct; its persisted
    row is cleaned up."""
    trader = self._make_trader()
    trader._broker.get_split_evidence.return_value = {
        "corp_splits": {},
        "activity_applied": set(),
    }

    with (
        patch(
            "ggTrader.paper.trader.get_open_split_corrections",
            return_value={"MNST": 2.0},
        ),
        patch("ggTrader.paper.trader.clear_split_correction") as mock_clear,
    ):
        result = trader._compute_split_corrections(
            {}, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
        )

    mock_clear.assert_called_once_with("MNST")
    assert result == {}
```

(If `_make_trader()` doesn't already exist as a shared fixture/helper in
that test file, use whatever existing helper the neighboring split-check
tests already use to construct a `PaperTrader` with a mocked `_broker` —
match that pattern exactly rather than inventing a new one.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_trader.py -k split_correction -v`
Expected: FAIL — persisted corrections aren't consulted yet, so
`test_split_correction_persists_past_lookback_window` gets `{}` instead of
`{"MNST": 2.0}`.

- [ ] **Step 3: Rewrite `_compute_split_corrections`**

In `src/ggTrader/paper/trader.py`, update the import block to add the three
Task 1 functions to the existing `from ggTrader.paper.persist import (...)`
block (alphabetically): `clear_split_correction`, `get_open_split_corrections`,
`save_split_correction`.

Replace the body of `_compute_split_corrections` (currently lines ~238-289)
with:

```python
    def _compute_split_corrections(
        self, positions: dict[str, dict], split_since: date, today: date
    ) -> dict[str, float]:
        """Determine unapplied-split corrections for currently held
        `positions`, using `paper_snapshots`/`paper_trades` history as the
        primary evidence and the account's own SPLIT activities as a
        secondary, fallback signal -- see `split_check.py`'s module
        docstring for why the activities feed alone is no longer trusted
        (it is a no-op on this paper account, so a real applied split would
        otherwise be double-corrected).

        Merges in `paper_split_state` (see `persist.get_open_split_corrections`)
        so a split detected on a prior run keeps being corrected even once
        its ex_date falls outside `split_since` -- the rolling lookback
        window only bounds *new* detection from the broker's feed, not how
        long an already-known correction is honored (the MNST incident:
        `_SPLIT_LOOKBACK_DAYS` expired 2026-08-25 while still held, see
        `docs/next_steps.md`). Persisted state is cleared once snapshot
        evidence confirms the broker applied the split, or the position is
        no longer held.

        Returns `{symbol: correction_factor}`, same shape as
        `AlpacaBroker.get_split_corrections` (see
        `split_check.apply_corrections_to_positions`). Fails soft at every
        step -- any error degrades toward the broker's corporate-actions +
        activities view alone, never aborts the run.
        """
        held_symbols = set(positions)

        try:
            persisted = get_open_split_corrections()
        except Exception as exc:
            _log.warning("Could not load persisted split state (non-fatal): %s", exc)
            persisted = {}

        # A symbol no longer held has nothing left to correct -- drop its
        # persisted row so the table doesn't grow unboundedly.
        for symbol in list(persisted):
            if symbol not in held_symbols:
                try:
                    clear_split_correction(symbol)
                except Exception as exc:
                    _log.warning(
                        "Could not clear stale split state for %s (non-fatal): %s", symbol, exc
                    )
                persisted.pop(symbol)

        evidence = self._broker.get_split_evidence(list(positions), split_since)
        corp_splits = evidence.get("corp_splits", {})
        activity_applied = evidence.get("activity_applied", set())

        if not corp_splits:
            return persisted

        try:
            snapshot_history = get_snapshot_history()
        except Exception as exc:
            _log.warning("Could not load snapshot history for split check (non-fatal): %s", exc)
            snapshot_history = []

        trade_dates_by_symbol: dict[str, list] = {}
        for symbol in corp_splits:
            try:
                trade_dates_by_symbol[symbol] = get_trade_history_dates(symbol)
            except Exception as exc:
                _log.warning(
                    "Could not load trade history for %s split check (non-fatal): %s",
                    symbol,
                    exc,
                )
                trade_dates_by_symbol[symbol] = []

        snapshot_applied, unresolved = find_split_applied_symbols(
            corp_splits, snapshot_history, trade_dates_by_symbol, today
        )
        if unresolved:
            _log.warning(
                "No snapshot evidence to confirm split status for %s -- falling back to "
                "the activities feed / correcting by default",
                sorted(unresolved),
            )

        applied_symbols = snapshot_applied | set(activity_applied)
        for symbol in applied_symbols:
            if symbol in persisted:
                try:
                    clear_split_correction(symbol)
                except Exception as exc:
                    _log.warning(
                        "Could not clear confirmed-applied split state for %s (non-fatal): %s",
                        symbol,
                        exc,
                    )
                persisted.pop(symbol, None)

        new_corrections = compute_split_corrections(corp_splits, applied_symbols)
        for symbol, factor in new_corrections.items():
            ex_date = min(ex for ex, _ in corp_splits[symbol])
            try:
                save_split_correction(symbol, str(ex_date), factor)
            except Exception as exc:
                _log.warning("Could not persist split state for %s (non-fatal): %s", symbol, exc)

        return {**persisted, **new_corrections}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_trader.py -v`
Expected: PASS, full file (this touches a method other split/concentration
tests exercise indirectly — confirm nothing else regressed).

- [ ] **Step 5: Lint and commit**

```bash
ruff check --fix src/ggTrader/paper/trader.py tests/paper/test_trader.py
ruff format src/ggTrader/paper/trader.py tests/paper/test_trader.py
git add src/ggTrader/paper/trader.py tests/paper/test_trader.py
git commit -m "fix(paper): persist unapplied-split corrections past the lookback window

MNST's 2-for-1 split (ex 2026-08-11) went uncorrected from 2026-08-25
onward because _SPLIT_LOOKBACK_DAYS=14 expired while still held. Persist
detected-unapplied splits in paper_split_state and keep correcting them
until the broker applies the split or the position closes.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Deploy**

```bash
git push origin main   # or the working branch, per repo convention
# wait ~2.5 min for .github/workflows/docker-build.yml to publish
# ghcr.io/garykuepper/ggtrader:latest, then:
cd ~/ggTrader   # or wherever ggtrader_live's compose file lives
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
```

- [ ] **Step 7: One-off restatement of the 2026-08-26 → 2026-09-08 snapshot history**

Not shipped code — a manual data-fix script run once against the live DB,
per `docs/next_steps.md` R1's "one-off restatement" ask. Write and run:

```python
# /tmp/restate_mnst_snapshots.py
"""One-off: restate paper_snapshots.positions['MNST'] for 2026-08-26 through
2026-09-08 to reflect the corrected (2x) market_value/unrealized_pl, now
that persistent split-state correction (see trader.py) exists going
forward. Run once, by hand, against the live DB. Not part of the app."""

import json
from datetime import date

from sqlalchemy import text

from ggTrader.lab.persist import get_engine
from ggTrader.paper.split_check import corrected_market_value, corrected_unrealized_pl

FACTOR = 2.0  # MNST 2-for-1, ex 2026-08-11
START, END = date(2026, 8, 26), date(2026, 9, 8)

engine = get_engine()
with engine.connect() as conn:
    rows = conn.execute(
        text(
            "SELECT run_date, positions FROM paper_snapshots "
            "WHERE run_date BETWEEN :start AND :end ORDER BY run_date"
        ),
        {"start": START, "end": END},
    ).all()

    for run_date, positions in rows:
        pos = positions if isinstance(positions, dict) else json.loads(positions)
        mnst = pos.get("MNST")
        if not mnst:
            continue
        cost_basis = mnst.get("cost_basis", 0.0)
        mv = mnst.get("market_value", 0.0)
        # Idempotency guard: skip a row already corrected by a prior run of
        # this script (true post-split MV for 20.8041 shares is >$1800;
        # anything already above ~$1500 is not the halved broker figure).
        if mv > 1500:
            print(f"{run_date}: already looks corrected (mv={mv:.2f}), skipping")
            continue
        new_mv = corrected_market_value(mv, FACTOR)
        new_pl = corrected_unrealized_pl(mv, FACTOR, cost_basis)
        pos["MNST"] = {
            **mnst,
            "market_value": new_mv,
            "unrealized_pl": new_pl,
            "unrealized_plpc": (new_pl / cost_basis) if cost_basis else mnst.get("unrealized_plpc"),
        }
        conn.execute(
            text("UPDATE paper_snapshots SET positions = :positions WHERE run_date = :run_date"),
            {"positions": json.dumps(pos), "run_date": run_date},
        )
        print(f"{run_date}: mv {mv:.2f} -> {new_mv:.2f}, unrealized_pl -> {new_pl:.2f}")
    conn.commit()
```

Run: `docker compose exec ggtrader_live python /tmp/restate_mnst_snapshots.py`
(or natively with `DB_HOST=localhost` — see `AGENTS.md`'s Docker-research
note — whichever this session has DB credentials for).

**Verification:** re-run
`docker exec ggtrader_db psql -U <user> -d <db> -c "SELECT run_date, positions->'MNST' FROM paper_snapshots WHERE run_date BETWEEN '2026-08-26' AND '2026-09-08' ORDER BY run_date;"`
and confirm MNST's `unrealized_pl` is now ~+$86 (not the ~-$900 broker
figure) across that range, matching the 2026-08-20 withdrawn-bug-report
reconciliation in `docs/next_steps.md`.

---

### Task 3: Enable the catastrophe stop

**Files:** none (env-only; no code change — `catastrophe_stop.py` already
implements the feature-flagged logic, gated on Task 2 being deployed and
verified).

**Interfaces:** none — pure config flip.

- [ ] **Step 1: Verify Task 2's fix is live and correct**

```bash
docker exec ggtrader_db psql -U <user> -d <db> -c \
  "SELECT symbol, factor, updated_at FROM paper_split_state;"
```
Confirm MNST no longer appears (broker hasn't applied it, so it should
still be a live open correction — expected to appear with factor=2.0,
*not* absent). Then confirm the corrected book's worst position:
```bash
docker compose exec ggtrader_live python -c "
from ggTrader.paper.trader import PaperTrader
# or query paper_snapshots directly for the latest run_date's positions,
# applying split_check.apply_corrections_to_positions with paper_split_state
"
```
Expected: MNST's unrealized loss reads ~-4.8%, not -52%; the worst true
position in the book is PNR at roughly -8.1% (per the 2026-09-09 review) —
well above the -25% floor.

- [ ] **Step 2: Enable the flag**

In `ggtrader_live`'s `.env` (not committed — matches how `CASH_SWEEP_ENABLED`
was enabled 2026-08-22):
```bash
echo "CATASTROPHE_STOP_ENABLED=true" >> .env
docker compose up -d ggtrader_live   # recreate to pick up the new env var
```

- [ ] **Step 3: Verify armed, watch one week**

```bash
docker compose logs ggtrader_live --since 1h | grep -i catastrophe
```
Confirm no unexpected force-sell fires on the next cron run (12:45 PT).
Watch the Telegram notifier for a week for any `catastrophe_stop` reason
trade before considering this step closed.

- [ ] **Step 4: Record in changelog**

```bash
git add docs/changelog.md  # after adding a dated entry describing the flag flip
git commit -m "docs: record CATASTROPHE_STOP_ENABLED=true, gated on persistent split-state fix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Revert live trading from the 3-sleeve blend to the standalone SP500 core

**Files:**
- Modify: `src/ggTrader/paper/signal_runner.py` (new function)
- Modify: `src/ggTrader/paper/trader.py:33` (import), `:412` (call site)
- Test: `tests/paper/test_signal_runner.py` (new test class),
  `tests/paper/test_trader.py` (bulk rename of the patch target)

**Interfaces:**
- Consumes: `generate_signals(universe="sp500")` (existing,
  `signal_runner.py:20`) — returns `{"buys": [...], "sells": [...],
  "as_of": str, "universe_size": int, "gate": {...}}`.
- Produces: `generate_core_signals() -> dict` — returns the same shape as
  the existing `generate_blended_signals()` (`{"sleeves": {...}, "weights":
  {...}, "scale": float, "rebalanced_today": bool, "fallback_used": bool}`),
  so `trader.py`'s downstream sleeve-iteration/buy-sizing loop
  (`weights.get(universe, 0.0), scale` at `trader.py:637`) needs **zero**
  changes — it already correctly resolves to the full portfolio value when
  `weights == {"sp500": 1.0}` and `scale == 1.0`.

- [ ] **Step 1: Write the failing test**

Add to `tests/paper/test_signal_runner.py`, alongside the existing
`TestGenerateBlendedSignals` class:

```python
class TestGenerateCoreSignals:
    @patch("ggTrader.paper.signal_runner.generate_signals")
    def test_wraps_sp500_signals_in_blend_shape(self, mock_generate):
        mock_generate.return_value = {
            "buys": ["AAPL", "MSFT"],
            "sells": ["TSLA"],
            "as_of": "2026-09-10",
            "universe_size": 503,
            "gate": {"gate_enabled": False},
        }

        result = generate_core_signals()

        mock_generate.assert_called_once_with(universe="sp500")
        assert result["sleeves"] == {"sp500": mock_generate.return_value}
        assert result["weights"] == {"sp500": 1.0}
        assert result["scale"] == 1.0
        assert result["rebalanced_today"] is False
        assert result["fallback_used"] is False
```

Add `generate_core_signals` to that test file's existing
`from ggTrader.paper.signal_runner import (...)` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_signal_runner.py::TestGenerateCoreSignals -v`
Expected: FAIL — `ImportError: cannot import name 'generate_core_signals'`.

- [ ] **Step 3: Implement `generate_core_signals`**

In `src/ggTrader/paper/signal_runner.py`, add after `generate_blended_signals`:

```python
def generate_core_signals() -> dict:
    """Generate today's signals for the standalone SP500 core strategy,
    wrapped in the same shape `generate_blended_signals()` returns so
    `trader.py`'s sleeve-iteration/buy-sizing logic needs no changes.

    Deployed 2026-09 in place of the 3-sleeve blend: the corrected-tape
    pinned-window re-baseline
    (`docs/research/_rebaseline_corrected_tape_20260822.json`) shows the
    blend underperforming this standalone core (Sharpe 0.69 vs 0.99, third
    independent confirmation) -- see `docs/next_steps.md`. Kept alongside
    `generate_blended_signals` (not deleted) since the blend's WFO/research
    infrastructure (`ggt lab --blend`) is still valid tooling for any future
    diversification-sleeve candidate that actually clears the bar.
    """
    core_signals = generate_signals(universe="sp500")
    return {
        "sleeves": {"sp500": core_signals},
        "weights": {"sp500": 1.0},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_signal_runner.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Switch `trader.py` to call it, and re-point existing tests**

In `src/ggTrader/paper/trader.py`:
- Line 33: change
  `from ggTrader.paper.signal_runner import generate_blended_signals`
  to `from ggTrader.paper.signal_runner import generate_core_signals`.
- Line 412: change `blend = generate_blended_signals()` to
  `blend = generate_core_signals()`. (The local variable name `blend` stays
  — every downstream reference in `run()` already treats it as "the
  sleeve-shaped signal bundle for this run," which is still accurate.)

`tests/paper/test_trader.py` has ~20 `@patch("ggTrader.paper.trader.generate_blended_signals")`
decorators (and matching in-body references) that all need to track the
rename, since the patch target must match what `trader.py` actually
imports. Bulk-rename rather than hand-editing each:

```bash
sed -i 's/generate_blended_signals/generate_core_signals/g' tests/paper/test_trader.py
```

- [ ] **Step 6: Run the full paper test suite**

Run: `PYTHONPATH=src python -m pytest tests/paper/ -v`
Expected: PASS, all files — this confirms both the rename and that nothing
in `run()`'s downstream logic silently assumed multi-sleeve behavior.

- [ ] **Step 7: Lint and commit**

```bash
ruff check --fix src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py
ruff format src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py
git add src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py
git commit -m "feat(paper): revert live trading from 3-sleeve blend to standalone SP500 core

Corrected-tape re-baseline shows the blend underperforming the core
(Sharpe 0.69 vs 0.99, three independent confirmations since June).
generate_blended_signals() kept for future blend research; trader.py now
calls the new generate_core_signals() wrapper instead.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Deploy**

```bash
git push origin main
# wait for CI build, then:
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
```

- [ ] **Step 9: Verify on the next live run**

After the next 12:45 PT cron run, confirm via Telegram/logs that the
account is now trading only SP500-universe symbols (no MidCap400/Nasdaq100
buys), and `paper_rebalance_state` stops updating (the blend's monthly
overlay recompute is no longer called).

---

### Task 5: Cash-sweep buy-side hysteresis dead-band

**Files:**
- Modify: `src/ggTrader/paper/cash_sweep.py`
- Modify: `src/ggTrader/paper/trader.py` (~line 795-803, sweep-buy call site)
- Test: `tests/paper/test_cash_sweep.py`

**Interfaces:**
- Produces: `buy_trigger_pct() -> float` (env-var reader, mirrors
  `reserve_pct()`); `compute_sweep_buy(..., buy_trigger_pct: float =
  DEFAULT_BUY_TRIGGER_PCT)` — new optional parameter, backward compatible
  (existing callers/tests that don't pass it get the old-reserve-based
  trigger behavior only if they also override the new default; document
  this explicitly since it's a behavior change, not just a signature
  change — see Step 3).

**Scope note (read before implementing):** this task implements only the
**buy-side** dead-band (`docs/next_steps.md`'s "buy only when cash exceeds
8.0%"). The **sell-side** trigger ("sell only when cash drops below 2.0%")
described in the same next_steps.md paragraph is deliberately *not*
implemented here: `compute_sweep_sell_for_funding` only ever sells to cover
a genuine shortfall for the day's strategy buys (see its docstring) — it is
already need-driven, not periodic, and gating it further on a separate 2%
floor would mean strategy buys sometimes go unfunded rather than the sweep
covering them, which changes buy-sizing behavior well beyond this task's
blast radius. If sweep-driven round-trips are still frequent after this
change ships and is observed for a week or two, that sell-side gate is a
separate, follow-up task requiring its own design pass — flag it in
`docs/next_steps.md` at that point rather than guessing at it now.

- [ ] **Step 1: Write the failing tests**

Add to `tests/paper/test_cash_sweep.py`, alongside the existing
`compute_sweep_buy` test class:

```python
def test_buy_skipped_when_cash_above_reserve_but_below_trigger(self):
    """A small daily surplus just above the 5% reserve (but below the 8%
    trigger) must not fire a sweep buy -- this is the daily-churn bug."""
    action = compute_sweep_buy(
        cash_after_strategy_orders=6_500.0,  # 6.5% of 100k: > 5% reserve, < 8% trigger
        portfolio_value=100_000.0,
        reserve_pct=0.05,
        min_clip=500.0,
        buy_trigger_pct=0.08,
    )
    assert action.side is None

def test_buy_fires_once_cash_exceeds_trigger_and_targets_reserve(self):
    """Once cash clears the 8% trigger, the buy still sweeps down to the
    5% reserve floor, same target math as before."""
    action = compute_sweep_buy(
        cash_after_strategy_orders=9_000.0,  # 9% of 100k: > 8% trigger
        portfolio_value=100_000.0,
        reserve_pct=0.05,
        min_clip=500.0,
        buy_trigger_pct=0.08,
    )
    assert action.side == "buy"
    assert action.notional == 4_000.0  # sweeps 9,000 down to the 5,000 reserve

def test_buy_trigger_pct_env_var_default(self):
    assert buy_trigger_pct() == DEFAULT_BUY_TRIGGER_PCT

@patch.dict("os.environ", {"SWEEP_BUY_TRIGGER_PCT": "0.10"})
def test_buy_trigger_pct_env_var_override(self):
    assert buy_trigger_pct() == 0.10
```

Add `buy_trigger_pct`, `DEFAULT_BUY_TRIGGER_PCT` to that test file's
`from ggTrader.paper.cash_sweep import (...)` block.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_cash_sweep.py -k trigger -v`
Expected: FAIL — `ImportError: cannot import name 'buy_trigger_pct'`.

- [ ] **Step 3: Implement the dead-band**

In `src/ggTrader/paper/cash_sweep.py`, add near the other env-var readers:

```python
_BUY_TRIGGER_ENV_VAR = "SWEEP_BUY_TRIGGER_PCT"
DEFAULT_BUY_TRIGGER_PCT = 0.08


def buy_trigger_pct() -> float:
    """Fraction of portfolio value cash must exceed before a sweep buy
    fires -- a hysteresis dead-band above `reserve_pct()`'s 5% target, so a
    small daily surplus doesn't round-trip a buy every session (observed:
    15 sweep trades / $82.4K notional across 12 sessions,
    ~$5.5K/day churning against the reserve -- see
    docs/research/artifacts-2026-09-09-perf-review/). Default 8%."""
    raw = os.environ.get(_BUY_TRIGGER_ENV_VAR, "").strip()
    return float(raw) if raw else DEFAULT_BUY_TRIGGER_PCT
```

Then update `compute_sweep_buy`'s signature and body:

```python
def compute_sweep_buy(
    cash_after_strategy_orders: float,
    portfolio_value: float,
    reserve_pct: float = DEFAULT_RESERVE_PCT,
    min_clip: float = DEFAULT_MIN_CLIP_USD,
    buy_trigger_pct: float = DEFAULT_BUY_TRIGGER_PCT,
) -> SweepAction:
    """Size a sweep BUY from cash left over after the day's strategy orders.

    Hysteresis dead-band: only triggers once `cash_after_strategy_orders`
    exceeds `buy_trigger_pct` of portfolio value (default 8%), not merely
    the `reserve_pct` floor (default 5%) -- prevents a daily few-hundred-
    dollar surplus just above the reserve from round-tripping a sweep buy
    every session. Once triggered, still sweeps down to the `reserve_pct`
    floor, same target as before this change.
    """
    if portfolio_value <= 0:
        return SweepAction(None, 0.0)
    trigger = buy_trigger_pct * portfolio_value
    if cash_after_strategy_orders <= trigger:
        return SweepAction(None, 0.0)
    reserve = reserve_pct * portfolio_value
    sweep_target = max(0.0, cash_after_strategy_orders - reserve)
    if sweep_target < min_clip:
        return SweepAction(None, 0.0)
    return SweepAction("buy", round(sweep_target, 2))
```

**Note on the parameter's default:** giving `buy_trigger_pct` a default of
`0.08` (not `reserve_pct`'s value) means every existing caller/test that
doesn't pass it explicitly now gets the *new*, stricter trigger behavior
automatically -- this is the intended behavior change, not an oversight.
Any pre-existing test in `test_cash_sweep.py` that asserts a buy fires with
`cash_after_strategy_orders` between the old 5% reserve and the new 8%
trigger will now fail; fix those by bumping their `cash_after_strategy_orders`
value above 8% of the portfolio value in that test, not by reverting the
default.

- [ ] **Step 4: Update the call site**

In `src/ggTrader/paper/trader.py` (~line 795-803), add the new keyword arg:

```python
                buy_action = cash_sweep.compute_sweep_buy(
                    cash_after_strategy_orders,
                    portfolio_value,
                    reserve_pct=cash_sweep.reserve_pct(),
                    min_clip=cash_sweep.min_clip_usd(),
                    buy_trigger_pct=cash_sweep.buy_trigger_pct(),
                )
```

- [ ] **Step 5: Run tests to verify they pass, fix any pre-existing breaks**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_cash_sweep.py tests/paper/test_trader.py -v`
Expected: PASS after adjusting any pre-existing `compute_sweep_buy` test
whose `cash_after_strategy_orders` fixture now falls in the new dead-band
(per the note in Step 3) — bump those fixture values above the 8% trigger.

- [ ] **Step 6: Lint and commit**

```bash
ruff check --fix src/ggTrader/paper/cash_sweep.py src/ggTrader/paper/trader.py \
  tests/paper/test_cash_sweep.py
ruff format src/ggTrader/paper/cash_sweep.py src/ggTrader/paper/trader.py \
  tests/paper/test_cash_sweep.py
git add src/ggTrader/paper/cash_sweep.py src/ggTrader/paper/trader.py \
  tests/paper/test_cash_sweep.py
git commit -m "feat(paper): cash-sweep buy hysteresis dead-band (8% trigger vs 5% reserve)

15 sweep trades / \$82.4K notional across 12 sessions round-tripped
against the 5% reserve for zero expected return. A sweep buy now only
fires once cash clears 8% of portfolio value, still sweeping down to the
5% floor once triggered. Sell-side trigger deliberately not touched --
see cash_sweep.py's compute_sweep_sell_for_funding docstring and this
task's scope note in docs/superpowers/plans/.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Deploy**

```bash
git push origin main
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
```

---

### Task 6: Research re-run queue (no code — `ggt lab` CLI + dated reports)

**Files:**
- Create: `docs/research/<date>-ensemble-ic-kelly-rebaseline.md` (Task 6a)
- Create: `docs/research/<date>-cross-asset-trend-sleeve.md` (Task 6b)
- Create: `docs/research/<date>-xs-momentum-modern-wfo.md` (Task 6c)
- Modify: `docs/research/RESEARCH_SNAPSHOT.md` (re-run the
  `research-snapshot` skill after each closes)

**Interfaces:** none — these are WFO research runs, not application code.
Each follows `docs/research/TEMPLATE-research-report.md`'s structure per
`docs/research/prompts/local-implementation-prompt-TEMPLATE.md`'s
deliverable expectations (GO or NO-GO either way, no live-config changes
without a separate ask).

- [ ] **Step 6a: Re-baseline `ensemble_ic` and `ensemble_kelly` against the
  real 0.99 core**

```bash
python ggt.py lab --strategy ensemble_ic --eval-start 2021-01-31 --eval-end 2026-04-30
python ggt.py lab --strategy ensemble_kelly --eval-start 2021-01-31 --eval-end 2026-04-30
```
Compare both against **Sharpe 0.99** (not the retired 1.12 baseline —
`RESEARCH_SNAPSHOT.md` §1). Per the audit's constrained-`ensemble_ic`
proposal (`RESEARCH_SNAPSHOT.md` §6 Tier 1 #3), if the unconstrained run
still shows fold-instability (<8/17) or MaxDD worse than -10%, implement
and test a minimum-10%-per-voter weight floor before writing the final
verdict — don't report on the unconstrained run alone if it's borderline.
Write the dated report; commit it.

- [ ] **Step 6b: Cross-asset trend sleeve (`TLT`/`GLD`/`DBC`)**

Build the strategy (new file under `src/ggTrader/lab/strategies/`,
registered in `STRATEGY_REGISTRY`, following the `Strategy` protocol per
`AGENTS.md` §3) and test it via:
```bash
python ggt.py lab --blend "ensemble@sp500,trend@cross_asset" --eval-start 2021-01-31 --eval-end 2026-04-30
```
Success bar (`RESEARCH_SNAPSHOT.md` §6): blending into the SP500 core
improves Sharpe beyond 0.99 and reduces MaxDD below -7.0%. Full TDD cycle
(failing test for the signal logic, then implementation) per
`AGENTS.md`'s "Vectorization First" rule — this is a real strategy
addition, not a config change, and belongs in its own plan/commit sequence
if it grows beyond a single sitting. Write the dated report; commit it.

- [ ] **Step 6c: `xs_momentum`/`dual_momentum` on the current lab/WFO stack**

```bash
python ggt.py lab --strategy xs_momentum --eval-start 2021-01-31 --eval-end 2026-04-30
python ggt.py lab --strategy dual_momentum --eval-start 2021-01-31 --eval-end 2026-04-30
```
Success bar: Sharpe ≥ 0.99, MaxDD ≤ -10%, gate pass ≥ 12/17
(`RESEARCH_SNAPSHOT.md` §6 Tier 1 #1). Zero new code — the strategy and
data are already wired into the registry. Write the dated report; commit
it.

- [ ] **Step 6d: Regenerate the research snapshot**

After each of 6a-6c closes (GO or NO-GO), invoke the `research-snapshot`
skill to fold the new verdict into `RESEARCH_SNAPSHOT.md` and both prompt
templates, rather than hand-editing them — matches this project's standing
"never hand-patch, always regenerate" rule (see that file's own header).

---

## Self-Review

**Spec coverage** (against `docs/next_steps.md`'s ACTIVE STEP numbered list):
1. Sweep — already done, no task needed. ✅ (context only, Task list starts
   at the next undone step)
2. MNST persistent split-state fix + NAV restatement — Tasks 1-2. ✅
3. Catastrophe stop, gated on step 2 — Task 3. ✅
4. Core revert — Task 4. ✅
5. Cash-sweep hysteresis — Task 5 (buy-side only; sell-side explicitly
   scoped out with rationale, not silently dropped). ✅
6. Research queue (`ensemble_ic`/`ensemble_kelly`, cross-asset sleeve,
   `xs_momentum`/`dual_momentum`) — Task 6. ✅
Also-flagged ops-alerting gap (`daily_pnl_report.sh`) from `next_steps.md`
— intentionally left as a flagged-not-queued item in this plan too, since
`next_steps.md` itself didn't queue it as a numbered step; call it out to
the user as available follow-up work, not silently omitted.

**Placeholder scan:** no TBD/"add appropriate handling"/unshown code found —
every code step above has the actual diff or full new function body; every
CLI-only task (3, 6) has the exact commands, not "run the appropriate
tests."

**Type/interface consistency:** `generate_core_signals()`'s return shape
(`sleeves`/`weights`/`scale`/`rebalanced_today`/`fallback_used`) matches
exactly what `trader.py:423`'s `weights, scale = blend["weights"],
blend["scale"]` and the `for universe, sleeve_signals in
blend["sleeves"].items()` loop already expect — verified against the
current `generate_blended_signals()` return statement, not assumed.
`compute_sweep_buy`'s new `buy_trigger_pct` parameter is consumed with the
same name at both its `cash_sweep.py` definition and its `trader.py` call
site. `get_open_split_corrections`/`save_split_correction`/
`clear_split_correction` are used with matching names and argument order in
both Task 1 (definition) and Task 2 (`trader.py` call sites).
