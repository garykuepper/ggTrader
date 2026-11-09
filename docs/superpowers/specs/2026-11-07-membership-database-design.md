**Status:** draft for owner review; not approved. Written 2026-11-07, after the owner decided to keep S&P 500 membership history in the database with every refresh kept as a version, and to stop keeping updated data as files. Appendix A records the fresh-reader check and how each finding was settled.

# S&P 500 membership in the database, with every refresh kept

Status: design spec, 2026-11-07. Read-only design: nothing was run against the database. Base: `main` at 68f61e7d. Terms: **PIT** = point-in-time. **Vintage** = one load of the upstream list, kept for ever. **Coverage** = the first and last change date a vintage knows about. **Fixture** = a frozen, checksummed file in git used by tests and CI.

## 0. In plain words

- The list of which stocks were in the S&P 500 on each day moves out of the file `data/universe/sp500_constituents_history.csv.gz` into three small database tables.
- Each refresh is stored as a new **vintage**. Old vintages are never edited or deleted. A correction adds rows and marks the old ones "replaced by vintage N". So you can ask: "who was in the index on 2023-11-01, as we believed on 2026-06-09?" and "what did the 2026-11-07 refresh change?".
- Code reads membership through one door, `MembershipSource`. It has two versions: the database (research on the host, the live trader) and frozen fixture files (tests, GitHub CI, evidence replays). You must say which one. If the database is chosen and cannot be read, the program stops. It never quietly switches to a file.
- The live trader refuses to trade if the membership list is unreadable or has not been checked against upstream recently.
- Every result records which vintage it used (vintage number and a content checksum). To reproduce an old result, pin that vintage.
- A weekly job on Encom fetches upstream, loads a new vintage automatically when only the recent end changed, and sends the diff to Telegram. If upstream rewrote older history, the job stops and waits for you.

Flaws in the idea, said plainly:

1. **"No files" cannot be absolute.** GitHub CI and the laptop test runs have no database. They need frozen files. These files are not "updated files": each is a fixed vintage under its own name, never edited. A new vintage gets a new name. Only vintages that research evidence depends on are committed.
2. **The live trader already depends on the database** (its state tables and the price cache). This change adds a new freshness check: if the weekly job dies, trading halts after the limit. That is the right failure, but it means the job needs monitoring.
3. **Vintages give a true "what we knew then" only from the first vintage (2026-06-09) forward.** Everything from 1996 to 2026-06 in vintage 1 is upstream's later reconstruction, with tickers mapped to today's symbols. Example: vintage 1 puts BLDR, JBL and UBER into the index on 2023-10-18. The upstream fix moves this to 2023-12-18. So vintage 1 has two months of lookahead on those names. Storing vintages makes errors like this visible. It does not fix them.
4. **Survivorship in the price data is unchanged.** Membership removes the bias of picking from today's index. But 12 members in 2021–2026, and 62–69 in 2019–2026, have no bars in the frozen tape.
5. **Defaulting to "latest" brings back the `--eval-end` problem.** If "latest vintage" is the default, results drift silently. Cited runs must pin a vintage (section 3.6).
6. **Backups get weaker unless we act.** Today the file is in git and copied to GitHub, which is a free off-site copy. The database is excluded from borgmatic ("backed up separately"; I could not check how). Section 8 fixes this.
7. **The concurrent refresh branch conflicts with this design.** `refresh/sp500-membership` overwrites the file in place. That breaks the event-timing audit's pinned file hash `b30739a3…`. It also moves every oracle. Its output should become vintage 2, not an overwrite.

## 1. Schema

### 1.1 Shape chosen: (c) intervals as the base, plus a per-vintage date list and a snapshot reconstruction

| Shape | Rows per vintage | As-of-date query | Vintage diff | Reproduces the file exactly | Verdict |
|---|---|---|---|---|---|
| (a) (date, symbol) per snapshot | ~1.37M (2,720 dates x ~503) | latest date <= d, then rows | 1.37M-row compare | yes | too big to version per refresh |
| (b) intervals only | ~1.3k | one range predicate | small set difference | **no**: snapshot dates with no change, and the last snapshot date (the age check uses it) are lost | not exact |
| **(c) intervals + snapshot-date list per vintage** | ~1.3k intervals (shared across vintages when unchanged) + 2.7k dates | one range predicate | set difference of intervals | **yes**: the date list plus intervals rebuild every row; the raw bytes are also kept | **chosen** |

Intervals are half-open `[valid_from, valid_to)`. `valid_to` is the first snapshot date where the symbol is absent. `NULL` means "still a member at the end of coverage", which matches the file's carry-forward rule. A symbol that leaves and rejoins has two intervals. Membership on day d (same-day effective) is `valid_from <= d AND (valid_to IS NULL OR valid_to > d)`. This equals `sp500_members_asof`: the latest row dated on or before d.

Lineage is per row: `load_id` is the first vintage that stated the interval. `superseded_by_load_id` is the vintage that stopped stating it. An interval that is identical in a later vintage is not rewritten. Visibility at vintage v is `load_id <= v AND (superseded_by_load_id IS NULL OR superseded_by_load_id > v)`. This is exact because load ids only increase, and each load supersedes every visible row that is not in its own set.

Types: calendar dates are `DATE`. Instants (`retrieved_at`, `created_at`) are `TIMESTAMPTZ`, written as UTC-aware values or `now()`. **No `timestamp without time zone` column**: this avoids the naive-UTC trap (`cached_yfinance_loader.py:215-226`). TimescaleDB features: **none**. These are plain tables, and they are tiny.

Tickers: `symbol_raw` is stored exactly as upstream wrote it (`BRK.B`). `symbol_yf` is computed at load time by `normalize_yf_ticker` (`BRK-B`). The loader rejects a vintage where two raw symbols map to the same `symbol_yf` on any date. If the normalisation rule changes, that is a new vintage (`source='renormalize'`), never an UPDATE.

### 1.2 DDL (idempotent; `index_name` lets other indices share the tables later)

```sql
CREATE TABLE IF NOT EXISTS universe_loads (
    load_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name         TEXT        NOT NULL CHECK (index_name ~ '^[a-z0-9_]+$'),
    source             TEXT        NOT NULL,  -- 'fja05680/sp500' | 'fixture-import' | 'reinstate' | 'renormalize'
    source_url         TEXT        NOT NULL,
    upstream_commit    TEXT        CHECK (upstream_commit ~ '^[0-9a-f]{40}$'),  -- NULL = unknown (vintage 1)
    upstream_etag      TEXT,
    retrieved_at       TIMESTAMPTZ NOT NULL,
    raw_sha256         CHAR(64)    NOT NULL CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),     -- uncompressed bytes
    raw_gzip           BYTEA       NOT NULL,                                          -- the bytes, gzipped
    content_sha256     CHAR(64)    NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'), -- canonical form, 1.4
    coverage_start     DATE        NOT NULL,
    coverage_end       DATE        NOT NULL CHECK (coverage_end >= coverage_start),
    n_snapshot_dates   INT         NOT NULL CHECK (n_snapshot_dates > 0),
    n_intervals        INT         NOT NULL CHECK (n_intervals > 0),
    n_symbols          INT         NOT NULL CHECK (n_symbols > 0),
    min_members        INT         NOT NULL,
    max_members        INT         NOT NULL CHECK (max_members >= min_members),
    rows_added         INT         NOT NULL,
    rows_superseded    INT         NOT NULL,
    parent_load_id     BIGINT      REFERENCES universe_loads (load_id),  -- latest at load time
    reinstates_load_id BIGINT      REFERENCES universe_loads (load_id),
    forced             BOOLEAN     NOT NULL DEFAULT false,
    force_reason       TEXT,
    loader_git_sha     TEXT        NOT NULL,
    note               TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    loaded_by          TEXT        NOT NULL DEFAULT current_user,
    CHECK (NOT forced OR force_reason IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS universe_snapshot_dates (
    load_id       BIGINT NOT NULL REFERENCES universe_loads (load_id),
    snapshot_date DATE   NOT NULL,
    n_members     INT    NOT NULL CHECK (n_members > 0),
    PRIMARY KEY (load_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS universe_membership (
    interval_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name            TEXT   NOT NULL,
    symbol_raw            TEXT   NOT NULL CHECK (symbol_raw <> '' AND symbol_raw = btrim(symbol_raw)),
    symbol_yf             TEXT   NOT NULL CHECK (symbol_yf <> ''),
    valid_from            DATE   NOT NULL,
    valid_to              DATE,  -- exclusive; NULL = member at coverage_end
    load_id               BIGINT NOT NULL REFERENCES universe_loads (load_id),
    superseded_by_load_id BIGINT REFERENCES universe_loads (load_id),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    CHECK (superseded_by_load_id IS NULL OR superseded_by_load_id > load_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS universe_membership_live_uq
    ON universe_membership (index_name, symbol_raw, valid_from) WHERE superseded_by_load_id IS NULL;
CREATE INDEX IF NOT EXISTS universe_membership_range_ix
    ON universe_membership (index_name, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS universe_membership_lineage_ix
    ON universe_membership (index_name, load_id, superseded_by_load_id);

-- Upstream checks that found no change: they reset the age clock without storing a new vintage.
CREATE TABLE IF NOT EXISTS universe_checks (
    check_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name      TEXT        NOT NULL,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_sha256      CHAR(64),
    upstream_commit TEXT,
    outcome         TEXT        NOT NULL CHECK (outcome IN ('unchanged','loaded','refused','fetch_failed')),
    matched_load_id BIGINT      REFERENCES universe_loads (load_id),
    detail          JSONB
);

-- Append-only enforcement, whatever role connects. The only allowed UPDATE sets superseded_by once.
CREATE OR REPLACE FUNCTION universe_membership_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'universe_membership is append-only'; END IF;
    IF OLD.superseded_by_load_id IS NOT NULL
       OR (NEW.interval_id, NEW.index_name, NEW.symbol_raw, NEW.symbol_yf, NEW.valid_from, NEW.valid_to,
           NEW.load_id)
          IS DISTINCT FROM
          (OLD.interval_id, OLD.index_name, OLD.symbol_raw, OLD.symbol_yf, OLD.valid_from, OLD.valid_to,
           OLD.load_id)
    THEN RAISE EXCEPTION 'only superseded_by_load_id may change, once'; END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE FUNCTION universe_frozen_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION '% is append-only (% refused)', TG_TABLE_NAME, TG_OP; END $$;

DROP TRIGGER IF EXISTS universe_membership_guard_t ON universe_membership;
CREATE TRIGGER universe_membership_guard_t BEFORE UPDATE OR DELETE ON universe_membership
    FOR EACH ROW EXECUTE FUNCTION universe_membership_guard();
-- Repeat the two triggers below for universe_loads, universe_snapshot_dates, universe_checks.
DROP TRIGGER IF EXISTS universe_loads_frozen_t ON universe_loads;
CREATE TRIGGER universe_loads_frozen_t BEFORE UPDATE OR DELETE ON universe_loads
    FOR EACH ROW EXECUTE FUNCTION universe_frozen_guard();
DROP TRIGGER IF EXISTS universe_loads_truncate_t ON universe_loads;
CREATE TRIGGER universe_loads_truncate_t BEFORE TRUNCATE ON universe_loads
    FOR EACH STATEMENT EXECUTE FUNCTION universe_frozen_guard();

-- Snapshot reconstruction of any vintage (the "view" of shape (c)).
CREATE OR REPLACE FUNCTION universe_snapshot(p_index TEXT, p_load BIGINT)
RETURNS TABLE (snapshot_date DATE, symbol_raw TEXT, symbol_yf TEXT) LANGUAGE sql STABLE AS $$
    SELECT s.snapshot_date, m.symbol_raw, m.symbol_yf
    FROM universe_snapshot_dates s
    JOIN universe_membership m
      ON m.index_name = p_index
     AND m.load_id <= p_load
     AND (m.superseded_by_load_id IS NULL OR m.superseded_by_load_id > p_load)
     AND m.valid_from <= s.snapshot_date
     AND (m.valid_to IS NULL OR m.valid_to > s.snapshot_date)
    WHERE s.load_id = p_load
$$;
```

A vintage's `index_name` must equal the `index_name` of its rows. The loader asserts this inside its transaction. `load_id` is shared across indices, which is harmless because every query filters on `index_name`. No exclusion constraint on overlapping intervals: that would need `btree_gist`. The loader and the integration test check it.

### 1.3 The four queries (`:d` DATE, `:v`/`:v1`/`:v2` load ids)

```sql
-- Q1 members on :d at the latest vintage (reader first checks :d >= latest coverage_start, else raises)
SELECT symbol_raw, symbol_yf FROM universe_membership
WHERE index_name = 'sp500' AND superseded_by_load_id IS NULL
  AND valid_from <= :d AND (valid_to IS NULL OR valid_to > :d)
ORDER BY symbol_raw;

-- Q2 members on :d as known at vintage :v
SELECT symbol_raw, symbol_yf FROM universe_membership
WHERE index_name = 'sp500' AND load_id <= :v
  AND (superseded_by_load_id IS NULL OR superseded_by_load_id > :v)
  AND valid_from <= :d AND (valid_to IS NULL OR valid_to > :d)
ORDER BY symbol_raw;

-- Q3a intervals that changed between :v1 and :v2 (EXCEPT treats NULL valid_to as equal)
WITH a AS (SELECT symbol_raw, valid_from, valid_to FROM universe_membership
           WHERE index_name = 'sp500' AND load_id <= :v1
             AND (superseded_by_load_id IS NULL OR superseded_by_load_id > :v1)),
     b AS (SELECT symbol_raw, valid_from, valid_to FROM universe_membership
           WHERE index_name = 'sp500' AND load_id <= :v2
             AND (superseded_by_load_id IS NULL OR superseded_by_load_id > :v2))
SELECT 'only_v1' AS side, * FROM (SELECT * FROM a EXCEPT SELECT * FROM b) x
UNION ALL
SELECT 'only_v2', * FROM (SELECT * FROM b EXCEPT SELECT * FROM a) y
ORDER BY symbol_raw, valid_from, side;

-- Q3b snapshot dates present in only one of the two vintages
SELECT snapshot_date, bool_or(load_id = :v1) AS in_v1, bool_or(load_id = :v2) AS in_v2
FROM universe_snapshot_dates WHERE load_id IN (:v1, :v2)
GROUP BY snapshot_date HAVING count(*) = 1 ORDER BY snapshot_date;

-- Q4 age of the latest vintage: since its last change date, and since upstream was last verified
SELECT l.load_id, l.coverage_end, l.retrieved_at, l.content_sha256,
       (now() AT TIME ZONE 'UTC')::date - l.coverage_end AS days_since_last_change,
       GREATEST(l.retrieved_at, COALESCE(c.last_ok, l.retrieved_at)) AS verified_at,
       now() - GREATEST(l.retrieved_at, COALESCE(c.last_ok, l.retrieved_at)) AS since_verified
FROM universe_loads l
LEFT JOIN LATERAL (SELECT max(checked_at) AS last_ok FROM universe_checks c
                   WHERE c.index_name = l.index_name AND c.matched_load_id = l.load_id
                     AND c.outcome = 'unchanged') c ON true
WHERE l.index_name = 'sp500'
ORDER BY l.load_id DESC LIMIT 1;
```

### 1.4 Canonical content hash

`content_sha256` is the sha256 of UTF-8 lines `YYYY-MM-DD,<raw tickers sorted, joined by ','>\n`, one line per snapshot date in ascending order, with no header. The loader computes it from the parsed input. The DB reader computes it from what it read back. The fixture reader computes it from the file. **This value, not `load_id`, is the vintage's identity in provenance.** Load ids differ between the production database and a disposable test schema.

## 2. Write path: `ggt db load-sp500`

- Inputs: `--from-upstream` (default) resolves the latest upstream commit for the CSV path through the GitHub commits API. It then fetches `raw.githubusercontent.com/fja05680/sp500/<commit>/…`, so the bytes and the commit always match. The alternative is `--from-file PATH --retrieved-at … [--upstream-commit …] --source fixture-import`, used to import the two existing vintages. Optional `--expect-sha256` (of the uncompressed bytes) must match.
- **Dry run by default.** `--commit` writes. Every run prints a diff report against the parent (latest) vintage, and `--report-json PATH` saves it. The report covers: coverage change; dates only in the new or only in the parent (Q3b); per-date set differences; interval rewrites split into **historical** (touching dates <= parent `coverage_end`), **tail** (after it) and **renames** (raw symbol A's intervals replaced by B's with identical dates); member-count range; raw and content hashes; upstream commit and ETag.
- Validation always runs. The loader refuses with exit 2 if any check fails: header `date,tickers`; ISO dates that strictly increase; no empty or duplicate ticker in a row; 480 <= members <= 520 on every date (today's range is 487–507); `symbol_yf` is injective per date; the reconstruction round-trip is identical (snapshots → intervals → snapshots).
- **Suspicious loads are refused unless `--force --reason "…"`**: coverage_start moves later or coverage_end moves earlier (history shrank); more than **5** historical interval rewrites, renames excluded; more than **25** tail interval changes; more than **3** renames; the parent changed between the dry run and the commit (`--parent N` pinned from the dry run). Vintage 1 → vintage 2 has **10** historical rewrites (ALK, SEDG, SEE, BLDR, JBL, UBER, VFC, XRAY, GEV, SOLV), so it needs `--force`. That is intended.
- **No-op:** if `raw_sha256` equals the latest load's hash, no vintage is stored. With `--commit` the loader writes a `universe_checks` row (`unchanged`), which resets the verification age.
- **One transaction:** `pg_advisory_xact_lock(hashtext('universe_load:sp500'))`. Re-read the latest load and assert it equals `--parent`. Insert the load row, including `raw_gzip`. Insert all snapshot dates. Mark the visible rows that are missing from the new set as superseded. Insert the new rows. Read the vintage back with Q2 logic, recompute `content_sha256`, and roll back on any mismatch. Insert the `universe_checks` row (`loaded`). Commit.
- Companion subcommands under `ggt db membership`: `status` (Q4), `diff --from V1 --to V2` (Q3a/Q3b plus per-date sets), `export --load-id N --out F` (writes `raw_gzip` back out, byte-exact, sha-verified), `reinstate --load-id N --reason … --commit`. Reinstate is the rollback: it creates a **new** vintage with the old content and `reinstates_load_id = N`. It never deletes.
- `download_sp500_history()` (the write-to-file function) is removed from `src/`. Nothing in `src/` writes a membership file again.
- **Credentials (described, not created).** Today one role from `.env` (`DB_*` / `POSTGRES_CONNECTION_STRING`) does everything. Proposed: a `ggt_universe_writer` role with SELECT and INSERT on the four tables, `UPDATE (superseded_by_load_id)` on `universe_membership`, and USAGE on their identity sequences. It gets no DELETE or TRUNCATE and no other tables. Its URL goes in `GGT_UNIVERSE_WRITER_URL`, used only by the loader. Readers keep the existing role but open a server-enforced read-only session (`default_transaction_read_only=on`, the `data/bars.py` pattern). Until the owner creates the role, the loader uses the existing role and the triggers carry the append-only guarantee. Research code and tests never call the loader.

## 3. Read path

### 3.1 Interface (new module `src/ggTrader/data/membership.py`)

```python
@dataclass(frozen=True)
class MembershipVintage:
    index_name: str; kind: str            # 'db' | 'fixture'
    load_id: int | None; fixture: str | None
    content_sha256: str; raw_sha256: str
    coverage_start: date; coverage_end: date
    retrieved_at: datetime; verified_at: datetime

class MembershipSource(Protocol):
    def vintage(self) -> MembershipVintage: ...
    def history(self) -> pd.DataFrame: ...   # exactly load_sp500_history()'s shape: UTC DatetimeIndex 'date',
                                             # column 'tickers' = sorted raw tickers per snapshot date
```

- `DbMembershipSource(vintage="latest" | int)` resolves "latest" to a concrete `load_id` **once, at construction**, so one run cannot straddle a load. It reads the intervals (~1.3k rows) and the snapshot dates in a read-only session and rebuilds the history in Python. It then checks the rebuilt `content_sha256` against `universe_loads` and raises `MembershipIntegrityError` on a mismatch.
- `FixtureMembershipSource(name)` reads a file listed in `data/universe/membership/MANIFEST.json`. It checks the file sha256 and the content hash before returning.
- `membership_source(spec)` takes `db`, `db:<load_id>` or `fixture:<name>`. It reads `spec`, else the environment variable `GGT_MEMBERSHIP`. If neither is set, it raises `MembershipSourceNotSelected` with instructions. **There is no fallback in any direction.** If the database is unreachable under `db`, it raises. It never reads a fixture instead.
- Who selects what: `ggt lab` and `ggt paper` have `--membership` with CLI default `db`, so the default is explicit at the entry point. Scripts take `--membership`, required for evidence drivers. `tests/conftest.py` sets `GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02` (autouse). The CI environment sets the same.

### 3.2 Pure functions keep their shape

`sp500_members_asof`, `all_members_between`, `membership_mask`, `pit_mask` and `eligible_at` keep their logic. Their implicit `load_sp500_history()` default is replaced by an explicit `history` / `membership` argument. If that argument is `None`, they call `membership_source(None)`, which raises unless the source was selected. The module-global `_history_cache` (keyed by file path) is replaced by a per-process cache keyed by `content_sha256`. Equal inputs give equal masks, so P3's 0-mismatch parity carries over unchanged.

### 3.3 Readers to migrate (every module that reads the membership data)

| Module | Today | Change |
|---|---|---|
| `data/core/index_constituents.py` | `load_sp500_history(csv_path)`, `SP500_HISTORY_CSV`, `download_sp500_history` | `load_sp500_history()` becomes a thin wrapper: `membership_source(None).history()`. Remove `download_…`. `universe_members_asof` / `universe_all_between` gain `membership=`. |
| `lab/universe.py` | `load_sp500_history()` default in `check_constituent_age`, `membership_mask`, `pit_mask` | Take `membership: MembershipSource`. The age check uses `vintage()` (3.4). |
| `lab/data.py` | `eligible_at`, `equity_universe_between` call `universe_*` | Add a `membership=` parameter and pass it through. |
| `lab/wfo/inputs.py` | `universe_mask` → `pit_mask(on_stale="ignore")` | Add `membership=` and pass it through. |
| `lab/wfo/cli_support.py` | `equity_universe_between`, `load_bars`, `universe_mask` | Build one source from `--membership` and pass it to all three. Record it in provenance (3.5). |
| `data/bars.py` | `universe_all_between` when `universe=` is given | `load_bars(..., membership=)`. Required when `universe='sp500'`. |
| `paper/signal_runner.py` | `universe_members_asof(universe, today)` | `generate_signals(..., membership=)`. Live gate (3.4). |
| `paper/overlay.py` | `universe_members_asof` (blend path, not live) | `membership=` passed through. |
| `scripts/p5_wfo_validation.py`, `p7_live_golden.py`, `p3_pit_parity.py`, `make_tape_snapshot.py`, `legacy_reference_runs.py`, `equity_backfill.py`, `form4_backfill.py`, `google_trends_backfill.py`, `earnings_surprise_backfill.py` | implicit file | `--membership` argument. Oracle drivers default to `fixture:sp500_v1_2026-06-02` until the re-baseline (slice S5). |
| `research/audits/event-timing/code/event_timing_audit.py`, `research/audits/production-replica/code/run.py`, `research/audits/production-replica/code/replica_parity_check.py` | file path, or `universe_members_asof` | **Do not edit.** These are pinned study code, and PLAN_V2 moves them. The audit keeps reading the frozen v1 file at its old path, which has the same bytes (`b30739a3…`). The two replica scripts run with `GGT_MEMBERSHIP=fixture:sp500_v1_2026-06-02`. |
| `research/audits/event-timing/code/event_timing_sources.py`, `scripts/fetch_midcap400_snapshot.py`, `lab/wfo/run.py`, `lab/strategies/xs_momentum.py` | normaliser import or docstring only | no change |

**Revision 2026-11-08 (after the study-isolation moves, checked by `git grep`).** The study code named in the table above has already moved out of `src/` and `scripts/` (paths updated in place). The code that reads the membership list today is: in `src/` only `data/bars.py`, `data/core/index_constituents.py`, `lab/data.py`, `lab/universe.py`, `paper/overlay.py` and `paper/signal_runner.py`; in `scripts/` `equity_backfill.py`, `legacy_reference_runs.py` and `p3_pit_parity.py` (the oracle drivers `p5_wfo_validation.py` and `p7_live_golden.py` reach it through `lab/`); and the two pinned studies, which keep reading the frozen vintage-1 file at its old path and are **not** edited (`research/audits/event-timing/code/event_timing_audit.py`, `research/audits/production-replica/code/run.py` and `replica_parity_check.py`). About ten test files read it. That is a smaller migration than the table suggested: S3's estimate drops from 6 to about 4 agent-hours, so the six slices total about 27 rather than 31 (still an estimate). Because `src/` no longer holds any study code, nothing in a study needs the database to run.

### 3.4 Age check and the live gate

`check_constituent_age(asof, vintage, max_age_days, on_stale)` reports `age_since_change = asof - coverage_end` (for information; a quiet spell is legitimate) and `age_since_verified = asof - verified_at`. **Stale means `age_since_verified > max_age_days`.** A vintage retrieved after a backtest's last bar is never stale for that backtest. Research keeps `warn`, and WFO keeps `ignore`. Live: `generate_signals` raises `SignalGateError("membership_unavailable")` when the read fails, and `SignalGateError("membership_stale")` when `age_since_verified > LIVE_MAX_MEMBERSHIP_AGE_DAYS` (default **10** with the weekly job, **45** before it). The existing fail-closed path turns either into "no orders" plus a Telegram message. Today's file would fail this gate: its last row is 2026-06-02 and it was committed 2026-06-09, about 150 days ago. **The trader has been trading on a stale universe:** AVB and EQR were removed on 2026-08-18, and MRVL and FLEX were added on 2026-06-22.

### 3.5 Provenance and per-result JSON

Add a top-level `membership_sha256` (content hash) to the provenance block. Add a `membership` detail block holding `{kind, load_id, fixture, raw_sha256, coverage_start, coverage_end, retrieved_at, verified_at, age_since_verified_days}`. Add `membership` to `_IGNORED_KEYS`, and add a flag `MEMBERSHIP_DIFF` (not comparable, same rank as `TAPE_DIFF`) for a `membership_sha256` difference. When the field is absent on either side, the result is unknown: no flag, but a note. Every driver JSON and `ggt lab --wfo` record gets the same `membership` block next to `tape_sha256`.

### 3.6 Reproducing an old result

Re-run with `--membership db:<load_id>` or `fixture:<name>`, plus `--expect-membership-sha <content_sha256>`. The run refuses if the hash differs. Every result from 2026-06-09 up to the cutover used the one committed file (`b30739a3…`, git shows a single commit, `f8c7efe0`). Those results equal vintage 1. **Rule:** any run whose number is cited pins `--membership`. Evidence drivers make it a required argument. The "latest" default exists only for ad-hoc `ggt lab` runs, and they print the load id they used.

## 4. Fixtures and evidence

These files stay in git. They are immutable: one name = one vintage, never edited. A new vintage gets a new file, and only if something pins it.

| Fixture | Why it must stay |
|---|---|
| `data/universe/sp500_constituents_history.csv.gz` (**vintage 1**, gz sha `b30739a3…`, CSV sha `bc9c2aae…`, coverage 1996-01-02 → 2026-06-02) | The path and bytes are pinned by the event-timing audit (code and `_event_evidence_resolution_workbook.json`). It feeds the guarded P5 oracles and the P7 live golden. **The bytes stay frozen at this path** (manifest alias `sp500_v1_2026-06-02`) until PLAN_V2 slice S7 moves the audit. Only then is a rename considered, as one coordinated slice. |
| `data/universe/membership/sp500_v2_2026-11-07_<content8>.csv.gz` (**vintage 2**) | Committed only when the re-baselined oracles and golden pin it (slice S5). It is produced by `ggt db membership export`, so it is byte-identical to the DB copy. |
| Synthetic histories built in test code (`tests/lab/test_universe.py`) | Unchanged. They stay in code, not in files. |

Registry: `data/universe/membership/MANIFEST.json` lists `name → {path, gz_sha256, raw_sha256, content_sha256, coverage, upstream_commit, retrieved_at, db_load_id_on_encom}`. A fast test, in the style of `test_lock_consistency.py`, checks that every entry exists, that its hashes match, that the v1 alias still hashes to `b30739a3…`, and that no unlisted file sits in the folder.

Loading the two vintages:
1. `ggt db load-sp500 --from-file data/universe/sp500_constituents_history.csv.gz --source fixture-import --retrieved-at 2026-06-09T00:00:00Z --note "committed f8c7efe0; true retrieval time unknown" --commit` creates vintage 1. There is no parent, so no refusal rules apply.
2. Vintage 2 comes from the refresh agent's download (`pit-refresh/dl/upstream.csv`, sha `36326709…`, retrieved 2026-11-07T18:26:50Z, ETag recorded; the upstream commit should be resolved before loading). Run `--from-file … --expect-sha256 36326709… --force --reason "upstream re-dated the Dec-2023 and Apr-2024 rebalances; 9 new dates to 2026-08-18" --commit`. After that, `ggt db membership diff --from 1 --to 2` shows the same changes as the refresh agent's `diff_report.json`: 9 dates only in v2, 1 date only in v1, and 1 shared date that differs.

## 5. Scheduling

| Option | Verdict |
|---|---|
| (a) GitHub Actions opens a PR | No longer fits: the truth is not in git, and CI has no database. The draft `sp500-membership-check.yml` should not be merged. |
| **(b) Weekly host job** | **Recommended.** |
| (c) Manual | Fine until the preconditions below hold. It is also the reason the file reached 150 days stale. |

Recommended job: a systemd user timer (deterministic, no LLM; preferred over a Hermes agent cron job). It runs Saturday 06:00 PT, as `docker compose run --rm ggtrader_live ggt db load-sp500 --auto`, so it uses the same code as the live reader. `--auto` means: fetch, dry-run, and commit **only** if the diff is tail-only (no historical rewrite, renames within limit, coverage not shrinking). A no-change run records an `unchanged` check. Anything else is refused. Every outcome (loaded / unchanged / refused / fetch failed) goes through the existing `TelegramNotifier`, with the diff summary. A refusal waits for the owner to run it with `--force`.

Why an automatic load is safe here: it never destroys anything, because the previous vintage stays queryable. Undo is one command (`reinstate`). Every result records its content hash, so later loads cannot change what a recorded result meant. The remaining risk is a wrong recent change (an upstream typo) reaching the live universe for up to a week. That is bounded by the validation gates and the tail-only rule, and it is smaller than today's 5-month staleness.

Before enabling: S1–S4 merged and deployed; vintages 1 and 2 loaded; the equivalence checks green; three manual weekly dry runs that match expectations; the Telegram path tested through `DryRunNotifier`; a restore drill of the new tables (section 8); and the live age limit set (owner decision D1). The trader is stopped, so the job can run before trading resumes.

## 6. Other mutable data (tracked under `data/` unless noted)

| Data | Readers today | Class | Notes |
|---|---|---|---|
| `universe/sp500_constituents_history.csv.gz` | 11 src modules + 9 scripts | **MOVE TO DB NOW** | This spec. The file stays as the frozen v1 fixture. |
| `universe/sp500_sectors.json` (yfinance sectors, 2026-07-06) | none in `src/` (sector cap removed in P5c); research scripts only | **MOVE LATER** | A `security_attributes(load_id, symbol_yf, attribute, value, valid_from, valid_to)` table on the same loads/lineage pattern, when a sector cap returns. Until then it is an immutable fixture for the S6 evidence scripts. |
| `universe/fomc_meeting_dates.csv` | legacy script only; evidence for the fomc-drift NO-GO | **MOVE LATER** | An `event_calendar(load_id, calendar, event_date, kind)` table with vintages, when an event study needs it. The file stays frozen as evidence. |
| `universe/nasdaq100_…_2026-06-09.txt`, `russell2000_…_2026-06-09.txt`, `midcap400_…_2026-06-24.txt` | `snapshot_members` (lab, blend path, event-timing audit) | **MOVE LATER** | Already dated and immutable. If any is ever refreshed, load it as a `universe_loads` vintage with one snapshot date (same tables, `index_name='nasdaq100'`). Never overwrite it. |
| `universe/{fx_hedge,fomc_treasury,commodity_trend,treasury_curve}_…_2026-07-20.txt`, `cross_asset_sleeve_…_2026-09-28.txt` | `snapshot_members` | **KEEP AS IMMUTABLE FIXTURE** | Hand-chosen baskets. They are configuration, not external data. |
| `universe/sp500_tickers_snapshot_2026-06-09.txt` | archived midcap script; docs | **KEEP AS IMMUTABLE FIXTURE** | Evidence for the midcap studies. |
| `external/french/*.zip` (8) + `MANIFEST.json` | `lab/french_data.py`, `lab/century_leverage.py`, screens | **KEEP AS IMMUTABLE FIXTURE** | Byte-frozen evidence with sha manifest (factor-screen and momentum NO-GOs). A future refresh = new dated folder. |
| `external/shiller/ie_data.xls` + manifest | `century_leverage.py`, `lens2_build.py` | **KEEP AS IMMUTABLE FIXTURE** | Evidence for the century-leverage study. |
| `external/a23_inputs/{SPY,SSO,BIL,DTB3}.csv` + manifest | A23 scripts and tests; gate Y `--frozen-inputs` (`645d4ee2…`) | **KEEP AS IMMUTABLE FIXTURE** | Pinned by PLAN_V2 check 10. |
| `snapshots/MANIFEST.json` (+ git-ignored parquet tape) | `data/snapshot.py`, `load_bars` | **KEEP AS IMMUTABLE FIXTURE** | Already DB-derived and hashed. Note: the v1 tape lacks the 7 names that are new in v2. |
| `universe/kraken_listings.json`, `binanceus_listings.json`, `top_50_ccxt_volume.json` | none (crypto parked) | **ARCHIVE** | Move to `docs/archive/data/` or a tag. Do not delete: plans cite them. |
| `data/live/dashboard/*.html` (untracked, 28 MB, May 2026 crypto dashboards) | none | **DELETE** (untracked artefacts) | Check inbound references first, as CLAUDE.md requires. None were found in `.py` files. |
| `data/external/event_timing_audit/` (untracked, ~350 MB) | event-timing audit checkpoints | **DO NOT TOUCH** | Evidence working set. |
| DB tables `form4_transactions`, `earnings_surprise`, `fred_series`, `google_trends_interest` | study loaders | out of scope; flag | Already in the DB but **unversioned and overwritable**: the same vintage problem. Apply this pattern when a study cites them again. |

Counts (grouped as rows above): MOVE NOW 1, MOVE LATER 3, KEEP IMMUTABLE 6, ARCHIVE 1, DELETE 1, DO-NOT-TOUCH 1.

## 7. Testing without the real database

1. **Unit (FAST, runs in CI):** SQL builders return the expected text and parameters. A pure-Python `IntervalAlgebra`, shared with the loader, round-trips snapshots → intervals → snapshots on the v1 fixture and on random synthetic histories (leave/rejoin, empty diffs, renames). Supersession property: for random vintage sequences, the visible set at v equals the intervals of vintage v. Refusal rules fire on crafted shrinking and rewriting inputs. `FakeMembershipStore` implements Q1–Q4 in memory, so `DbMembershipSource` is tested without psycopg2. Selection: unset env → `MembershipSourceNotSelected`; `db` with a failing connection raises and never reads a fixture.
2. **Fixture equivalence (FAST, CI):** for every calendar day from 1996-01-02 to 2026-07-02, members from the `FixtureMembershipSource` v1 history equal members from intervals built from the same file. The existing P3 parity tests run on the fixture source unchanged.
3. **Disposable-schema integration** (`@pytest.mark.db` and `@pytest.mark.integration`, so the fast and full tiers exclude it; skipped unless `GGT_TEST_DB_URL` is set; never in CI). It refuses when the URL equals the production connection string, or the database is `ggtrader`, unless `GGT_TEST_DB_ALLOW_SHARED=1`. Even then it works only inside a fresh schema `ggt_test_<uuid>` (`search_path` set), dropped in `finally`. Cases: the DDL runs twice (idempotent); load v1, then v2 (forced); Q1–Q4; DB reader equals fixture reader on every calendar day, for both vintages; UPDATE, DELETE and TRUNCATE are refused by triggers; two concurrent loads of the same parent end with one refused; `export` is byte-exact.
4. **Production read-only check (manual, host):** `ggt db membership verify --load-id 1 --against-fixture sp500_v1_2026-06-02` checks per-day equality and the content hash over a read-only session.
5. Register the `db` marker in `pyproject.toml` and add it to `tests/tiers.py:FAST_EXCLUDED`. `tests/test_tiers.py` checks that the two lists agree.

## 8. Backups, ops, migration, effort

- **Backups.** `ggt db export` runs `pg_dump -Fd` on the whole database, so the new tables are included with no change. Borgmatic excludes `/home/flynn/ggtraderdb/*` ("backed up separately"; I did not verify the mechanism). So the weekly job also writes `pg_dump -t 'universe_*'` into git-ignored `data/exports/universe/`, which borgmatic does back up (about 0.3 MB per vintage). Two more recovery paths exist: the upstream commit of every vintage (fja05680's public history), and the fixtures promoted into git.
- **Live image:** it needs no new dependency (SQLAlchemy and psycopg2 are present) and no new environment variable (it reads with the existing `DB_*`). It does need the rebuilt code: push, wait for the image build, then `docker compose pull && up -d`, using the `ggtrader-deploy` skill.

Migration order (each step has its own rollback):

| Step | Action | Rollback |
|---|---|---|
| M0 | DDL (idempotent) | Drop the empty tables. Allowed only before M1. |
| M1 | Load vintage 1 from the repo file; `verify` against the fixture | Before any reader cuts over: drop and recreate (the only time this is allowed). |
| M2 | Ship the reader, fixtures manifest and call sites; tests use the v1 fixture | Revert the commit. |
| M3 | Live trader reads `db` (latest = v1); dry run; golden on `db:1` | Redeploy the previous image. |
| M4 | Load vintage 2 (forced) | `reinstate --load-id 1` (creates vintage 3 = v1 content). |
| M5 | Re-baseline oracles and live golden on v2; promote the v2 fixture | Drivers pin `fixture:sp500_v1…` again. |
| M6 | Enable the weekly timer | Disable the timer. |

Slices (ESTIMATE, agent-hours): **S1** schema + loader + interval algebra + unit tests (6). **S2** `MembershipSource` + fixtures manifest + equivalence and integration tests (6). **S3** call-site migration + provenance + `--membership` flags (6). **S4** live gate, fail-closed path, dry run, deploy (4). **S5** v2 load + oracle re-baseline + golden regeneration (4 + run time). **S6** timer + notifier + backup dump (3). **S7** docs, changelog, cli_reference (2). **Total ~31 h.** Run this after PLAN_V2's S1 (root conftest and tiers), because both edit those files. Do not edit any file that PLAN_V2 marks pinned or do-not-edit.

Oracles that must not move in S1–S4 (vintage 1, from either `fixture:` or `db:1`): core17 Sharpe **0.3905**, CAGR **3.50 %**, gates **1/17**; dsr0 **0.6374 / 3.41 % / 17/17**; live golden **120/120**. Expected in S5 on v2, according to the refresh agent's runs (not re-run by me): core17 0.3958 / 3.56 % / 1/17, dsr0 unchanged, and the golden equal on only 27/120 sessions, all differences after 2026-06-22. S5 therefore needs a new golden signed off by the owner.

## 9. Risks, owner decisions, unverified

Risks: (1) the weekly job dies and the live trader halts after N days. That is intended, but alerts must fire. (2) A bad tail-only auto-load reaches live for up to a week. (3) Upstream renames rewrite history, and `--auto` refuses more often than expected. Mitigation: the rename category, and tune the limits after three dry runs. (4) Readers forget to pin, and the "latest" drift returns. Mitigation: required flag in drivers, plus `MEMBERSHIP_DIFF` in compare. (5) The DB backup path is unverified. (6) Merging the refresh branch as an overwrite breaks pinned evidence.

Owner decisions, in plain words (recommended default in bold):
- **D1** How old may the membership check be before the live trader refuses to trade? **10 days once the weekly job runs; 45 days until then.**
- **D2** Should the weekly job load changes by itself? **Yes for recent changes only. Rewrites of older history wait for you.**
- **D3** Should the loader get its own database login that cannot delete? **Yes, later. The triggers protect the data now.**
- **D4** Keep the old file where it is, frozen, because a finished study points at it? **Yes, until the study files are moved.**
- **D5** After loading the November refresh, re-run the reference results and accept new reference numbers and a new live-signal reference? **Yes, as its own step, with your sign-off.**
- **D6** Commit a refreshed list to git only when a result depends on it? **Yes.**
- **D7** Do not merge the `refresh/sp500-membership` overwrite. Use its download as vintage 2. **Yes.**
- **D8** The upstream list stopped at 2026-08-18, so "our copy equals upstream" cannot see real lag. Should the live gate also bound how far the list's last date trails today? **Yes: warn at 90 days, refuse at 180.**

**Answered 2026-11-07.** The owner asked the lead to answer D1 to D8. The recommended defaults above are adopted, with two changes: D3 is revised per Appendix B5 (create the read and load roles now, enforce them later) and D8 is the Appendix A answer. D5 still needs the owner's sign-off at that step, not now. Any answer can be reversed before the build starts.

Not verified: the PostgreSQL and TimescaleDB versions (identity columns need PG 10+; nothing newer is used); how the `ggtraderdb` directory is actually backed up; the upstream commit of vintage 1, and whether fja05680 ever rewrites its git history; the true retrieval time of v1; the refresh agent's v2 oracle and golden figures (read from its notes, not re-run); Hermes cron and systemd timer details on Encom; whether the live container can reach GitHub; whether `TelegramNotifier` works from outside the trader process; how often upstream renames tickers.

## Appendix A: Fresh-reader check (2026-11-07)

A fresh reader (no contact with the author) read this spec as the builder and raised nine areas. Eight are settled below as **LEAD CALLS** (change them if you disagree); one needs the owner.

| # | Finding | Resolution |
|---|---|---|
| 1 | The staleness gate would halt replays, pinned vintages and the migration's own step M3 (v1 is 151 days past `verified_at`). | **LEAD CALL.** The gate applies only to a real run (`now` is wall-clock) that reads the latest vintage through `kind=db`. It is skipped when `now` is injected, when a specific vintage is pinned (`db:N`), and for fixtures. Migration order changes: load v2 (M4) **before** the live trader reads the database (M3). |
| 2 | `verified_at` only proves our copy equals upstream; upstream ended 2026-08-18 and had no commits after 2026-09-07, so the gate cannot see real lag. | **NEEDS THE OWNER (D8).** Recommended: the live gate also bounds `asof - coverage_end`: warn at 90 days, refuse at 180. A second free change source (for example the public index-changes table) is a separate later task. |
| 3 | `reinstate` (rollback) semantics: copied `retrieved_at` would trip the gate at once; the weekly job would stall diffing against the reinstated vintage. | **LEAD CALL.** A reinstated vintage gets `retrieved_at = verified_at = now`, records `reinstated_from`, and the weekly job diffs against the last **upstream-sourced** vintage, not the reinstated one. |
| 4 | `symbol_yf` is stored but unused, and a normaliser change would be invisible to the diff and the hash. | **LEAD CALL.** `symbol_yf` is derived, not identity. Each load records `normalizer_version`; the content checksum covers the raw tickers **and** the normalizer version. Readers normalise at read time. The 'renormalize' vintage source is dropped; a normaliser change is a new vintage with the same raw content and a new normalizer version. |
| 5 | CLI default `db` would make `GGT_MEMBERSHIP` never consulted, so CLI tests could hit the real database (the spinning-clock failure CLAUDE.md records). | **LEAD CALL.** Precedence: `--membership` flag, then `GGT_MEMBERSHIP`, then `db` for `ggt paper` and `ggt lab` only outside pytest. The repo-root `conftest.py` autouse fixture sets `GGT_MEMBERSHIP` to an immutable fixture for every test, and a sentinel test fails if any test can reach the database. |
| 6 | `load_sp500_history()` becoming a source-selecting wrapper would break the event-timing audit and about 30 closed scripts. | **LEAD CALL.** `load_sp500_history(csv_path)` stays a pure file reader whenever a path is given; only the no-argument form selects a source. Closed research scripts may break only if they are archived with a note that they run at the legacy tag; the others keep working through the path form. |
| 7 | 'Historical' versus 'tail' rewrite is undefined for `--auto`. | **LEAD CALL.** A change is a tail change only if every day whose membership differs lies **after** the parent vintage's `coverage_end`. The set compared is the symmetric difference of covered days. Re-dating the most recent event is therefore a historical rewrite and `--auto` refuses; a human decides. |
| 8 | `pg_dump -t universe_*` omits the trigger functions; `universe_membership` lacks a TRUNCATE trigger. | **LEAD CALL.** The DDL ships as a versioned file in git and is idempotent. Restore means: run the DDL, then a data-only restore; the restore drill tests exactly that order. Add the missing TRUNCATE trigger on `universe_membership` (the spec's 'repeat the two triggers' line is corrected). |
| 9 | Small items: gzip header bytes change the checksum; `tests/conftest.py`; timezone-aware dates shifting a day in SQL. | **LEAD CALL.** The loader stores input bytes verbatim; 'export is byte-exact' means against `raw_sha256` (uncompressed), and the stored gzip is exported unchanged. The `GGT_MEMBERSHIP` autouse fixture lives in the repo-root `conftest.py` (the S1 work moved it there). SQL helpers accept only `datetime.date` and cast with `%s::date`; a test passes a UTC-aware Timestamp and expects a clear error (the container timezone is America/Los_Angeles, the same family of bug as the earlier day-shift). |

Owner decisions still open: D1 to D7 (see the main decision list) and D8 (item 2 above).

## Appendix B: PostgreSQL best-practice review (2026-11-07)

Reviewed against the postgres-best-practices guide, with read-only checks of the real server: **PostgreSQL 16.11** (supported to November 2028), **TimescaleDB 2.25.1**, session time zone **America/Los_Angeles**, schemas: `public` plus Timescale's own, extensions: `btree_gist` available but **not installed**, and exactly **one** role, `ggtrader`, which is a **superuser** used by the trader, research and everything else. These changes **amend section 1** (apply them when writing the DDL; the tables, queries and the four-table shape are unchanged). All are LEAD CALLS except where marked.

| # | Change | Why (guide section) |
|---|---|---|
| B1 | Put every object in a dedicated schema `universe` (tables `loads`, `membership`, `snapshot_dates`, `checks`, `load_blobs`; the `universe_` prefixes go away). Drop `PUBLIC` access to it. | Schema-based access control; and `pg_dump -n universe` then captures the functions, triggers and grants that `-t` missed (resolves Appendix A item 8 properly). Nothing else lives in `public` today. |
| B2 | Block overlapping live intervals with an exclusion constraint: `CREATE EXTENSION IF NOT EXISTS btree_gist;` then `EXCLUDE USING gist (index_name WITH =, symbol_raw WITH =, daterange(valid_from, valid_to, '[)') WITH &&) WHERE (superseded_by_load_id IS NULL)`. The current partial unique index on `(index_name, symbol_raw, valid_from)` does not stop `(X, 2020, 2023)` and `(X, 2021, NULL)` coexisting; the exclusion constraint does, and its GiST index replaces the unique index. | Schema design: exclusion constraints. `WITHOUT OVERLAPS` is PG18+ and the server is 16, so the exclusion form is the right one. A loader bug that double-inserts becomes a refused transaction. |
| B3 | Drop `universe_membership_range_ix` and `universe_membership_lineage_ix` for now. The table is about 1,300 live rows (about 2,600 with superseded ones): a sequential scan is far below a millisecond, so extra B-trees only add write cost and noise. Keep the primary keys and the exclusion index. Add an index only if `EXPLAIN` shows a need. Foreign-key columns do not need indexes here either: parents are never deleted (the triggers refuse it). | Indexing: finding unused indexes; FK-index rule applies to deletes. |
| B4 | Replace `CHAR(64)` with `TEXT` plus the existing hex `CHECK` (CHAR pads and buys nothing). Move the gzip bytes into `universe.load_blobs(load_id PRIMARY KEY REFERENCES loads, raw_gzip BYTEA)` with `ALTER COLUMN raw_gzip SET STORAGE EXTERNAL`, so listing loads never drags a blob and the already-compressed bytes are not compressed twice. | Schema design: data types and TOAST. |
| B5 | Roles. Today every connection is a superuser, so `GRANT`/`REVOKE` cannot restrict it; the **append-only triggers fire even for superusers and are the real guard now**. Add `universe_reader` (USAGE on schema, SELECT) and `universe_loader` (INSERT on the tables, `UPDATE (superseded_by_load_id)` as a **column privilege** on `membership`, USAGE on sequences, EXECUTE on the functions) as NOLOGIN group roles in the DDL, plus login roles later. New code reads the connection string for them from `GGT_MEMBERSHIP_DB_URL` / `GGT_MEMBERSHIP_LOADER_URL` and **defaults to the existing connection** until the owner switches. **This revises D3:** create the roles now (cheap, in the DDL); enforcing them is a separate step. | Security and roles: least privilege, column privileges, default privileges. |
| B6 | Every function gets `SET search_path = universe, pg_catalog` (and `pg_temp` last for any `SECURITY DEFINER`, which this design avoids). | Security: search path. |
| B7 | **Concurrent loads.** The loader takes `pg_advisory_xact_lock(hashtext('universe:' \|\| index_name))` as its first statement and re-reads the latest vintage **after** the lock, refusing if it changed (compare and set). Without it, the weekly timer and a manual `--force` can both supersede the same rows. READ COMMITTED is enough with the lock. **Readers** that need the latest vintage and its rows (the live gate does) use a single statement (a CTE that selects the latest `load_id`) or `BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY`, never two separate queries. A two-session test proves it. | Transaction isolation: lost updates, advisory locks, read skew. |
| B8 | Date parameters are `datetime.date` only, bound as `%s::date`. The session zone is America/Los_Angeles, so a timezone-aware timestamp compared with a DATE column shifts a day; a test passes a UTC-aware `pd.Timestamp` and expects a clear error. | Schema design: timestamps; the project's earlier day-shift bug. |
| B9 | Loading ~4,000 rows uses `psycopg2.extras.execute_values` in one transaction. **Do not disable triggers** for speed (the bulk-loading guide's trick): the triggers are the integrity guard and the volume is tiny. | Bulk loading. |
| B10 | `source` and `outcome` stay `TEXT` with `CHECK` lists (a small closed set), and `index_name` stays a regex-checked `TEXT` until a second index exists. If it does, introduce a lookup table then. | Enum vs check vs lookup table. |
| B11 | Backups: with B1, the weekly dump is `pg_dump -n universe --format=custom`; restore is `CREATE EXTENSION btree_gist`, then `pg_restore`. The restore drill runs exactly that on a scratch database. The DDL ships as the versioned file `sql/universe_schema.sql`. | Backup and restore. |
| B12 | No Timescale features, no hypertables, no tuning: the tables are tiny, default autovacuum is fine, and `pg_stat_statements` is not installed (do not add it for this). | Performance diagnostics: do not tune what is not slow. |

Owner decisions: none new. D3 is revised as B5 says (roles created now, enforced later); D8 (upstream lag) is unchanged.

## Appendix C: Revisions accepted by the owner (2026-11-08)

Context: the upstream feed (fja05680/sp500) stopped on 2026-08-18 and its repository is dead. A research report on free alternatives and an independent Opus review (see `research/` notes if archived, and the scratch evidence under the session folder) concluded that the system no longer has an upstream to copy; from now on the list is written from evidence. The owner accepted three changes. Where this appendix contradicts an earlier section, **this appendix wins**.

**C1. The weekly job proposes; a human loads.** (Replaces D2, the `--auto` text in sections 0 and 5, and the auto-load path of section 2.)
- The job fetches and archives the raw bytes of every source it uses (Wikipedia ledger via the MediaWiki API, the Wikipedia current-constituents page, chinobing/historical_sp500_constituents CSVs, the SPDR SPY holdings xlsx, and any S&P DJI press release it can find), checks our latest vintage against the SPY holdings, and records the outcome in `universe.checks`.
- A match writes an `unchanged`/`matched` check and extends the freshness clock. Any difference writes a **proposed events file** and sends it to Telegram; it never loads.
- A human confirms the effective dates against the S&P DJI press release and runs `ggt db load-sp500 --from-events FILE --commit`.
- The job refuses and alerts when: our list differs from SPY for the holdings' as-of date; a ledger row is dated in the future or on its announcement date; a removal has no matching addition and the count would leave 480 to 520; a source is missing or has not updated in 14 days; a source's page structure changed; or the SPY fetch fails.
- Auto-load may be reconsidered later, only when all sources agree and the effective date comes from the press release.

**C2. The live golden stays pinned to the original file.** (Replaces the re-baseline of the golden in D5 and migration step M5.)
- `scripts/p7_live_golden.py` keeps reading `fixture:sp500_v1_2026-06-02` permanently: it checks the rebuilt trader against what the legacy trader actually did, and that trader read the vintage-1 file. It is never re-baselined.
- A separate regression test on the current list may be added. The core17 and dsr0 reference numbers are re-baselined once, on vintage 3, with the owner's sign-off (C5).

**C3. The gate counts from an independent match.** (Replaces the `verified_at` meaning in section 3.4, D1 and D8.)
- `verified_at` for a vintage is the time of the last successful match between its latest set and the SPDR SPY holdings file for the holdings' own as-of date. A check against an upstream feed does not extend it.
- `known_through` is the latest date through which the list is attested: each successful SPY match sets it to that holdings as-of date (and never to a date earlier than the vintage's own coverage end).
- Gate: refuse when the age of the last SPY match exceeds the limit (45 days now; 10 once the weekly job runs), and warn from 90 days and refuse from 180 days of `asof - known_through`. A quiet spell without index changes therefore raises no alarm; a dead pipeline does.

**C4. Vintages and the gap-fill.**
- Vintage 2: the fja05680 download of 2026-11-07 (coverage to 2026-08-18), loaded verbatim with `--force` (10 historical rewrites are expected, by design).
- Vintage 3: `source='curated'`, `parent_load_id=2`, no upstream commit. The raw bytes are the full generated `date,tickers` CSV; an events file (date, action, ticker, announcement date, press-release URL, sha256 of the saved HTML) and the saved press-release HTML are stored in `load_blobs`. Renames stay "remove plus add" (as in all earlier data); company continuity goes in a new table `universe.symbol_changes(load_id, old_symbol, new_symbol, effective_date, evidence)`.
- The gap events (all effective dates confirmed against S&P DJI press releases by an independent review): 2026-09-21 +BE +ILMN +P -TTD -BLDR -TAP; 2026-10-01 +VYLR; 2026-10-06 +TWLO +SKYD -CTVA -WBD -PSKY (PSKY to SKYD is the Paramount Skydance rename, effective 2026-10-06); 2026-11-04 +OKTA -KVUE. The index has 504 names from 2026-10-01 to 2026-10-05 (Vylor joins one business week before Corteva leaves): vintage 3 therefore needs dated snapshots on each of the four dates, and no code may assume 503.
- Proof: core17 and dsr0 must come out bit-identical on vintages 2 and 3 (their evaluation ends 2026-04-30), and the vintage-3 set must equal the SPY holdings set.

**C5. Slices from here (replacing the old S4 remainder to S6).**
- M4: gate as first built (done).
- M4b: redefine the gate per C3 (SPY holdings fetch with a standard-library xlsx reader, `ggt db membership check-holdings`, `known_through`).
- M5a: load vintage 2 verbatim; add `export` and `reinstate`.
- M5b: curated vintage 3, `symbol_changes`, the four-date snapshots, the SPY match, the oracle proof (bit-identical to vintage 2); also check for stale price bars under reused tickers (for example `P`).
- M5c: STOP for the owner's sign-off: show the vintage 1 to 3 interval diff and the reference-number change (core17 Sharpe 0.3905 to about 0.3958, expected from the refresh agent's runs); then re-baseline core17 and dsr0 on vintage 3 and promote the fixture. The live golden stays on vintage 1 (C2).
- M6: the weekly fetch, archive, cross-check, alert and propose job (C1), with its timer disabled until three manual dry runs match expectations, the Telegram path is tested through `DryRunNotifier`, a restore drill of the new tables is done and the owner enables it.
- Then: independent final review, merge and push (owner's yes).

**C6. Open checks (not yet verified).** Any S&P release between 2026-08-18 and 2026-11-08 beyond the four reviewed (an add-then-remove inside the gap would be missed); whether KVUE's removal slipped past 2026-11-04 (SPY and the maintained snapshot say it happened by 2026-11-05); whether S&P announced Vylor before 2026-10-01; the exact effective dates of the 2026 renames SATS to ECHO and BK to BNY (fja05680 and the maintained repo differ by one day); whether the `ohlcv` table holds old bars under the reused ticker `P`. **Closed 2026-11-09:** all five checks are answered in the [evidence note](../../../research/programme/membership-database/2026-11-09-open-checks-closed-with-press-releases.md); vintage 3 needs no change.

**C7. What the content hash covers, and the normalizer version (2026-11-09, review).** `content_sha256` covers the snapshot dates and the *raw* tickers only, not the derived `symbol_yf` column. The normalizer version (`universe.loads.normalizer_version`, `NORMALIZER_VERSION` in code) is stored beside it and is not part of the hash. This replaces the Appendix A4 wording that the checksum covers the normalizer version; the hash definition stays as built so every recorded `membership_sha256` remains valid. The loader's no-op test now compares both: identical raw bytes loaded under the *same* normalizer version are a no-op (an `unchanged` check), identical bytes under a *different* version are written as a new vintage with the same raw and content hash and a new normalizer version.
