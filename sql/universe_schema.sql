-- Versioned index membership ("universe" schema): idempotent DDL, safe to run again at any time.
--
-- Spec: docs/superpowers/specs/2026-11-07-membership-database-design.md (section 1.2, amended by
-- Appendix A and B). Applied by `ggt db membership apply-ddl` (src/ggTrader/data/universe_db.py).
--
-- Rules this file keeps:
--   * every object lives in schema `universe`; nothing in `public` or any other schema is touched
--     (btree_gist is installed INTO `universe`, never into `public`);
--   * every statement is idempotent (IF NOT EXISTS / CREATE OR REPLACE / DROP TRIGGER IF EXISTS);
--   * no `timestamp without time zone` column (the naive-UTC trap of cached_yfinance_loader.py);
--   * append-only is enforced by triggers, which fire for every role including superusers.

CREATE SCHEMA IF NOT EXISTS universe;
REVOKE ALL ON SCHEMA universe FROM PUBLIC;

-- The extension's operator classes land in `universe`; the exclusion constraint below finds the
-- default gist operator class by type, not by search_path. If the extension is already installed
-- elsewhere (an older database), IF NOT EXISTS leaves it where it is.
CREATE EXTENSION IF NOT EXISTS btree_gist SCHEMA universe;

CREATE TABLE IF NOT EXISTS universe.loads (
    load_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name         TEXT        NOT NULL CHECK (index_name ~ '^[a-z0-9_]+$'),
    source             TEXT        NOT NULL
                       CHECK (source IN ('fja05680/sp500', 'fixture-import', 'reinstate', 'curated')),
    source_url         TEXT        NOT NULL,
    upstream_commit    TEXT        CHECK (upstream_commit ~ '^[0-9a-f]{40}$'),  -- NULL = unknown
    upstream_etag      TEXT,
    retrieved_at       TIMESTAMPTZ NOT NULL,
    raw_sha256         TEXT        NOT NULL CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),  -- uncompressed
    content_sha256     TEXT        NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    normalizer_version INT         NOT NULL CHECK (normalizer_version > 0),
    coverage_start     DATE        NOT NULL,
    coverage_end       DATE        NOT NULL CHECK (coverage_end >= coverage_start),
    n_snapshot_dates   INT         NOT NULL CHECK (n_snapshot_dates > 0),
    n_intervals        INT         NOT NULL CHECK (n_intervals > 0),
    n_symbols          INT         NOT NULL CHECK (n_symbols > 0),
    min_members        INT         NOT NULL,
    max_members        INT         NOT NULL CHECK (max_members >= min_members),
    rows_added         INT         NOT NULL,
    rows_superseded    INT         NOT NULL,
    parent_load_id     BIGINT      REFERENCES universe.loads (load_id),
    reinstates_load_id BIGINT      REFERENCES universe.loads (load_id),
    forced             BOOLEAN     NOT NULL DEFAULT false,
    force_reason       TEXT,
    loader_git_sha     TEXT        NOT NULL,
    note               TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    loaded_by          TEXT        NOT NULL DEFAULT current_user,
    CHECK (NOT forced OR force_reason IS NOT NULL)
);

-- Widen the source list on a table created by an earlier version of this file (M5b adds
-- 'curated'). Skipped when the constraint already allows it. No row is rewritten.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'universe.loads'::regclass AND conname = 'loads_source_check'
          AND pg_get_constraintdef(oid) LIKE '%curated%'
    ) THEN
        ALTER TABLE universe.loads DROP CONSTRAINT IF EXISTS loads_source_check;
        ALTER TABLE universe.loads ADD CONSTRAINT loads_source_check CHECK (
            source IN ('fja05680/sp500', 'fixture-import', 'reinstate', 'curated'));
    END IF;
END $$;

-- The uncompressed bytes' gzip, kept apart so listing loads never reads a blob (B4).
CREATE TABLE IF NOT EXISTS universe.load_blobs (
    load_id  BIGINT PRIMARY KEY REFERENCES universe.loads (load_id),
    raw_gzip BYTEA  NOT NULL
);
ALTER TABLE universe.load_blobs ALTER COLUMN raw_gzip SET STORAGE EXTERNAL;

CREATE TABLE IF NOT EXISTS universe.snapshot_dates (
    load_id       BIGINT NOT NULL REFERENCES universe.loads (load_id),
    snapshot_date DATE   NOT NULL,
    n_members     INT    NOT NULL CHECK (n_members > 0),
    PRIMARY KEY (load_id, snapshot_date)
);

-- Half-open intervals [valid_from, valid_to); NULL valid_to = member at the end of coverage.
-- The exclusion constraint (B2) stops two live intervals of one symbol from overlapping; its
-- GiST index is the table's only secondary index (B3: ~1,300 rows, a seq scan is fine).
CREATE TABLE IF NOT EXISTS universe.membership (
    interval_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name            TEXT   NOT NULL,
    symbol_raw            TEXT   NOT NULL CHECK (symbol_raw <> '' AND symbol_raw = btrim(symbol_raw)),
    symbol_yf             TEXT   NOT NULL CHECK (symbol_yf <> ''),
    valid_from            DATE   NOT NULL,
    valid_to              DATE,
    load_id               BIGINT NOT NULL REFERENCES universe.loads (load_id),
    superseded_by_load_id BIGINT REFERENCES universe.loads (load_id),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    CHECK (superseded_by_load_id IS NULL OR superseded_by_load_id > load_id),
    CONSTRAINT membership_no_live_overlap EXCLUDE USING gist (
        index_name WITH =,
        symbol_raw WITH =,
        daterange(valid_from, valid_to, '[)') WITH &&
    ) WHERE (superseded_by_load_id IS NULL)
);

-- One row per check run. 'matched' / 'mismatch' come from the SPDR SPY holdings comparison
-- (Appendix C, C3: a 'matched' row is the only thing that advances the live gate);
-- 'unchanged' / 'refused' are the informational upstream checks and 'loaded' is a load's own row.
CREATE TABLE IF NOT EXISTS universe.checks (
    check_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_name      TEXT        NOT NULL,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_sha256      TEXT,
    upstream_commit TEXT,
    outcome         TEXT        NOT NULL,
    matched_load_id BIGINT      REFERENCES universe.loads (load_id),
    detail          JSONB
);

-- Widen the outcome list on a table created by an earlier version of this file (M4b). Skipped
-- when the constraint already allows 'mismatch'. No row is rewritten: ADD CONSTRAINT only scans
-- the (tiny) table to validate it.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'universe.checks'::regclass AND conname = 'checks_outcome_check'
          AND pg_get_constraintdef(oid) LIKE '%mismatch%'
    ) THEN
        ALTER TABLE universe.checks DROP CONSTRAINT IF EXISTS checks_outcome_check;
        ALTER TABLE universe.checks ADD CONSTRAINT checks_outcome_check CHECK (
            outcome IN ('unchanged', 'loaded', 'refused', 'fetch_failed', 'matched', 'mismatch'));
    END IF;
END $$;

-- The raw bytes behind a holdings check (the SPY workbook), gzipped, kept apart like load_blobs.
CREATE TABLE IF NOT EXISTS universe.check_blobs (
    check_id   BIGINT PRIMARY KEY REFERENCES universe.checks (check_id),
    raw_sha256 TEXT  NOT NULL CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),  -- of the uncompressed bytes
    raw_gzip   BYTEA NOT NULL
);
ALTER TABLE universe.check_blobs ALTER COLUMN raw_gzip SET STORAGE EXTERNAL;

-- Every source the weekly job fetched (M6): one row per source per run, with the exact bytes
-- (gzip) so "what did Wikipedia say that Saturday?" can be answered from the database alone.
-- The SPY workbook of a run stays in check_blobs (raw_gzip is NULL for it here); a proposed
-- events file is stored here too (source 'proposal'). Append-only like every other table.
CREATE TABLE IF NOT EXISTS universe.source_snapshots (
    snapshot_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    check_id    BIGINT      NOT NULL REFERENCES universe.checks (check_id),
    source      TEXT        NOT NULL CHECK (source <> ''),
    url         TEXT        NOT NULL,
    status      TEXT        NOT NULL CHECK (status IN ('ok', 'failed')),
    fetched_at  TIMESTAMPTZ NOT NULL,
    last_change DATE,        -- the source's own last-change date (NULL = unknown)
    raw_sha256  TEXT        CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),  -- of the uncompressed bytes
    size_bytes  INT         CHECK (size_bytes >= 0),
    raw_gzip    BYTEA,
    detail      JSONB,
    UNIQUE (check_id, source)
);
ALTER TABLE universe.source_snapshots ALTER COLUMN raw_gzip SET STORAGE EXTERNAL;

-- Company continuity across a ticker change (M5b). Membership itself stays "remove plus add"
-- as in every earlier vintage; this table says that two symbols are one company, from when,
-- and on what evidence. Written together with the curated vintage that states it.
CREATE TABLE IF NOT EXISTS universe.symbol_changes (
    load_id        BIGINT NOT NULL REFERENCES universe.loads (load_id),
    old_symbol     TEXT   NOT NULL CHECK (old_symbol <> '' AND old_symbol = btrim(old_symbol)),
    new_symbol     TEXT   NOT NULL CHECK (new_symbol <> '' AND new_symbol = btrim(new_symbol)),
    effective_date DATE   NOT NULL,
    evidence       TEXT   NOT NULL,
    PRIMARY KEY (load_id, old_symbol, new_symbol, effective_date),
    CHECK (old_symbol <> new_symbol)
);

-- The events file and the saved press-release pages behind a curated vintage, gzipped, so the
-- evidence is reproducible from the database alone (M5b; spec C4 calls this part of load_blobs).
CREATE TABLE IF NOT EXISTS universe.load_evidence (
    load_id    BIGINT NOT NULL REFERENCES universe.loads (load_id),
    name       TEXT   NOT NULL CHECK (name <> ''),
    kind       TEXT   NOT NULL CHECK (kind IN ('events_file', 'evidence')),
    source_url TEXT,
    sha256     TEXT   NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),  -- of the uncompressed bytes
    size_bytes INT    NOT NULL CHECK (size_bytes >= 0),
    content_gz BYTEA  NOT NULL,
    PRIMARY KEY (load_id, name)
);
ALTER TABLE universe.load_evidence ALTER COLUMN content_gz SET STORAGE EXTERNAL;

-- Append-only guards (B6: every function pins its search_path).
CREATE OR REPLACE FUNCTION universe.membership_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = universe, pg_catalog AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'universe.membership is append-only (DELETE refused)';
    END IF;
    IF OLD.superseded_by_load_id IS NOT NULL
       OR (NEW.interval_id, NEW.index_name, NEW.symbol_raw, NEW.symbol_yf, NEW.valid_from,
           NEW.valid_to, NEW.load_id)
          IS DISTINCT FROM
          (OLD.interval_id, OLD.index_name, OLD.symbol_raw, OLD.symbol_yf, OLD.valid_from,
           OLD.valid_to, OLD.load_id)
    THEN
        RAISE EXCEPTION 'only superseded_by_load_id may change, and only once';
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION universe.frozen_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = universe, pg_catalog AS $$
BEGIN
    RAISE EXCEPTION '% is append-only (% refused)', TG_TABLE_NAME, TG_OP;
END $$;

DROP TRIGGER IF EXISTS membership_guard_t ON universe.membership;
CREATE TRIGGER membership_guard_t BEFORE UPDATE OR DELETE ON universe.membership
    FOR EACH ROW EXECUTE FUNCTION universe.membership_guard();
DROP TRIGGER IF EXISTS membership_truncate_t ON universe.membership;
CREATE TRIGGER membership_truncate_t BEFORE TRUNCATE ON universe.membership
    FOR EACH STATEMENT EXECUTE FUNCTION universe.frozen_guard();

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['loads', 'load_blobs', 'snapshot_dates', 'checks', 'check_blobs',
                          'symbol_changes', 'load_evidence', 'source_snapshots'] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON universe.%I', t || '_frozen_t', t);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON universe.%I '
            'FOR EACH ROW EXECUTE FUNCTION universe.frozen_guard()', t || '_frozen_t', t);
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON universe.%I', t || '_truncate_t', t);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE TRUNCATE ON universe.%I '
            'FOR EACH STATEMENT EXECUTE FUNCTION universe.frozen_guard()', t || '_truncate_t', t);
    END LOOP;
END $$;

-- Snapshot reconstruction of any vintage (the "view" of shape (c)).
CREATE OR REPLACE FUNCTION universe.snapshot(p_index TEXT, p_load BIGINT)
RETURNS TABLE (snapshot_date DATE, symbol_raw TEXT, symbol_yf TEXT)
LANGUAGE sql STABLE SET search_path = universe, pg_catalog AS $$
    SELECT s.snapshot_date, m.symbol_raw, m.symbol_yf
    FROM universe.snapshot_dates s
    JOIN universe.membership m
      ON m.index_name = p_index
     AND m.load_id <= p_load
     AND (m.superseded_by_load_id IS NULL OR m.superseded_by_load_id > p_load)
     AND m.valid_from <= s.snapshot_date
     AND (m.valid_to IS NULL OR m.valid_to > s.snapshot_date)
    WHERE s.load_id = p_load
$$;

-- Roles (B5). NOLOGIN group roles: login roles are added later, by the owner. Today every
-- connection is the existing superuser, so these grants document intent; the triggers above are
-- the guard that holds now.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'universe_reader') THEN
        CREATE ROLE universe_reader NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'universe_loader') THEN
        CREATE ROLE universe_loader NOLOGIN;
    END IF;
END $$;

GRANT USAGE ON SCHEMA universe TO universe_reader, universe_loader;
GRANT SELECT ON universe.loads, universe.load_blobs, universe.snapshot_dates,
    universe.membership, universe.checks, universe.check_blobs, universe.symbol_changes,
    universe.load_evidence, universe.source_snapshots TO universe_reader, universe_loader;
ALTER DEFAULT PRIVILEGES IN SCHEMA universe GRANT SELECT ON TABLES TO universe_reader;

GRANT INSERT ON universe.loads, universe.load_blobs, universe.snapshot_dates,
    universe.membership, universe.checks, universe.check_blobs, universe.symbol_changes,
    universe.load_evidence, universe.source_snapshots TO universe_loader;
-- The only UPDATE the loader may ever do, as a column privilege.
GRANT UPDATE (superseded_by_load_id) ON universe.membership TO universe_loader;
GRANT USAGE ON SEQUENCE universe.loads_load_id_seq, universe.membership_interval_id_seq,
    universe.checks_check_id_seq, universe.source_snapshots_snapshot_id_seq TO universe_loader;

REVOKE ALL ON FUNCTION universe.membership_guard(), universe.frozen_guard(),
    universe.snapshot(TEXT, BIGINT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION universe.membership_guard(), universe.frozen_guard(),
    universe.snapshot(TEXT, BIGINT) TO universe_reader, universe_loader;
