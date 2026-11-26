CREATE SCHEMA IF NOT EXISTS sec;

CREATE TABLE IF NOT EXISTS ref.load_files (
    load_id bigint,
    path text,
    sha256 text,
    bytes bigint,
    PRIMARY KEY (load_id, path)
);

CREATE TABLE IF NOT EXISTS ref.load_coverage (
    load_id bigint PRIMARY KEY REFERENCES ref.loads(load_id),
    coverage text NOT NULL CHECK (coverage IN (
        'includes_dead_stocks', 'survivors_only', 'partial_dead_stocks',
        'large_stocks_only', 'unaudited_sample', 'not_a_stock_source'
    )),
    note text NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ref.integrity_checks (
    check_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ran_at timestamptz NOT NULL DEFAULT now(),
    mode text NOT NULL CHECK (mode IN ('ordinary', 'deep')),
    ok boolean NOT NULL,
    report jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS sec.fetched_files (
    load_id bigint,
    url_hash text,
    url text,
    content_sha256 text,
    fetched_at timestamptz,
    status int,
    bytes bigint,
    path text,
    kind text CHECK (kind IN (
        'quarterly_index', 'submission_list', 'directory_listing',
        'form25_xml', 'filing_text', 'other'
    )),
    PRIMARY KEY (load_id, url_hash)
);

CREATE TABLE IF NOT EXISTS sec.index_entries (
    load_id bigint,
    quarter text,
    cik bigint,
    company_name text,
    form_type text,
    date_filed date,
    filename text
);
CREATE INDEX IF NOT EXISTS index_entries_load_cik_date_idx
    ON sec.index_entries (load_id, cik, date_filed);
CREATE INDEX IF NOT EXISTS index_entries_load_form_date_idx
    ON sec.index_entries (load_id, form_type, date_filed);
-- Finds a filing by its accession number (the part of the file name before ".txt"): about 0.02
-- seconds instead of a scan of every row; `verify-data` and any study use this exact expression.
CREATE INDEX IF NOT EXISTS index_entries_accession_idx
    ON sec.index_entries ((substring(filename from '/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$')));

CREATE TABLE IF NOT EXISTS sec.filing_headers (
    load_id bigint,
    accession text,
    cik bigint,
    form_type text,
    filed_date date,
    items text[],
    PRIMARY KEY (load_id, accession, cik)
);

CREATE TABLE IF NOT EXISTS sec.death_labels (
    load_id bigint,
    study text,
    method_version text,
    permno bigint,
    last_date date,
    release_id bigint,
    label text,
    confidence text,
    evidence_text text,
    suggested_loss double precision,
    accessions text[],
    source_file text,
    source_file_sha256 text,
    source_row_sha256 text,
    source_fields jsonb,
    PRIMARY KEY (load_id, study, method_version, permno, last_date)
);

-- Readers: rows of a load that is still running or failed are never returned.
CREATE OR REPLACE FUNCTION sec.fetched_files_of(p_load_id bigint)
RETURNS SETOF sec.fetched_files LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
        RAISE EXCEPTION 'load % is not complete', p_load_id;
    END IF;
    RETURN QUERY SELECT * FROM sec.fetched_files WHERE load_id = p_load_id;
END $$;

CREATE OR REPLACE FUNCTION sec.index_entries_of(p_load_id bigint)
RETURNS SETOF sec.index_entries LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
        RAISE EXCEPTION 'load % is not complete', p_load_id;
    END IF;
    RETURN QUERY SELECT * FROM sec.index_entries WHERE load_id = p_load_id;
END $$;

-- Fast filtered reads: the planner pushes a company or date filter into the index, which the
-- function above cannot do. Only rows of completed loads are visible.
CREATE OR REPLACE VIEW sec.index_entries_complete AS
SELECT e.* FROM sec.index_entries e
JOIN ref.loads l ON l.load_id = e.load_id
WHERE l.status = 'complete';

CREATE OR REPLACE FUNCTION sec.filing_headers_of(p_load_id bigint)
RETURNS SETOF sec.filing_headers LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
        RAISE EXCEPTION 'load % is not complete', p_load_id;
    END IF;
    RETURN QUERY SELECT * FROM sec.filing_headers WHERE load_id = p_load_id;
END $$;

CREATE OR REPLACE FUNCTION sec.death_labels_of(p_load_id bigint)
RETURNS SETOF sec.death_labels LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
        RAISE EXCEPTION 'load % is not complete', p_load_id;
    END IF;
    RETURN QUERY SELECT * FROM sec.death_labels WHERE load_id = p_load_id;
END $$;

CREATE OR REPLACE FUNCTION ref.require_coverage() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status = 'running' AND NEW.status = 'complete'
        AND (NEW.dataset IS NULL
            OR NEW.dataset NOT IN (
                'crsp-daily', 'crsp-daily-pilot', 'mcgill-fiam', 'crsp-derived-bars'))
        AND NOT EXISTS (SELECT 1 FROM ref.load_coverage WHERE load_id = NEW.load_id) THEN
        RAISE EXCEPTION 'coverage label required';
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS require_coverage ON ref.loads;
CREATE TRIGGER require_coverage BEFORE UPDATE ON ref.loads
    FOR EACH ROW EXECUTE FUNCTION ref.require_coverage();
ALTER TABLE ref.loads ENABLE ALWAYS TRIGGER require_coverage;

DO $$
DECLARE
    target regclass;
BEGIN
    FOREACH target IN ARRAY ARRAY[
        'ref.load_files'::regclass, 'ref.load_coverage'::regclass, 'ref.integrity_checks'::regclass,
        'sec.fetched_files'::regclass, 'sec.index_entries'::regclass,
        'sec.filing_headers'::regclass, 'sec.death_labels'::regclass
    ] LOOP
        -- A row trigger on update or delete costs the bulk load nothing: it never fires on insert.
        EXECUTE format('DROP TRIGGER IF EXISTS frozen ON %s', target);
        EXECUTE format(
            'CREATE TRIGGER frozen BEFORE UPDATE OR DELETE ON %s'
            ' FOR EACH ROW EXECUTE FUNCTION ref.reject_change()', target
        );
        EXECUTE format('ALTER TABLE %s ENABLE ALWAYS TRIGGER frozen', target);
        EXECUTE format('DROP TRIGGER IF EXISTS no_truncate ON %s', target);
        EXECUTE format(
            'CREATE TRIGGER no_truncate BEFORE TRUNCATE ON %s'
            ' FOR EACH STATEMENT EXECUTE FUNCTION ref.reject_change()', target
        );
        EXECUTE format('ALTER TABLE %s ENABLE ALWAYS TRIGGER no_truncate', target);
    END LOOP;
END $$;

INSERT INTO ref.load_coverage (load_id, coverage, note)
SELECT l.load_id, legacy.coverage, legacy.note
FROM ref.loads l
JOIN (VALUES
    ('crsp-daily', 'includes_dead_stocks', 'CRSP; no delisting returns'),
    ('mcgill-fiam', 'large_stocks_only', 'McGill-FIAM, 2000 to 2023, large stocks'),
    ('crsp-derived-bars', 'includes_dead_stocks', 'derived from the CRSP load')
) AS legacy(dataset, coverage, note) ON legacy.dataset = l.dataset
WHERE l.status = 'complete'
  AND NOT EXISTS (SELECT 1 FROM ref.load_coverage c WHERE c.load_id = l.load_id);
