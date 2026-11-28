-- Append-only forward collection storage. Apply only with owner authorization.
CREATE SCHEMA IF NOT EXISTS forward;

CREATE TABLE IF NOT EXISTS forward.loads (
    load_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    day date NOT NULL,
    source text NOT NULL,
    raw_sha256 text NOT NULL CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),
    normalized_sha256 text NOT NULL CHECK (normalized_sha256 ~ '^[0-9a-f]{64}$'),
    code_commit text NOT NULL,
    actor text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Catch-up rows use the catch-up day with an earlier listing; inspect forward.loads.source.
CREATE TABLE IF NOT EXISTS forward.listing_snapshots (
    load_id bigint NOT NULL REFERENCES forward.loads(load_id),
    day date NOT NULL,
    symbol text NOT NULL,
    exchange text NOT NULL,
    security_name text NOT NULL,
    is_etf boolean NOT NULL,
    is_test boolean NOT NULL,
    PRIMARY KEY (load_id, day, symbol)
);

CREATE TABLE IF NOT EXISTS forward.daily_bars (
    load_id bigint NOT NULL REFERENCES forward.loads(load_id),
    day date NOT NULL,
    symbol text NOT NULL,
    feed text NOT NULL CHECK (feed = 'sip'),
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume bigint NOT NULL CHECK (volume >= 0),
    PRIMARY KEY (load_id, day, symbol, feed)
);
COMMENT ON COLUMN forward.daily_bars.open IS 'Raw unadjusted opening price.';
COMMENT ON COLUMN forward.daily_bars.high IS 'Raw unadjusted high price.';
COMMENT ON COLUMN forward.daily_bars.low IS 'Raw unadjusted low price.';
COMMENT ON COLUMN forward.daily_bars.close IS 'Raw unadjusted closing price.';
COMMENT ON COLUMN forward.daily_bars.feed IS 'Consolidated SIP market data feed.';

CREATE TABLE IF NOT EXISTS forward.gaps (
    load_id bigint NOT NULL REFERENCES forward.loads(load_id),
    day date NOT NULL,
    symbol text NOT NULL,
    reason text NOT NULL,
    PRIMARY KEY (load_id, day, symbol)
);

-- Status observations are appended; the latest observation represents current status.
CREATE TABLE IF NOT EXISTS forward.symbol_status (
    status_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol text NOT NULL,
    first_seen date NOT NULL,
    last_seen date NOT NULL,
    ended_on date,
    CHECK (last_seen >= first_seen),
    CHECK (ended_on IS NULL OR ended_on >= last_seen)
);

CREATE TABLE IF NOT EXISTS forward.shares_outstanding (
    load_id bigint NOT NULL REFERENCES forward.loads(load_id),
    day date NOT NULL,
    symbol text NOT NULL,
    shares bigint NOT NULL CHECK (shares >= 0),
    source text NOT NULL,
    PRIMARY KEY (load_id, day, symbol, source)
);

CREATE OR REPLACE FUNCTION forward.reject_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only table';
END;
$$;

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'loads', 'listing_snapshots', 'daily_bars', 'gaps', 'symbol_status', 'shares_outstanding'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgrelid = format('forward.%I', table_name)::regclass
              AND tgname = 'forward_no_change' AND NOT tgisinternal
        ) THEN
            EXECUTE format('CREATE TRIGGER forward_no_change BEFORE UPDATE OR DELETE ON forward.%I '
                'FOR EACH ROW EXECUTE FUNCTION forward.reject_change()', table_name);
        END IF;
        EXECUTE format('ALTER TABLE forward.%I ENABLE ALWAYS TRIGGER forward_no_change', table_name);
        IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgrelid = format('forward.%I', table_name)::regclass
              AND tgname = 'forward_no_truncate' AND NOT tgisinternal
        ) THEN
            EXECUTE format('CREATE TRIGGER forward_no_truncate BEFORE TRUNCATE ON forward.%I '
                'FOR EACH STATEMENT EXECUTE FUNCTION forward.reject_change()', table_name);
        END IF;
        EXECUTE format('ALTER TABLE forward.%I ENABLE ALWAYS TRIGGER forward_no_truncate', table_name);
    END LOOP;
END;
$$;
