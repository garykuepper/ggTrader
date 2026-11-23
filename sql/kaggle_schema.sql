CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS kaggle_crsp;
CREATE SCHEMA IF NOT EXISTS kaggle_chars;
CREATE TABLE IF NOT EXISTS ref.loads (
 load_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, dataset text, kaggle_slug text,
 file_path text, file_sha256 text, parser_version text, content_sha256 text,
 rows_parsed bigint, rows_accepted bigint, rows_quarantined bigint, loader_git_sha text,
 loaded_by text, started_at timestamptz, completed_at timestamptz,
 status text CHECK (status IN ('running','complete','failed','withdrawn')), checks jsonb
);
CREATE TABLE IF NOT EXISTS ref.load_chunks (load_id bigint, chunk_key text, rows bigint, committed_at timestamptz, PRIMARY KEY(load_id, chunk_key));
CREATE TABLE IF NOT EXISTS ref.quarantine (load_id bigint, chunk_key text, row_ref text, raw jsonb, reason text);
CREATE TABLE IF NOT EXISTS ref.releases (release_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, created_at timestamptz, settings jsonb, status text CHECK (status IN ('published','withdrawn')), yahoo_fingerprint jsonb);
ALTER TABLE ref.releases DROP CONSTRAINT IF EXISTS releases_status_check;
ALTER TABLE ref.releases ADD CONSTRAINT releases_status_check CHECK (status IN ('building','published','withdrawn'));
CREATE TABLE IF NOT EXISTS ref.release_loads (release_id bigint, load_id bigint, role text);
CREATE TABLE IF NOT EXISTS ref.symbol_map (permno bigint, symbol text, valid_from date, valid_to date, source text, release_id bigint);
CREATE INDEX IF NOT EXISTS symbol_map_release_symbol_dates_idx ON ref.symbol_map (release_id, symbol, valid_from, valid_to);
CREATE TABLE IF NOT EXISTS ref.splice_links (release_id bigint, permno bigint, symbol text, first_yf_date date, common_day date, join_returns double precision[], link_day_returns_hash text, agreement jsonb, PRIMARY KEY(release_id, permno));
CREATE TABLE IF NOT EXISTS kaggle_crsp.daily (load_id bigint, permno bigint, date date, shrcd int, exchcd int, prc double precision, ret double precision, shrout double precision, vwretd double precision, year int, month int, PRIMARY KEY(load_id, permno, date));
CREATE TABLE IF NOT EXISTS kaggle_chars.mcgill (
    load_id bigint,
    date date,
    ret_eom date,
    permno bigint,
    shrcd integer,
    exchcd integer,
    mspread double precision,
    year integer,
    month integer,
    size_port text,
    rf double precision,
    stock_exret double precision,
    stock_ticker text,
    cusip text,
    comp_name text,
    eps_medest double precision,
    eps_meanest double precision,
    eps_stdevest double precision,
    eps_actual double precision,
    prc double precision,
    market_equity double precision,
    div12m_me double precision,
    chcsho_12m double precision,
    eqnpo_12m double precision,
    ret_1_0 double precision,
    ret_3_1 double precision,
    ret_6_1 double precision,
    ret_9_1 double precision,
    ret_12_1 double precision,
    ret_12_7 double precision,
    ret_60_12 double precision,
    seas_1_1an double precision,
    seas_1_1na double precision,
    seas_2_5an double precision,
    seas_2_5na double precision,
    at_gr1 double precision,
    sale_gr1 double precision,
    capx_gr1 double precision,
    inv_gr1 double precision,
    debt_gr3 double precision,
    sale_gr3 double precision,
    capx_gr3 double precision,
    inv_gr1a double precision,
    lti_gr1a double precision,
    sti_gr1a double precision,
    coa_gr1a double precision,
    col_gr1a double precision,
    cowc_gr1a double precision,
    ncoa_gr1a double precision,
    ncol_gr1a double precision,
    nncoa_gr1a double precision,
    fnl_gr1a double precision,
    nfna_gr1a double precision,
    tax_gr1a double precision,
    be_gr1a double precision,
    ebit_sale double precision,
    gp_at double precision,
    cop_at double precision,
    ope_be double precision,
    ni_be double precision,
    ebit_bev double precision,
    netis_at double precision,
    eqnetis_at double precision,
    dbnetis_at double precision,
    oaccruals_at double precision,
    oaccruals_ni double precision,
    taccruals_at double precision,
    taccruals_ni double precision,
    noa_at double precision,
    opex_at double precision,
    at_turnover double precision,
    sale_bev double precision,
    rd_sale double precision,
    cash_at double precision,
    sale_emp_gr1 double precision,
    emp_gr1 double precision,
    ni_inc8q double precision,
    noa_gr1a double precision,
    ppeinv_gr1a double precision,
    lnoa_gr1a double precision,
    capx_gr2 double precision,
    saleq_gr1 double precision,
    niq_be double precision,
    niq_at double precision,
    niq_be_chg1 double precision,
    niq_at_chg1 double precision,
    rd5_at double precision,
    dsale_dinv double precision,
    dsale_drec double precision,
    dgp_dsale double precision,
    dsale_dsga double precision,
    saleq_su double precision,
    niq_su double precision,
    capex_abn double precision,
    op_atl1 double precision,
    gp_atl1 double precision,
    ope_bel1 double precision,
    cop_atl1 double precision,
    pi_nix double precision,
    ocf_at double precision,
    op_at double precision,
    ocf_at_chg1 double precision,
    at_be double precision,
    ocfq_saleq_std double precision,
    tangibility double precision,
    earnings_variability double precision,
    aliq_at double precision,
    f_score double precision,
    o_score double precision,
    z_score double precision,
    intrinsic_value double precision,
    kz_index double precision,
    ni_ar1 double precision,
    ni_ivol double precision,
    at_me double precision,
    be_me double precision,
    debt_me double precision,
    netdebt_me double precision,
    sale_me double precision,
    ni_me double precision,
    ocf_me double precision,
    fcf_me double precision,
    eqpo_me double precision,
    eqnpo_me double precision,
    rd_me double precision,
    bev_mev double precision,
    ebitda_mev double precision,
    aliq_mat double precision,
    eq_dur double precision,
    beta_60m double precision,
    resff3_12_1 double precision,
    resff3_6_1 double precision,
    mispricing_mgmt double precision,
    mispricing_perf double precision,
    ivol_capm_21d double precision,
    iskew_capm_21d double precision,
    coskew_21d double precision,
    beta_dimson_21d double precision,
    ivol_ff3_21d double precision,
    iskew_ff3_21d double precision,
    ivol_hxz4_21d double precision,
    iskew_hxz4_21d double precision,
    rmax5_21d double precision,
    rmax1_21d double precision,
    rvol_21d double precision,
    rskew_21d double precision,
    zero_trades_21d double precision,
    dolvol_126d double precision,
    dolvol_var_126d double precision,
    turnover_126d double precision,
    turnover_var_126d double precision,
    zero_trades_126d double precision,
    zero_trades_252d double precision,
    ami_126d double precision,
    ivol_capm_252d double precision,
    prc_highprc_252d double precision,
    betadown_252d double precision,
    bidaskhl_21d double precision,
    corr_1260d double precision,
    betabab_1260d double precision,
    rmax5_rvol_21d double precision,
    age double precision,
    qmj double precision,
    qmj_prof double precision,
    qmj_growth double precision,
    qmj_safety double precision
);
CREATE TABLE IF NOT EXISTS market.daily_bars (load_id bigint, permno bigint, spell int, date date, close double precision, abs_prc double precision, midpoint_flag boolean, gap_return_approx boolean, final_flag boolean, PRIMARY KEY(load_id, permno, date));
CREATE TABLE IF NOT EXISTS market.stock_spans (release_id bigint, symbol text, permno bigint, spell int, first_date date, last_date date, linked boolean, kind text CHECK (kind IN ('linked_yahoo','yahoo','placeholder','spell')), crsp_until date, first_yf_date date, possible_double boolean, PRIMARY KEY(release_id, symbol));
ALTER TABLE ref.splice_links ADD COLUMN IF NOT EXISTS common_day date;
ALTER TABLE ref.splice_links ADD COLUMN IF NOT EXISTS join_returns double precision[];
ALTER TABLE market.stock_spans ADD COLUMN IF NOT EXISTS kind text CHECK (kind IN ('linked_yahoo','yahoo','placeholder','spell'));
ALTER TABLE market.stock_spans ADD COLUMN IF NOT EXISTS crsp_until date;
ALTER TABLE market.stock_spans ADD COLUMN IF NOT EXISTS first_yf_date date;
ALTER TABLE market.stock_spans ADD COLUMN IF NOT EXISTS possible_double boolean;
SELECT create_hypertable('kaggle_crsp.daily', 'date', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
SELECT create_hypertable('market.daily_bars', 'date', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
ALTER TABLE kaggle_crsp.daily SET (timescaledb.compress, timescaledb.compress_segmentby = 'load_id, permno');
ALTER TABLE market.daily_bars SET (timescaledb.compress, timescaledb.compress_segmentby = 'load_id, permno');
CREATE OR REPLACE FUNCTION ref.guard_load() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'loads cannot be deleted'; END IF;
 IF OLD.status = 'complete' AND NEW.status = 'withdrawn' AND EXISTS (
   SELECT 1 FROM ref.release_loads rl JOIN ref.releases r USING (release_id)
   WHERE rl.load_id = OLD.load_id AND r.status = 'published'
 ) THEN RAISE EXCEPTION 'load belongs to a published release'; END IF;
 IF OLD.status IN ('complete', 'withdrawn', 'failed') AND NOT
   (OLD.status = 'complete' AND NEW.status = 'withdrawn' AND
    (to_jsonb(NEW) - 'status') = (to_jsonb(OLD) - 'status')) THEN
   RAISE EXCEPTION 'completed loads are immutable';
 END IF;
 IF OLD.status = 'running' AND (NEW.status IS NULL OR NEW.status NOT IN ('running', 'complete', 'failed')) THEN
   RAISE EXCEPTION 'invalid load status transition';
 END IF;
 RETURN NEW;
END $$;
CREATE OR REPLACE FUNCTION ref.guard_release() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'releases cannot be deleted'; END IF;
 IF (((OLD.status = 'published' AND NEW.status = 'withdrawn') OR
      (OLD.status = 'building' AND NEW.status = 'published')) AND
     (to_jsonb(NEW) - 'status') = (to_jsonb(OLD) - 'status')) IS NOT TRUE THEN
  RAISE EXCEPTION 'releases are immutable';
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS frozen ON ref.releases;
DROP TRIGGER IF EXISTS guard_release ON ref.releases;
CREATE TRIGGER guard_release BEFORE UPDATE OR DELETE ON ref.releases FOR EACH ROW EXECUTE FUNCTION ref.guard_release();
ALTER TABLE ref.releases ENABLE ALWAYS TRIGGER guard_release;
CREATE OR REPLACE FUNCTION ref.reject_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'append-only table'; END $$;
DROP TRIGGER IF EXISTS guard_load ON ref.loads;
CREATE TRIGGER guard_load BEFORE UPDATE OR DELETE ON ref.loads FOR EACH ROW EXECUTE FUNCTION ref.guard_load();
ALTER TABLE ref.loads ENABLE ALWAYS TRIGGER guard_load;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['release_loads','symbol_map','splice_links'] LOOP
  EXECUTE format('DROP TRIGGER IF EXISTS frozen ON ref.%I', t);
  EXECUTE format('CREATE TRIGGER frozen BEFORE UPDATE OR DELETE ON ref.%I FOR EACH ROW EXECUTE FUNCTION ref.reject_change()', t);
  EXECUTE format('ALTER TABLE ref.%I ENABLE ALWAYS TRIGGER frozen', t);
 END LOOP;
 FOREACH t IN ARRAY ARRAY['loads','releases','release_loads','symbol_map','splice_links'] LOOP
  EXECUTE format('DROP TRIGGER IF EXISTS no_truncate ON ref.%I', t);
  EXECUTE format('CREATE TRIGGER no_truncate BEFORE TRUNCATE ON ref.%I FOR EACH STATEMENT EXECUTE FUNCTION ref.reject_change()', t);
  EXECUTE format('ALTER TABLE ref.%I ENABLE ALWAYS TRIGGER no_truncate', t);
 END LOOP;
END $$;

DROP TRIGGER IF EXISTS frozen ON market.stock_spans;
CREATE TRIGGER frozen BEFORE UPDATE OR DELETE ON market.stock_spans FOR EACH ROW EXECUTE FUNCTION ref.reject_change();
ALTER TABLE market.stock_spans ENABLE ALWAYS TRIGGER frozen;
DROP TRIGGER IF EXISTS no_truncate ON market.stock_spans;
CREATE TRIGGER no_truncate BEFORE TRUNCATE ON market.stock_spans FOR EACH STATEMENT EXECUTE FUNCTION ref.reject_change();
ALTER TABLE market.stock_spans ENABLE ALWAYS TRIGGER no_truncate;

CREATE OR REPLACE FUNCTION kaggle_crsp.daily_of(p_load_id bigint)
RETURNS SETOF kaggle_crsp.daily LANGUAGE plpgsql STABLE AS $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
  RAISE EXCEPTION 'load % is not complete', p_load_id;
 END IF;
 RETURN QUERY SELECT * FROM kaggle_crsp.daily WHERE load_id = p_load_id;
END $$;

CREATE OR REPLACE FUNCTION kaggle_chars.mcgill_of(p_load_id bigint)
RETURNS SETOF kaggle_chars.mcgill LANGUAGE plpgsql STABLE AS $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM ref.loads WHERE load_id = p_load_id AND status = 'complete') THEN
  RAISE EXCEPTION 'load % is not complete', p_load_id;
 END IF;
 RETURN QUERY SELECT * FROM kaggle_chars.mcgill WHERE load_id = p_load_id;
END $$;


CREATE OR REPLACE FUNCTION ref.guard_release_insert() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 PERFORM 1 FROM ref.releases WHERE release_id = NEW.release_id AND status = 'building' FOR SHARE;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'parent release must exist and be building';
 END IF;
 RETURN NEW;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS release_loads_role_unique ON ref.release_loads(release_id, role);
DO $$ DECLARE item text; BEGIN
 FOREACH item IN ARRAY ARRAY['ref.release_loads','ref.symbol_map','ref.splice_links','market.stock_spans'] LOOP
  EXECUTE format('DROP TRIGGER IF EXISTS guard_release_insert ON %s', item);
  EXECUTE format('CREATE TRIGGER guard_release_insert BEFORE INSERT ON %s FOR EACH ROW EXECUTE FUNCTION ref.guard_release_insert()', item);
  EXECUTE format('ALTER TABLE %s ENABLE ALWAYS TRIGGER guard_release_insert', item);
 END LOOP;
END $$;

CREATE OR REPLACE FUNCTION ref.require_available_release(p_release_id bigint) RETURNS void
LANGUAGE plpgsql STABLE AS $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM ref.releases WHERE release_id=p_release_id AND status='published')
    OR NOT EXISTS (SELECT 1 FROM ref.release_loads WHERE release_id=p_release_id AND role='crsp-bars')
    OR EXISTS (SELECT 1 FROM ref.release_loads rl LEFT JOIN ref.loads l ON l.load_id=rl.load_id
               WHERE rl.release_id=p_release_id AND l.status IS DISTINCT FROM 'complete') THEN
    RAISE EXCEPTION 'release % is unavailable', p_release_id;
 END IF;
END $$;
DROP FUNCTION IF EXISTS market.ohlcv_daily_combined(bigint, text[], date, date);
DROP FUNCTION IF EXISTS market._combined_linked(bigint, text[]);

SET LOCAL check_function_bodies = off;
CREATE OR REPLACE FUNCTION market._combined_linked(p_release_id bigint, p_symbols text[])
RETURNS TABLE (symbol text, permno bigint, spell int, load_id bigint, crsp_until date, first_yf_date date, possible_double boolean, scale double precision, drift boolean)
LANGUAGE plpgsql STABLE AS $$
BEGIN
PERFORM ref.require_available_release(p_release_id);
RETURN QUERY SELECT s.symbol,s.permno,s.spell,l.load_id,s.crsp_until,s.first_yf_date,s.possible_double,
       (SELECT y.close FROM public.ohlcv y WHERE y.symbol=s.symbol AND y.symbol = ANY(p_symbols) AND y.venue='yfinance'
           AND y.interval='1d' AND y.timestamp=sp.common_day::timestamp) /
       NULLIF((SELECT b.close FROM market.daily_bars b WHERE b.load_id=l.load_id
           AND b.permno=s.permno AND b.date=sp.common_day),0),
       EXISTS (SELECT 1 FROM public.ohlcv o WHERE o.symbol=s.symbol AND o.symbol = ANY(p_symbols) AND o.venue='yfinance' AND o.interval='1d' AND o.timestamp < s.first_yf_date::timestamp)
       OR EXISTS (SELECT 1 FROM generate_subscripts(sp.join_returns,1) n(i)
          WHERE c.closes[n.i] IS NULL OR c.closes[n.i+1] IS NULL
             OR c.closes[n.i] = 0
             OR abs(c.closes[n.i+1] / NULLIF(c.closes[n.i],0) - 1 - sp.join_returns[n.i]) > 1e-6)

FROM market.stock_spans s
JOIN ref.release_loads l ON l.release_id=s.release_id AND l.role='crsp-bars'
JOIN ref.splice_links sp ON sp.release_id=s.release_id AND sp.permno=s.permno
CROSS JOIN LATERAL (
    SELECT array_agg(q.close ORDER BY q.timestamp) AS closes FROM (
        SELECT x.close, x.timestamp FROM public.ohlcv x
        WHERE x.symbol=s.symbol AND x.symbol = ANY(p_symbols) AND x.venue='yfinance' AND x.interval='1d'
          AND x.timestamp >= s.first_yf_date::timestamp
        ORDER BY x.timestamp LIMIT COALESCE(array_length(sp.join_returns,1),0)+1
    ) q
) c
WHERE s.release_id=p_release_id AND s.symbol=ANY(p_symbols) AND s.kind='linked_yahoo';
END;
$$;
SET LOCAL check_function_bodies = on;

CREATE OR REPLACE FUNCTION market._raise_input_missing(p_symbol text)
RETURNS double precision LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'release input missing for %', p_symbol;
END;
$$;

CREATE OR REPLACE FUNCTION market.ohlcv_daily_combined(
    p_release_id bigint, p_symbols text[], p_start date, p_end date
) RETURNS TABLE (
    ts timestamp without time zone, symbol text, open double precision,
    high double precision, low double precision, close double precision,
    volume double precision, source text, abs_prc double precision,
    midpoint_flag boolean, possible_double boolean, drift boolean
) LANGUAGE plpgsql STABLE AS $$
-- Yahoo reads stop at the date recorded by the release fingerprint.
DECLARE missing_symbol text; yahoo_end date;
BEGIN
    PERFORM ref.require_available_release(p_release_id);
    SELECT left(yahoo_fingerprint->>'newest_yahoo_timestamp',10)::date INTO yahoo_end
    FROM ref.releases WHERE release_id=p_release_id;
    SELECT x.symbol INTO missing_symbol FROM market._combined_linked(p_release_id,p_symbols) x WHERE x.scale IS NULL LIMIT 1;
    IF FOUND THEN RAISE EXCEPTION 'release input missing for %', missing_symbol; END IF;
    RETURN QUERY
    WITH linked AS MATERIALIZED (SELECT * FROM market._combined_linked(p_release_id, p_symbols)),
    requested AS (SELECT DISTINCT unnest(p_symbols) AS symbol)
    SELECT y."timestamp"::date::timestamp, q.symbol, y.open, y.high, y.low, y.close, y.volume,
           'yfinance'::text, NULL::double precision, NULL::boolean,
           COALESCE(s.possible_double, false), false
    FROM requested q
    LEFT JOIN market.stock_spans s ON s.release_id = p_release_id AND s.symbol = q.symbol
    JOIN public.ohlcv y ON y.symbol = q.symbol AND y.venue = 'yfinance' AND y.interval = '1d'
    WHERE y.symbol = ANY(p_symbols) AND y.timestamp < (yahoo_end + 1)::timestamp
      AND (s.symbol IS NULL OR s.kind = 'yahoo')
      AND (p_start IS NULL OR y."timestamp" >= p_start::timestamp)
      AND (p_end IS NULL OR y."timestamp" < (p_end + 1)::timestamp)
    UNION ALL
    SELECT b.date::timestamp, q.symbol, NULL::double precision, NULL::double precision,
           NULL::double precision, b.close, NULL::double precision, 'crsp'::text,
           b.abs_prc, b.midpoint_flag, s.possible_double, false
    FROM requested q
    JOIN market.stock_spans s ON s.release_id = p_release_id AND s.symbol = q.symbol
                             AND s.kind IN ('placeholder', 'spell')
    JOIN ref.release_loads l ON l.release_id = p_release_id AND l.role = 'crsp-bars'
    JOIN LATERAL (SELECT b.* FROM market.daily_bars b
        WHERE b.load_id = l.load_id AND b.permno = s.permno AND b.spell = s.spell
          AND (p_start IS NULL OR b.date >= p_start) AND (p_end IS NULL OR b.date <= p_end)
        OFFSET 0) b ON true
    UNION ALL
    SELECT b.date::timestamp,x.symbol,NULL::double precision,NULL::double precision,NULL::double precision,b.close*COALESCE(x.scale, market._raise_input_missing(x.symbol)),NULL::double precision,'crsp'::text,b.abs_prc,b.midpoint_flag,x.possible_double,x.drift
    FROM linked x
    JOIN LATERAL (SELECT b.* FROM market.daily_bars b
        WHERE b.load_id=x.load_id AND b.permno=x.permno AND b.spell=x.spell
          AND b.date<x.crsp_until AND (p_start IS NULL OR b.date>=p_start)
          AND (p_end IS NULL OR b.date<=p_end) OFFSET 0) b ON true
    UNION ALL
    SELECT y.timestamp,x.symbol,y.open,y.high,y.low,y.close,y.volume,'yfinance'::text,NULL::double precision,NULL::boolean,x.possible_double,x.drift
    FROM linked x
    JOIN public.ohlcv y ON y.symbol=x.symbol AND y.venue='yfinance' AND y.interval='1d'
    WHERE y.symbol = ANY(p_symbols) AND y.timestamp < (yahoo_end + 1)::timestamp
      AND y.timestamp>=x.first_yf_date::timestamp AND (p_start IS NULL OR y.timestamp>=p_start::timestamp) AND (p_end IS NULL OR y.timestamp<(p_end+1)::timestamp);
END;
$$;
