"""PostgreSQL database manager for storing trading run results."""

import json
import math
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Connection, create_engine, text

from ggTrader.utils.config import get_db_connection_string


class ResultDBManager:
    """Manages the PostgreSQL database for storing trading run results."""

    def __init__(self, connection_string: Optional[str] = None, **_legacy_kwargs: Any) -> None:
        # Legacy ``log_path`` kwarg is accepted and ignored — runs_log.csv was
        # superseded by the runs table; callers can remove it at their leisure.
        self.connection_string = connection_string or get_db_connection_string()

        self.engine = create_engine(self.connection_string)
        self._init_db()

    # =========================================================================
    # Database Initialization & Migrations
    # =========================================================================

    def _init_db(self) -> None:
        """Initializes the PostgreSQL tables and updates schema if needed."""
        with self.engine.begin() as conn:
            self._create_core_tables(conn)
            self._run_schema_migrations(conn)
            self._create_aux_tables(conn)

    def _create_core_tables(self, conn: Connection) -> None:
        """Creates the primary runs table."""
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    run_type VARCHAR,
                    timestamp TIMESTAMPTZ,
                    script_name VARCHAR,
                    parameters JSONB,
                    metadata JSONB
                );
                """
            )
        )

    def _run_schema_migrations(self, conn: Connection) -> None:
        """Applies non-destructive column additions to existing tables."""
        new_columns = [
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS sharpe DOUBLE PRECISION;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS sortino DOUBLE PRECISION;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_profit DOUBLE PRECISION;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS start_date TIMESTAMPTZ;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS end_date TIMESTAMPTZ;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS interval VARCHAR;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS pipeline_stage VARCHAR;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS asset_class VARCHAR;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS strategy_params JSONB;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS phase_stats JSONB;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS status VARCHAR;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_dir VARCHAR;",
            "CREATE INDEX IF NOT EXISTS runs_research_lookup_idx ON runs (run_type, asset_class, timestamp DESC);",
            # Live position-close fields — null for non-live (backtest/research) rows.
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS amount DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS gross_pnl DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee_entry DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee_exit DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee_total DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS hold_duration_hours DOUBLE PRECISION;",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_reason VARCHAR;",
        ]
        for stmt in new_columns:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                print(f"Schema migration warning ({stmt}): {e}")

    def _create_aux_tables(self, conn: Connection) -> None:
        """Creates supplementary tables for metrics, trades, and WFO data."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS wfo_windows (
                run_id VARCHAR, window_id INTEGER, test_start TIMESTAMPTZ,
                test_end TIMESTAMPTZ, start_capital DOUBLE PRECISION,
                end_capital DOUBLE PRECISION, profit DOUBLE PRECISION,
                return_pct DOUBLE PRECISION, sharpe DOUBLE PRECISION,
                sortino DOUBLE PRECISION, params JSONB,
                PRIMARY KEY (run_id, window_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                run_id VARCHAR, metric_name VARCHAR, metric_value DOUBLE PRECISION,
                PRIMARY KEY (run_id, metric_name)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS equity_curves (
                run_id VARCHAR, timestamp TIMESTAMPTZ, equity_value DOUBLE PRECISION,
                PRIMARY KEY (run_id, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                run_id VARCHAR, symbol VARCHAR, entry_time TIMESTAMPTZ,
                exit_time TIMESTAMPTZ, entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION, profit DOUBLE PRECISION,
                profit_pct DOUBLE PRECISION, status VARCHAR,
                PRIMARY KEY (run_id, symbol, entry_time)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS orders (
                run_id VARCHAR, timestamp TIMESTAMPTZ, symbol VARCHAR,
                side VARCHAR, order_id VARCHAR, price DOUBLE PRECISION,
                amount DOUBLE PRECISION, amount_usd DOUBLE PRECISION,
                fee DOUBLE PRECISION, fee_currency VARCHAR,
                PRIMARY KEY (run_id, order_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS study_results (
                study_hash VARCHAR PRIMARY KEY, params JSONB, result JSONB, timestamp TIMESTAMPTZ
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS system_state (
                key VARCHAR PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS wfo_cache (
                cache_key VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                strategy_name VARCHAR,
                exit_name VARCHAR,
                cache_version INTEGER,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS wfo_cache_symbol_strategy_idx ON wfo_cache (symbol, strategy_name);",
            """
            CREATE TABLE IF NOT EXISTS kraken_ledger (
                ledger_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                type VARCHAR,
                currency VARCHAR,
                amount DOUBLE PRECISION,
                signed_amount DOUBLE PRECISION,
                last_fetched_at TIMESTAMPTZ DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS kraken_ledger_ts_idx ON kraken_ledger (timestamp DESC);",
            """
            CREATE TABLE IF NOT EXISTS universe_cache (
                asset_class VARCHAR NOT NULL,
                snapshot_date DATE NOT NULL,
                symbols JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (asset_class, snapshot_date)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_balance_snapshots (
                asset_class VARCHAR NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                total_usd DOUBLE PRECISION,
                free_usd DOUBLE PRECISION,
                positions_usd DOUBLE PRECISION,
                num_positions INTEGER,
                PRIMARY KEY (asset_class, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_positions_snapshot (
                asset_class VARCHAR NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                positions JSONB NOT NULL,
                circuit_breaker_triggered BOOLEAN,
                PRIMARY KEY (asset_class, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS fear_greed_index (
                snapshot_date DATE PRIMARY KEY,
                value INTEGER NOT NULL,
                classification VARCHAR NOT NULL,
                inserted_at TIMESTAMPTZ DEFAULT now()
            );
            """,
        ]
        for query in queries:
            conn.execute(text(query))
        # Try to convert live_balance_snapshots into a hypertable; tolerate failure
        # if TimescaleDB extension isn't available.
        try:
            conn.execute(
                text(
                    "SELECT create_hypertable('live_balance_snapshots', 'timestamp',"
                    " if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
        except Exception as e:
            print(f"Hypertable warning (live_balance_snapshots): {e}")
        try:
            conn.execute(
                text(
                    "SELECT create_hypertable('live_positions_snapshot', 'timestamp',"
                    " if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
        except Exception as e:
            print(f"Hypertable warning (live_positions_snapshot): {e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _safe_json_dumps(self, obj: Any) -> str:
        """JSON dumps that ensures NaNs and Infs are converted to null."""

        def clean_obj(v: Any) -> Any:
            if isinstance(v, float):
                if v != v or v == float("inf") or v == float("-inf"):
                    return None
            elif isinstance(v, dict):
                return {k: clean_obj(v2) for k, v2 in v.items()}
            elif isinstance(v, (list, tuple)):
                return [clean_obj(v2) for v2 in v]
            return v

        return json.dumps(clean_obj(obj))

    # =========================================================================
    # system_state — generic key/value store for small persistence flags
    # =========================================================================

    def get_state(self, key: str, default: Any = None) -> Any:
        """Read a value from system_state. Returns ``default`` if absent."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT value FROM system_state WHERE key = :k"),
                    {"k": key},
                ).fetchone()
        except Exception as e:
            print(f"WARNING: get_state({key!r}) failed: {e}")
            return default
        if row is None:
            return default
        # JSONB comes back as a parsed object via psycopg2 already.
        return row[0]

    def set_state(self, key: str, value: Any) -> None:
        """Upsert a value into system_state."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO system_state (key, value, updated_at)
                        VALUES (:k, CAST(:v AS JSONB), now())
                        ON CONFLICT (key) DO UPDATE
                          SET value = EXCLUDED.value,
                              updated_at = now()
                        """
                    ),
                    {"k": key, "v": self._safe_json_dumps(value)},
                )
        except Exception as e:
            print(f"WARNING: set_state({key!r}) failed: {e}")

    def _parse_trade_row(self, row: pd.Series, run_id: str) -> Optional[Dict[str, Any]]:
        """Extracts and sanitizes a single trade row for DB insertion."""
        mapping = {
            "entry_time": ["entry_time", "entry_date"],
            "exit_time": ["exit_time", "exit_date"],
        }
        try:
            entry_t = next(
                (row[c] for c in mapping["entry_time"] if c in row and pd.notna(row[c])), None
            )
            exit_t = next(
                (row[c] for c in mapping["exit_time"] if c in row and pd.notna(row[c])), None
            )

            return {
                "run_id": run_id,
                "symbol": row.get("symbol", "UNKNOWN"),
                "entry_time": entry_t,
                "exit_time": exit_t,
                "entry_price": float(row.get("entry_price", 0.0)),
                "exit_price": float(row.get("exit_price", 0.0)),
                "profit": float(row.get("profit", 0.0)),
                "profit_pct": float(row.get("profit_pct", 0.0)),
                "status": row.get("status", "closed"),
            }
        except Exception as e:
            print(f"Error processing trade row: {e}")
            return None

    # =========================================================================
    # Data Ingestion Methods
    # =========================================================================

    def add_run(
        self,
        run_id: str,
        run_type: str,
        script_name: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        pipeline_stage: Optional[str] = None,
        asset_class: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        phase_stats: Optional[Dict[str, Any]] = None,
        run_dir: Optional[str] = None,
        status: str = "success",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Upsert a primary run entry. The run_dir/strategy_params/phase_stats
        columns let the live trader and the report builder load everything
        from the DB without touching the on-disk artifacts."""
        ts = timestamp or datetime.now()
        meta = metadata or {}
        metr = metrics or {}

        sharpe = metr.get("oos_sharpe", metr.get("sharpe"))
        sortino = metr.get("sortino")
        total_profit = metr.get("total_profit")

        start_date = meta.get("START_DATE") or meta.get("start_date")
        end_date = meta.get("END_DATE") or meta.get("end_date")
        interval = meta.get("INTERVAL") or meta.get("interval")
        ac = asset_class or meta.get("ASSET_CLASS") or meta.get("asset_class")

        query = text(
            """
            INSERT INTO runs (
                run_id, run_type, timestamp, script_name, parameters, metadata,
                sharpe, sortino, total_profit, start_date, end_date, interval,
                pipeline_stage, asset_class, strategy_params, phase_stats,
                status, run_dir
            )
            VALUES (
                :rid, :rtype, :ts, :sname, CAST(:params AS JSONB), CAST(:meta AS JSONB),
                :sr, :sort, :prof, :sdate, :edate, :inter, :stage, :ac,
                CAST(:sp AS JSONB), CAST(:ps AS JSONB), :status, :rd
            )
            ON CONFLICT (run_id) DO UPDATE SET
                run_type = EXCLUDED.run_type,
                timestamp = EXCLUDED.timestamp,
                script_name = EXCLUDED.script_name,
                parameters = EXCLUDED.parameters,
                metadata = EXCLUDED.metadata,
                sharpe = EXCLUDED.sharpe,
                sortino = EXCLUDED.sortino,
                total_profit = EXCLUDED.total_profit,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                interval = EXCLUDED.interval,
                pipeline_stage = EXCLUDED.pipeline_stage,
                asset_class = COALESCE(EXCLUDED.asset_class, runs.asset_class),
                strategy_params = COALESCE(EXCLUDED.strategy_params, runs.strategy_params),
                phase_stats = COALESCE(EXCLUDED.phase_stats, runs.phase_stats),
                status = EXCLUDED.status,
                run_dir = COALESCE(EXCLUDED.run_dir, runs.run_dir)
            """
        )

        bind_params = {
            "rid": run_id,
            "rtype": run_type,
            "ts": ts,
            "sname": script_name,
            "params": self._safe_json_dumps(parameters),
            "meta": self._safe_json_dumps(meta),
            "sr": sharpe,
            "sort": sortino,
            "prof": total_profit,
            "sdate": start_date,
            "edate": end_date,
            "inter": interval,
            "stage": pipeline_stage,
            "ac": ac,
            "sp": self._safe_json_dumps(strategy_params) if strategy_params is not None else None,
            "ps": self._safe_json_dumps(phase_stats) if phase_stats is not None else None,
            "status": status,
            "rd": run_dir,
        }

        try:
            with self.engine.begin() as conn:
                conn.execute(query, bind_params)
        except Exception as e:
            print(f"WARNING: Could not save run results to DB: {e}")

    def add_wfo_results(self, run_id: str, df_res: pd.DataFrame) -> None:
        """Inserts WFO window results into the database."""
        if df_res.empty:
            return

        records = []
        for i, row in df_res.iterrows():
            records.append(
                {
                    "run_id": run_id,
                    "window_id": i,
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "start_capital": row["start_capital"],
                    "end_capital": row["end_capital"],
                    "profit": row["profit"],
                    "return_pct": row["return_pct"],
                    "sharpe": row.get("oos_sharpe", row.get("sharpe")),
                    "sortino": row["sortino"],
                    "params": self._safe_json_dumps(row["params"]),
                }
            )

        pd.DataFrame(records).to_sql(
            "wfo_windows", self.engine, if_exists="append", index=False, method="multi"
        )

    def add_metrics(self, run_id: str, metrics_dict: Dict[str, Any]) -> None:
        """Adds multiple numerical metrics for a specific run."""
        with self.engine.begin() as conn:
            for name, value in metrics_dict.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (np.bool_,)):
                    continue
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                fv = float(value)
                if math.isnan(fv) or math.isinf(fv):
                    continue
                stmt = text(
                    """
                    INSERT INTO performance_metrics (run_id, metric_name, metric_value)
                    VALUES (:run_id, :metric_name, :metric_value)
                    ON CONFLICT (run_id, metric_name)
                    DO UPDATE SET metric_value = EXCLUDED.metric_value
                    """
                )
                conn.execute(stmt, {"run_id": run_id, "metric_name": name, "metric_value": fv})

    def add_equity_curve(self, run_id: str, equity_series: pd.Series) -> None:
        """Inserts equity curve data into the database with conflict handling."""
        if equity_series.empty:
            return

        query = text(
            """
            INSERT INTO equity_curves (run_id, timestamp, equity_value)
            VALUES (:run_id, :timestamp, :equity_value)
            ON CONFLICT (run_id, timestamp) DO NOTHING
            """
        )

        records = [
            {
                "run_id": run_id,
                "timestamp": ts,
                "equity_value": float(val),
            }
            for ts, val in equity_series.items()
        ]

        try:
            with self.engine.begin() as conn:
                conn.execute(query, records)
        except Exception as e:
            print(f"Error inserting equity curve: {e}")

    def add_trades(self, run_id: str, df_trades: pd.DataFrame) -> None:
        """Parses and inserts trade records into the database.

        Optional live-only columns (``amount``, ``gross_pnl``, ``fee_entry``,
        ``fee_exit``, ``fee_total``, ``hold_duration_hours``, ``exit_reason``)
        are persisted when present in the DataFrame and left NULL otherwise.
        """
        if df_trades.empty:
            return

        records = [
            record
            for _, row in df_trades.iterrows()
            if (record := self._parse_trade_row(row, run_id)) is not None
        ]

        if not records:
            return

        # Augment each record with the optional live-only columns. Using a
        # wider INSERT is harmless for backtest rows where these are NULL.
        live_keys = (
            "amount",
            "gross_pnl",
            "fee_entry",
            "fee_exit",
            "fee_total",
            "hold_duration_hours",
            "exit_reason",
        )
        for rec, (_, row) in zip(records, df_trades.iterrows()):
            for k in live_keys:
                if k in row and pd.notna(row[k]):
                    rec[k] = row[k]
                else:
                    rec.setdefault(k, None)

        query = text(
            """
            INSERT INTO trades (
                run_id, symbol, entry_time, exit_time, entry_price, exit_price, profit,
                profit_pct, status, amount, gross_pnl, fee_entry, fee_exit, fee_total,
                hold_duration_hours, exit_reason
            )
            VALUES (
                :run_id, :symbol, :entry_time, :exit_time, :entry_price, :exit_price,
                :profit, :profit_pct, :status, :amount, :gross_pnl, :fee_entry, :fee_exit,
                :fee_total, :hold_duration_hours, :exit_reason
            )
            ON CONFLICT (run_id, symbol, entry_time) DO UPDATE SET
                exit_time = EXCLUDED.exit_time, entry_price = EXCLUDED.entry_price,
                exit_price = EXCLUDED.exit_price, profit = EXCLUDED.profit,
                profit_pct = EXCLUDED.profit_pct, status = EXCLUDED.status,
                amount = EXCLUDED.amount, gross_pnl = EXCLUDED.gross_pnl,
                fee_entry = EXCLUDED.fee_entry, fee_exit = EXCLUDED.fee_exit,
                fee_total = EXCLUDED.fee_total, hold_duration_hours = EXCLUDED.hold_duration_hours,
                exit_reason = EXCLUDED.exit_reason
            """
        )

        with self.engine.begin() as conn:
            conn.execute(query, records)

    def add_order(
        self,
        run_id: str,
        symbol: str,
        side: str,
        order_id: str,
        price: float,
        amount: float,
        amount_usd: float,
        fee: float = 0.0,
        fee_currency: str = "USD",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Adds a raw order fill to the database."""
        ts = timestamp or datetime.now()
        query = text(
            """
            INSERT INTO orders (
                run_id, timestamp, symbol, side, order_id, price, amount,
                amount_usd, fee, fee_currency
            )
            VALUES (
                :run_id, :timestamp, :symbol, :side, :order_id, :price, :amount,
                :amount_usd, :fee, :fee_currency
            )
            ON CONFLICT (run_id, order_id) DO NOTHING
            """
        )
        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "run_id": run_id,
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": side,
                    "order_id": order_id,
                    "price": price,
                    "amount": amount,
                    "amount_usd": amount_usd,
                    "fee": fee,
                    "fee_currency": fee_currency,
                },
            )

    def check_existing_study(self, study_hash: str) -> Optional[Any]:
        """Checks if a study with the given hash already exists."""
        query = text("SELECT result FROM study_results WHERE study_hash = :study_hash")
        with self.engine.connect() as conn:
            res = conn.execute(query, {"study_hash": study_hash}).fetchone()
            return res[0] if res else None

    def save_study_result(
        self, study_hash: str, params: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Saves a study result to the database."""
        query = text(
            """
            INSERT INTO study_results (study_hash, params, result, timestamp)
            VALUES (:study_hash, :params, :result, :timestamp)
            ON CONFLICT (study_hash) DO UPDATE SET
                result = EXCLUDED.result, timestamp = EXCLUDED.timestamp
            """
        )
        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "study_hash": study_hash,
                    "params": self._safe_json_dumps(params),
                    "result": self._safe_json_dumps(result),
                    "timestamp": datetime.now(),
                },
            )

    # =========================================================================
    # Live-trading data accessors (replace the data/live/*.csv files)
    # =========================================================================

    def add_balance_snapshot(
        self,
        asset_class: str,
        timestamp: datetime,
        total_usd: float,
        free_usd: float,
        positions_usd: float,
        num_positions: int,
    ) -> None:
        """Append a row to live_balance_snapshots."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO live_balance_snapshots
                            (asset_class, timestamp, total_usd, free_usd,
                             positions_usd, num_positions)
                        VALUES (:ac, :ts, :tot, :free, :pos, :n)
                        ON CONFLICT (asset_class, timestamp) DO UPDATE SET
                            total_usd = EXCLUDED.total_usd,
                            free_usd = EXCLUDED.free_usd,
                            positions_usd = EXCLUDED.positions_usd,
                            num_positions = EXCLUDED.num_positions
                        """
                    ),
                    {
                        "ac": asset_class,
                        "ts": timestamp,
                        "tot": float(total_usd) if total_usd is not None else None,
                        "free": float(free_usd) if free_usd is not None else None,
                        "pos": float(positions_usd) if positions_usd is not None else None,
                        "n": int(num_positions) if num_positions is not None else None,
                    },
                )
        except Exception as e:
            print(f"WARNING: add_balance_snapshot failed: {e}")

    def add_positions_snapshot(
        self,
        asset_class: str,
        timestamp: datetime,
        positions: Dict[str, Any],
        circuit_breaker_triggered: bool = False,
    ) -> None:
        """Append a row to live_positions_snapshot (full active_positions blob)."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO live_positions_snapshot
                            (asset_class, timestamp, positions, circuit_breaker_triggered)
                        VALUES (:ac, :ts, CAST(:p AS JSONB), :cb)
                        ON CONFLICT (asset_class, timestamp) DO UPDATE SET
                            positions = EXCLUDED.positions,
                            circuit_breaker_triggered = EXCLUDED.circuit_breaker_triggered
                        """
                    ),
                    {
                        "ac": asset_class,
                        "ts": timestamp,
                        "p": self._safe_json_dumps(positions),
                        "cb": bool(circuit_breaker_triggered),
                    },
                )
        except Exception as e:
            print(f"WARNING: add_positions_snapshot failed: {e}")

    def get_balance_history(self, asset_class: str) -> pd.DataFrame:
        """Read live_balance_snapshots for an asset class as a DataFrame.

        Columns match the legacy balance_snapshots.csv schema:
        ``timestamp, total_usd, free_usd, positions_usd, num_positions``.
        """
        query = text(
            """
            SELECT timestamp, total_usd, free_usd, positions_usd, num_positions
            FROM live_balance_snapshots
            WHERE asset_class = :ac
            ORDER BY timestamp
            """
        )
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={"ac": asset_class})
        except Exception as e:
            print(f"WARNING: get_balance_history failed: {e}")
            return pd.DataFrame(
                columns=["timestamp", "total_usd", "free_usd", "positions_usd", "num_positions"]
            )

    def upsert_fear_greed(self, snapshot_date: Any, value: int, classification: str) -> None:
        """Insert-or-update a Fear & Greed Index value for a given date.

        Idempotent — repeated calls for the same date overwrite the value.
        """
        query = text(
            """
            INSERT INTO fear_greed_index (snapshot_date, value, classification)
            VALUES (:d, :v, :c)
            ON CONFLICT (snapshot_date) DO UPDATE
              SET value = EXCLUDED.value,
                  classification = EXCLUDED.classification,
                  inserted_at = now()
            """
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(query, {"d": snapshot_date, "v": int(value), "c": classification})
        except Exception as e:
            print(f"WARNING: upsert_fear_greed failed: {e}")

    def get_fear_greed_history(self, days: int = 365) -> pd.DataFrame:
        """Read the last N days of Fear & Greed values, newest first."""
        query = text(
            """
            SELECT snapshot_date, value, classification
            FROM fear_greed_index
            ORDER BY snapshot_date DESC
            LIMIT :n
            """
        )
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={"n": int(days)})
        except Exception as e:
            print(f"WARNING: get_fear_greed_history failed: {e}")
            return pd.DataFrame(columns=["snapshot_date", "value", "classification"])

    def get_orders_df(self, run_id: str) -> pd.DataFrame:
        """Read the orders table for a run_id (replaces trade_log.csv)."""
        query = text(
            """
            SELECT timestamp, symbol, side, order_id, price, amount,
                   amount_usd, fee, fee_currency
            FROM orders
            WHERE run_id = :rid
            ORDER BY timestamp
            """
        )
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={"rid": run_id})
        except Exception as e:
            print(f"WARNING: get_orders_df failed: {e}")
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "order_id",
                    "price",
                    "amount",
                    "amount_usd",
                    "fee",
                    "fee_currency",
                ]
            )

    def get_closed_positions_df(self, run_id: str) -> pd.DataFrame:
        """Read closed positions from the trades table (replaces position_closes.csv).

        Returns columns mirroring the legacy CSV: ``close_timestamp, symbol,
        entry_time, exit_time, entry_price, exit_price, amount, gross_pnl,
        net_pnl, pnl_pct, fee_entry, fee_exit, fee_total, hold_duration_hours,
        exit_reason``.
        """
        query = text(
            """
            SELECT exit_time AS close_timestamp,
                   symbol,
                   entry_time, exit_time,
                   entry_price, exit_price,
                   amount,
                   gross_pnl,
                   profit AS net_pnl,
                   profit_pct AS pnl_pct,
                   fee_entry, fee_exit, fee_total,
                   hold_duration_hours, exit_reason
            FROM trades
            WHERE run_id = :rid AND status = 'closed'
            ORDER BY exit_time
            """
        )
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={"rid": run_id})
        except Exception as e:
            print(f"WARNING: get_closed_positions_df failed: {e}")
            return pd.DataFrame()

    def close(self) -> None:
        """Disposes of the SQLAlchemy engine pool."""
        self.engine.dispose()
