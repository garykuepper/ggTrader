"""PostgreSQL database manager for storing trading run results."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import Connection, create_engine, text

from ggTrader.utils.config import get_db_connection_string


class ResultDBManager:
    """Manages the PostgreSQL database for storing trading run results."""

    def __init__(
        self, connection_string: Optional[str] = None, log_path: str = "results/runs_log.csv"
    ) -> None:
        self.connection_string = connection_string or get_db_connection_string()

        self.log_path = Path(log_path).absolute()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(self.connection_string)
        self._init_db()
        self._init_log()

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
            CREATE TABLE IF NOT EXISTS study_results (
                study_hash VARCHAR PRIMARY KEY, params JSONB, result JSONB, timestamp TIMESTAMPTZ
            );
            """,
        ]
        for query in queries:
            conn.execute(text(query))

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

    def _init_log(self) -> None:
        """Initializes the CSV log file header if it doesn't exist."""
        if not self.log_path.exists():
            header = [
                "timestamp",
                "run_id",
                "type",
                "status",
                "sharpe",
                "sortino",
                "profit",
                "interval",
                "start_date",
                "end_date",
            ]
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _append_to_log(
        self,
        timestamp: datetime,
        run_id: str,
        run_type: str,
        status: str,
        sharpe: Any,
        sortino: Any,
        profit: Any,
        interval: Any,
        start: Any,
        end: Any,
    ) -> None:
        """Appends a line to the external CSV log."""

        def safe_fmt(val: Any, fmt: str = "{:.4f}") -> str:
            return fmt.format(val) if isinstance(val, (int, float)) else str(val) if val else ""

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    timestamp.isoformat(),
                    run_id,
                    run_type,
                    status,
                    safe_fmt(sharpe),
                    safe_fmt(sortino),
                    safe_fmt(profit, "{:.2f}"),
                    safe_fmt(interval),
                    safe_fmt(start),
                    safe_fmt(end),
                ]
            )

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
    ) -> None:
        """Adds a new primary run entry to the database and CSV log."""
        timestamp = datetime.now()
        meta = metadata or {}
        metr = metrics or {}

        sharpe = metr.get("oos_sharpe", metr.get("sharpe"))
        sortino = metr.get("sortino")
        total_profit = metr.get("total_profit")

        start_date = meta.get("START_DATE") or meta.get("start_date")
        end_date = meta.get("END_DATE") or meta.get("end_date")
        interval = meta.get("INTERVAL") or meta.get("interval")

        query = text(
            """
            INSERT INTO runs (
                "run_id", "run_type", "timestamp", "script_name", "parameters", "metadata",
                "sharpe", "sortino", "total_profit", "start_date", "end_date", "interval"
            )
            VALUES (:rid, :rtype, :ts, :sname, :params, :meta, :sr, :sort, :prof, :sdate, :edate, :inter)
            """
        )

        bind_params = {
            "rid": run_id,
            "rtype": run_type,
            "ts": timestamp,
            "sname": script_name,
            "params": self._safe_json_dumps(parameters),
            "meta": self._safe_json_dumps(meta),
            "sr": sharpe,
            "sort": sortino,
            "prof": total_profit,
            "sdate": start_date,
            "edate": end_date,
            "inter": interval,
        }

        try:
            with self.engine.begin() as conn:
                conn.execute(query, bind_params)
        except Exception as e:
            print(f"WARNING: Could not save run results to DB: {e}")

        self._append_to_log(
            timestamp,
            run_id,
            run_type,
            "SUCCESS",
            sharpe,
            sortino,
            total_profit,
            interval,
            start_date,
            end_date,
        )

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
                if isinstance(value, (int, float)):
                    stmt = text(
                        """
                        INSERT INTO performance_metrics (run_id, metric_name, metric_value)
                        VALUES (:run_id, :metric_name, :metric_value)
                        ON CONFLICT (run_id, metric_name) 
                        DO UPDATE SET metric_value = EXCLUDED.metric_value
                        """
                    )
                    conn.execute(
                        stmt, {"run_id": run_id, "metric_name": name, "metric_value": float(value)}
                    )

    def add_equity_curve(self, run_id: str, equity_series: pd.Series) -> None:
        """Inserts equity curve data into the database in bulk."""
        if equity_series.empty:
            return

        df = equity_series.reset_index()
        df.columns = ["timestamp", "equity_value"]
        df["run_id"] = run_id
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        try:
            df.to_sql(
                "equity_curves",
                self.engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
        except Exception as e:
            print(f"Error inserting equity curve: {e}")

    def add_trades(self, run_id: str, df_trades: pd.DataFrame) -> None:
        """Parses and inserts trade records into the database."""
        if df_trades.empty:
            return

        records = [
            record
            for _, row in df_trades.iterrows()
            if (record := self._parse_trade_row(row, run_id)) is not None
        ]

        if not records:
            return

        query = text(
            """
            INSERT INTO trades (run_id, symbol, entry_time, exit_time, entry_price, exit_price, profit, profit_pct, status)
            VALUES (:run_id, :symbol, :entry_time, :exit_time, :entry_price, :exit_price, :profit, :profit_pct, :status)
            ON CONFLICT (run_id, symbol, entry_time) DO UPDATE SET
                exit_time = EXCLUDED.exit_time, entry_price = EXCLUDED.entry_price,
                exit_price = EXCLUDED.exit_price, profit = EXCLUDED.profit,
                profit_pct = EXCLUDED.profit_pct, status = EXCLUDED.status
            """
        )

        with self.engine.begin() as conn:
            conn.execute(query, records)

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

    def close(self) -> None:
        """Disposes of the SQLAlchemy engine pool."""
        self.engine.dispose()
