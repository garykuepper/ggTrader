import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from typing import Any
from ggTrader.utils.config import get_db_connection_string


class ResultDBManager:
    """
    Manages the PostgreSQL database for storing trading run results.
    Replaces the previous DuckDB implementation.
    """

    def __init__(self, connection_string=None, log_path="results/runs_log.csv"):
        if connection_string is None:
            self.connection_string = get_db_connection_string()
        else:
            self.connection_string = connection_string

        self.log_path = Path(log_path).absolute()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(self.connection_string)
        self._init_db()
        self._init_log()

    def _init_db(self):
        """Initializes the PostgreSQL tables and updates schema if needed."""
        with self.engine.begin() as conn:
            # 1. Ensure RUNS table exists
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

            # 2. Schema Migration: Add new columns if they don't exist
            # Postgres 9.6+ supports IF NOT EXISTS
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

            # 3. Other Tables (WFO, Metrics, etc.) - Unchanged
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS wfo_windows (
                    run_id VARCHAR,
                    window_id INTEGER,
                    test_start TIMESTAMPTZ,
                    test_end TIMESTAMPTZ,
                    start_capital DOUBLE PRECISION,
                    end_capital DOUBLE PRECISION,
                    profit DOUBLE PRECISION,
                    return_pct DOUBLE PRECISION,
                    sharpe DOUBLE PRECISION,
                    sortino DOUBLE PRECISION,
                    params JSONB,
                    PRIMARY KEY (run_id, window_id)
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    run_id VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE PRECISION,
                    PRIMARY KEY (run_id, metric_name)
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS equity_curves (
                    run_id VARCHAR,
                    timestamp TIMESTAMPTZ,
                    equity_value DOUBLE PRECISION,
                    PRIMARY KEY (run_id, timestamp)
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS trades (
                    run_id VARCHAR,
                    symbol VARCHAR,
                    entry_time TIMESTAMPTZ,
                    exit_time TIMESTAMPTZ,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    profit DOUBLE PRECISION,
                    profit_pct DOUBLE PRECISION,
                    status VARCHAR,
                    PRIMARY KEY (run_id, symbol, entry_time)
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS study_results (
                    study_hash VARCHAR PRIMARY KEY,
                    params JSONB,
                    result JSONB,
                    timestamp TIMESTAMPTZ
                );
            """
                )
            )

    def _safe_json_dumps(self, obj: Any) -> str:
        """JSON dumps that ensures NaNs and Infs are converted to null."""

        def clean_obj(v):
            if isinstance(v, float):
                if v != v or v == float("inf") or v == float("-inf"):
                    return None
            elif isinstance(v, dict):
                return {k: clean_obj(v2) for k, v2 in v.items()}
            elif isinstance(v, (list, tuple)):
                return [clean_obj(v2) for v2 in v]
            return v

        return json.dumps(clean_obj(obj))

    def _init_log(self):
        """Initializes the CSV log file if it doesn't exist."""
        # Check if file exists to decide if we need header
        file_exists = self.log_path.exists()

        # If it exists, check if we need to migrate header (lazy migration: just append new cols if missing?)
        # For simplicity, we'll just ensure the file exists with the NEW header if it's new.
        # If old, we might have format mismatch.
        # Strategy: Read header, if mismatch, maybe rewrite?
        # Safer: Just append. But let's set the STANDARD header for new files.
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

        if not file_exists:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def add_run(self, run_id, run_type, script_name, parameters, metadata=None, metrics=None):
        """
        Adds a new run entry.

        Args:
            run_id (str): UUID
            run_type (str): 'backtest', 'wfo', 'sensitivity'
            script_name (str): Script filename
            parameters (dict): Strategy params
            metadata (dict): Configuration/Context (e.g. date range, symbols)
            metrics (dict): Results (sharpe, profit, etc.)
        """
        timestamp = datetime.now()
        meta = metadata or {}
        metr = metrics or {}

        # Extract core fields
        sharpe = metr.get("oos_sharpe", metr.get("sharpe"))
        sortino = metr.get("sortino")
        total_profit = metr.get("total_profit")

        # Extract config from metadata
        start_date = meta.get("START_DATE") or meta.get("start_date")
        end_date = meta.get("END_DATE") or meta.get("end_date")
        interval = meta.get("INTERVAL") or meta.get("interval")

        # Database Parsed Insert
        query = text(
            """
            INSERT INTO runs (
                "run_id", "run_type", "timestamp", "script_name", "parameters", "metadata",
                "sharpe", "sortino", "total_profit", "start_date", "end_date", "interval"
            )
            VALUES (
                :rid, :rtype, :ts, :sname, :params, :meta,
                :sr, :sort, :prof, :sdate, :edate, :inter
            )
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
            # Optionally print keys to check for missing binds
            # print(f"Available keys: {list(bind_params.keys())}")

        # Append to CSV Log
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

    def _append_to_log(
        self,
        timestamp,
        run_id,
        run_type,
        status,
        sharpe,
        sortino,
        profit,
        interval,
        start,
        end,
    ):
        """Appends a line to the external CSV log."""

        # Handle Nones for CSV
        def safe_fmt(val, fmt="{:.4f}"):
            return fmt.format(val) if isinstance(val, (int, float)) else str(val) if val else ""

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
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

    def add_wfo_results(self, run_id, df_res):
        """Inserts WFO window results into the database."""
        if df_res.empty:
            return

        records = []
        for i, row in df_res.iterrows():
            params_json = self._safe_json_dumps(row["params"])
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
                    "params": params_json,
                }
            )

        df_to_insert = pd.DataFrame(records)
        df_to_insert.to_sql(
            "wfo_windows", self.engine, if_exists="append", index=False, method="multi"
        )

    def add_metrics(self, run_id, metrics_dict):
        """Adds multiple metrics for a specific run."""
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
                        stmt,
                        {
                            "run_id": run_id,
                            "metric_name": name,
                            "metric_value": float(value),
                        },
                    )

    def add_equity_curve(self, run_id, equity_series):
        """Inserts equity curve data into the database."""
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

    def add_trades(self, run_id, df_trades):
        """Inserts trade records into the database."""
        if df_trades.empty:
            return

        mapping = {
            "symbol": "symbol",
            "entry_time": ["entry_time", "entry_date"],
            "exit_time": ["exit_time", "exit_date"],
            "entry_price": "entry_price",
            "exit_price": "exit_price",
            "profit": "profit",
            "profit_pct": "profit_pct",
            "status": "status",
        }

        records = []
        for _, row in df_trades.iterrows():
            try:
                symbol = row.get("symbol", "UNKNOWN")
                entry_time = next(
                    (row[c] for c in mapping["entry_time"] if c in row and pd.notna(row[c])),
                    None,
                )
                exit_time = next(
                    (row[c] for c in mapping["exit_time"] if c in row and pd.notna(row[c])),
                    None,
                )

                record = {
                    "run_id": run_id,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": float(row.get("entry_price", 0.0)),
                    "exit_price": float(row.get("exit_price", 0.0)),
                    "profit": float(row.get("profit", 0.0)),
                    "profit_pct": float(row.get("profit_pct", 0.0)),
                    "status": row.get("status", "closed"),
                }
                records.append(record)
            except Exception as e:
                print(f"Error processing trade row: {e}")

        if not records:
            return

        query = text(
            """
            INSERT INTO trades (run_id, symbol, entry_time, exit_time, entry_price, exit_price, profit, profit_pct, status)
            VALUES (:run_id, :symbol, :entry_time, :exit_time, :entry_price, :exit_price, :profit, :profit_pct, :status)
            ON CONFLICT (run_id, symbol, entry_time) DO UPDATE SET
                exit_time = EXCLUDED.exit_time,
                entry_price = EXCLUDED.entry_price,
                exit_price = EXCLUDED.exit_price,
                profit = EXCLUDED.profit,
                profit_pct = EXCLUDED.profit_pct,
                status = EXCLUDED.status
        """
        )

        with self.engine.begin() as conn:
            conn.execute(query, records)

    def check_existing_study(self, study_hash):
        """Checks if a study with the given hash already exists."""
        query = text("SELECT result FROM study_results WHERE study_hash = :study_hash")
        with self.engine.connect() as conn:
            res = conn.execute(query, {"study_hash": study_hash}).fetchone()
            if res:
                return res[0]
        return None

    def save_study_result(self, study_hash, params, result):
        """Saves a study result to the database."""
        timestamp = datetime.now()

        query = text(
            """
            INSERT INTO study_results (study_hash, params, result, timestamp)
            VALUES (:study_hash, :params, :result, :timestamp)
            ON CONFLICT (study_hash) DO UPDATE SET
                result = EXCLUDED.result,
                timestamp = EXCLUDED.timestamp
        """
        )

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "study_hash": study_hash,
                    "params": self._safe_json_dumps(params),
                    "result": self._safe_json_dumps(result),
                    "timestamp": timestamp,
                },
            )

    def close(self):
        self.engine.dispose()
