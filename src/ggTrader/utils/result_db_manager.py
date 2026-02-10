import duckdb
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import uuid
import csv


class ResultDBManager:
    """
    Manages the DuckDB database for storing trading run results.
    """

    def __init__(
        self, db_path="results/trading_results.db", log_path="results/runs_log.csv"
    ):
        self.db_path = Path(db_path).absolute()
        self.log_path = Path(log_path).absolute()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_log()

    def _init_db(self):
        """Initializes the DuckDB tables if they don't exist."""
        with duckdb.connect(str(self.db_path)) as conn:
            # Runs table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    run_type VARCHAR,
                    timestamp TIMESTAMPTZ,
                    script_name VARCHAR,
                    parameters JSON,
                    metadata JSON
                )
            """
            )

            # WFO Windows table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wfo_windows (
                    run_id VARCHAR,
                    window_id INTEGER,
                    test_start TIMESTAMPTZ,
                    test_end TIMESTAMPTZ,
                    start_capital DOUBLE,
                    end_capital DOUBLE,
                    profit DOUBLE,
                    return_pct DOUBLE,
                    sharpe DOUBLE,
                    sortino DOUBLE,
                    params JSON,
                    PRIMARY KEY (run_id, window_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Performance Metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    run_id VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE,
                    PRIMARY KEY (run_id, metric_name),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

            # Equity Curves table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_curves (
                    run_id VARCHAR,
                    timestamp TIMESTAMPTZ,
                    equity_value DOUBLE,
                    PRIMARY KEY (run_id, timestamp),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """
            )

    def _init_log(self):
        """Initializes the CSV log file if it doesn't exist."""
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "run_id", "type", "status", "summary_metric"]
                )

    def add_run(self, run_id, run_type, script_name, parameters, metadata=None):
        """Adds a new run entry to the database and CSV log."""
        timestamp = datetime.now()
        params_json = json.dumps(parameters)
        meta_json = json.dumps(metadata) if metadata else "{}"

        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, run_type, timestamp, script_name, parameters, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (run_id, run_type, timestamp, script_name, params_json, meta_json),
            )

        # Log to CSV (initial entry, status 'STARTED' or 'PENDING' usually)
        # However, scripts often call this at the end, so we might want a 'SUCCESS' status
        self._append_to_log(
            timestamp,
            run_id,
            run_type,
            "SUCCESS",
            metadata.get("total_return_pct", 0) if metadata else 0,
        )

    def _append_to_log(self, timestamp, run_id, run_type, status, summary_metric):
        """Appends a line to the external CSV log."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp.isoformat(),
                    run_id,
                    run_type,
                    status,
                    f"{summary_metric:.4f}",
                ]
            )

    def add_wfo_results(self, run_id, df_res):
        """Inserts WFO window results into the database."""
        with duckdb.connect(str(self.db_path)) as conn:
            for i, row in df_res.iterrows():
                # Handle test_start/test_end which might be date objects
                test_start = row["test_start"]
                if hasattr(test_start, "isoformat"):
                    test_start = test_start.isoformat()

                test_end = row["test_end"]
                if hasattr(test_end, "isoformat"):
                    test_end = test_end.isoformat()

                params_json = json.dumps(row["params"])

                conn.execute(
                    """
                    INSERT INTO wfo_windows 
                    (run_id, window_id, test_start, test_end, start_capital, end_capital, profit, return_pct, sharpe, sortino, params)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        i,
                        test_start,
                        test_end,
                        row["start_capital"],
                        row["end_capital"],
                        row["profit"],
                        row["return_pct"],
                        row["sharpe"],
                        row["sortino"],
                        params_json,
                    ),
                )

    def add_metrics(self, run_id, metrics_dict):
        """Adds multiple metrics for a specific run."""
        with duckdb.connect(str(self.db_path)) as conn:
            for name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    conn.execute(
                        """
                        INSERT INTO performance_metrics (run_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                        ON CONFLICT (run_id, metric_name) DO UPDATE SET metric_value = excluded.metric_value
                    """,
                        (run_id, name, float(value)),
                    )

    def add_equity_curve(self, run_id, equity_series):
        """Inserts equity curve data into the database."""
        with duckdb.connect(str(self.db_path)) as conn:
            for ts, val in equity_series.items():
                if hasattr(ts, "isoformat"):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)

                conn.execute(
                    """
                    INSERT INTO equity_curves (run_id, timestamp, equity_value)
                    VALUES (?, ?, ?)
                    ON CONFLICT (run_id, timestamp) DO UPDATE SET equity_value = excluded.equity_value
                """,
                    (run_id, ts_str, float(val)),
                )
