import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
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
        """Initializes the PostgreSQL tables if they don't exist."""
        with self.engine.begin() as conn:
            # Runs table
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

            # WFO Windows table
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

            # Performance Metrics table
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

            # Equity Curves table
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

            # Trades table
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

            # Study Results for Caching
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
        # sqlalchemy handles dict -> JSONB automatically if dialect supports
        # but manual json.dumps is safer if using simple params binding
        # Actually sqlalchemy with psycopg2 handles dicts correctly for JSONB

        query = text(
            """
            INSERT INTO runs (run_id, run_type, timestamp, script_name, parameters, metadata)
            VALUES (:run_id, :run_type, :timestamp, :script_name, :parameters, :metadata)
        """
        )

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "run_id": run_id,
                    "run_type": run_type,
                    "timestamp": timestamp,
                    "script_name": script_name,
                    "parameters": json.dumps(parameters),  # Ensure JSON string
                    "metadata": json.dumps(metadata) if metadata else "{}",
                },
            )

        self._append_to_log(
            timestamp,
            run_id,
            run_type,
            "SUCCESS",
            metadata.get("profit_pct", 0) if metadata else 0,
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
        if df_res.empty:
            return

        records = []
        for i, row in df_res.iterrows():
            params_json = json.dumps(row["params"])
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
                    "sharpe": row["sharpe"],
                    "sortino": row["sortino"],
                    "params": params_json,
                }
            )

        # Bulk insert
        # Check for conflicts? (run_id, window_id) is PK.
        # Simple insert is fine assuming clean run.

        # Using pandas to_sql is easiest
        df_to_insert = pd.DataFrame(records)
        df_to_insert.to_sql(
            "wfo_windows", self.engine, if_exists="append", index=False, method="multi"
        )

    def add_metrics(self, run_id, metrics_dict):
        """Adds multiple metrics for a specific run."""
        with self.engine.begin() as conn:
            for name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    # Upsert logic for Postgres using ON CONFLICT
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

        # Prepare DataFrame for bulk insert
        df = equity_series.reset_index()
        df.columns = ["timestamp", "equity_value"]
        df["run_id"] = run_id

        # Ensure timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # We can use to_sql, but what about conflicts?
        # If running same run_id twice, we might want upsert.
        # But usually runs are unique.

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
            # Fallback to row-by-row upsert if strict uniqueness needed?
            # For now assume unique runs.

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
                    (
                        row[c]
                        for c in mapping["entry_time"]
                        if c in row and pd.notna(row[c])
                    ),
                    None,
                )
                exit_time = next(
                    (
                        row[c]
                        for c in mapping["exit_time"]
                        if c in row and pd.notna(row[c])
                    ),
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

        # Bulk insert
        # We want to use ON CONFLICT DO UPDATE
        # SQLAlchemy supports this with dialects

        from sqlalchemy.dialects.postgresql import insert

        with self.engine.begin() as conn:
            stmt = insert(text("trades")).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id", "symbol", "entry_time"],
                set_={
                    "exit_time": stmt.excluded.exit_time,
                    "entry_price": stmt.excluded.entry_price,
                    "exit_price": stmt.excluded.exit_price,
                    "profit": stmt.excluded.profit,
                    "profit_pct": stmt.excluded.profit_pct,
                    "status": stmt.excluded.status,
                },
            )
            # This requires 'trades' Table object or reflection.
            # Since we used raw SQL creation, we might not have Table object easily unless we reflect.
            # Reflection is easy.
            pass

        # Alternative: use raw SQL with executemany
        # It's cleaner given we control schema
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
                # In Postgres, JSONB is returned as dict automatically by psycopg2
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
                    "params": json.dumps(params),
                    "result": json.dumps(result),
                    "timestamp": timestamp,
                },
            )

    def close(self):
        self.engine.dispose()
