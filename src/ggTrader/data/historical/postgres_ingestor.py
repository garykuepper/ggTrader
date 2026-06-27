import glob
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from tqdm import tqdm

from ggTrader.data.core.constants import QUOTES, kraken_map


class PostgresIngestor:
    """
    Ingests historical OHLCV data into a PostgreSQL database (TimescaleDB optimized).
    Uses a Producer-Consumer pattern to decouple parsing and database writing.
    """

    def __init__(self, connection_string: str, venue: str = "kraken_spot"):
        """
        Args:
            connection_string (str): SQLAlchemy connection string
        """
        self.engine = create_engine(connection_string, pool_size=20, max_overflow=10)
        self.connection_string = connection_string
        self.intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]
        self.venue = venue
        self._init_db()

    def _init_db(self) -> None:
        """Creates the OHLCV hypertable if it doesn't exist."""
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    interval VARCHAR(5) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    trades INT,
                    venue VARCHAR(20) NOT NULL DEFAULT 'kraken_spot',
                    PRIMARY KEY (timestamp, symbol, interval, venue)
                );
                """
                )
            )
            try:
                conn.execute(
                    text("SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);")
                )
            except Exception:
                pass
            conn.commit()

    def _clean_ccy(self, ccy: str) -> str:
        return kraken_map.get(ccy.upper(), ccy.upper())

    def _split_pair(self, filename_stem: str) -> str:
        p = filename_stem.upper()
        quotes = sorted(QUOTES, key=len, reverse=True)

        for q in quotes:
            if p.endswith(q):
                raw_base = p[: -len(q)]
                if not raw_base:
                    continue
                base = self._clean_ccy(raw_base)
                quote = self._clean_ccy(q)
                return f"{base}-{quote}"
        return self._clean_ccy(p)

    def _db_writer_worker(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        """Dedicated thread for consuming record batches and writing to Postgres."""
        conn = psycopg2.connect(
            self.connection_string.replace("postgresql+psycopg2://", "postgresql://")
        )
        conn.autocommit = False

        upsert_query = """
            INSERT INTO ohlcv (
                timestamp, symbol, interval, open, high, low, close, volume, trades, venue
            )
            VALUES %s
            ON CONFLICT (timestamp, symbol, interval, venue) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades;
        """

        while not stop_event.is_set() or not data_queue.empty():
            try:
                batch = data_queue.get(timeout=1.0)
                if batch:
                    with conn.cursor() as cur:
                        execute_values(cur, upsert_query, batch, page_size=10000)
                    conn.commit()
                data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Writer Error: {e}")
                conn.rollback()

        conn.close()

    def _parse_ohlc_file(self, file_path: str, symbol: str, interval_str: str) -> list:
        """Parse pre-aggregated OHLC file into list of tuples."""
        try:
            df = pd.read_csv(
                file_path,
                names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
                header=None,
            )
            if df.empty:
                return []

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["symbol"] = symbol
            df["interval"] = interval_str
            df["venue"] = self.venue

            records = list(
                df[
                    [
                        "timestamp",
                        "symbol",
                        "interval",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "trades",
                        "venue",
                    ]
                ].itertuples(index=False, name=None)
            )
            return records
        except Exception as e:
            print(f"Error parsing OHLC file {file_path}: {e}")
            return []

    def _resample_trades_to_ohlcv(self, df: pd.DataFrame, interval: str, symbol: str) -> list:
        """Helper method to resample trade data to OHLCV format."""
        rule = interval.replace("m", "min").replace("d", "D")
        ohlc = df["price"].resample(rule).ohlc()
        vol = df["volume"].resample(rule).sum()
        trades = df["price"].resample(rule).count()

        final = pd.concat([ohlc, vol, trades], axis=1)
        final.columns = ["open", "high", "low", "close", "volume", "trades"]
        final.dropna(inplace=True)
        if final.empty:
            return []

        final["symbol"] = symbol
        final["interval"] = interval
        final["venue"] = self.venue
        final.reset_index(inplace=True)

        return list(
            final[
                [
                    "timestamp",
                    "symbol",
                    "interval",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trades",
                    "venue",
                ]
            ].itertuples(index=False, name=None)
        )

    def _parse_raw_trades_file(self, file_path: str, symbol: str) -> list:
        """Parse raw trades, resample, and return list of tuples."""
        try:
            df = pd.read_csv(file_path, names=["timestamp", "price", "volume"], header=None)
            if df.empty:
                return []

            if (
                isinstance(df.iloc[0]["timestamp"], str)
                and not df.iloc[0]["timestamp"].replace(".", "", 1).isdigit()
            ):
                df = df.iloc[1:].copy()

            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[df["timestamp"].dt.year >= 2010]
            if df.empty:
                return []

            df.set_index("timestamp", inplace=True)
            all_records = []
            for interval in self.intervals:
                recs = self._resample_trades_to_ohlcv(df, interval, symbol)
                all_records.extend(recs)
            return all_records
        except Exception as e:
            print(f"Error parsing raw trades {file_path}: {e}")
            return []

    def _load_manifest(self, state_key: str) -> set:
        """Load the processed-items manifest from system_state."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT value FROM system_state WHERE key = :k"),
                    {"k": state_key},
                ).fetchone()
        except Exception:
            return set()
        if row is None or row[0] is None:
            return set()
        try:
            return set(row[0])
        except TypeError:
            return set()

    def _save_manifest(self, state_key: str, data: set) -> None:
        """Persist the processed-items manifest to system_state."""
        payload = json.dumps(sorted(data))
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (:k, CAST(:v AS JSONB), now())
                    ON CONFLICT (key) DO UPDATE
                      SET value = EXCLUDED.value, updated_at = now()
                    """
                ),
                {"k": state_key, "v": payload},
            )

    def _start_writer_pool(
        self, num_writers: int
    ) -> tuple[queue.Queue, threading.Event, list[threading.Thread]]:
        """Initialize the DB writer threads and queue."""
        data_queue = queue.Queue(maxsize=100)
        stop_event = threading.Event()
        writer_threads = []
        for _ in range(num_writers):
            t = threading.Thread(target=self._db_writer_worker, args=(data_queue, stop_event))
            t.start()
            writer_threads.append(t)
        return data_queue, stop_event, writer_threads

    def _stop_writer_pool(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        writer_threads: list[threading.Thread],
    ) -> None:
        """Shutdown the DB writer threads and empty the queue."""
        data_queue.join()
        stop_event.set()
        for t in writer_threads:
            t.join()

    def _parse_and_enqueue(self, file_path: str, data_queue: queue.Queue):
        """Parse a CSV data file and push the records into the queue for writing."""
        suffix_pairs = [
            ("_1440", "1d"),
            ("_720", "12h"),
            ("_240", "4h"),
            ("_60", "1h"),
            ("_30", "30m"),
            ("_15", "15m"),
            ("_5", "5m"),
            ("_1", "1m"),
        ]
        try:
            fname = os.path.basename(file_path)
            stem = os.path.splitext(fname)[0].strip()
            is_ohlc = False
            interval_str = None
            symbol = None

            for suffix, interval in suffix_pairs:
                if stem.endswith(suffix):
                    is_ohlc = True
                    interval_str = interval
                    raw_base = stem[: -len(suffix)]
                    symbol = self._split_pair(raw_base)
                    break

            if is_ohlc:
                records = self._parse_ohlc_file(file_path, symbol, interval_str)
            else:
                symbol = self._split_pair(stem)
                records = self._parse_raw_trades_file(file_path, symbol)

            if records:
                data_queue.put(records)
            return fname
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def _execute_thread_pool(
        self,
        pending_files: list,
        data_queue: queue.Queue,
        processed_files: set,
        manifest_key: str,
        max_workers: int,
    ) -> None:
        """Run the core ThreadPool parsing loop for the pending files."""
        pbar = tqdm(total=len(pending_files), desc="Processing Files", unit="file", leave=False)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._parse_and_enqueue, fp, data_queue): fp for fp in pending_files
            }
            files_since_manifest = 0
            for future in as_completed(futures):
                fname = future.result()
                if fname:
                    processed_files.add(fname)
                    files_since_manifest += 1

                if files_since_manifest >= 100:
                    self._save_manifest(manifest_key, processed_files)
                    files_since_manifest = 0

                pbar.update(1)
        pbar.close()

    def ingest_dir(self, raw_dir: str, max_workers: int = 16) -> None:
        """
        Ingest files using Producer-Consumer pattern.
        """
        csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            return

        manifest_key = f"ingest_processed_files:{os.path.basename(raw_dir)}"
        processed_files = self._load_manifest(manifest_key)

        pending_files = [f for f in csv_files if os.path.basename(f) not in processed_files]
        if not pending_files:
            return

        print(
            f"Ingesting {len(pending_files)} files in {os.path.basename(raw_dir)} "
            f"(Workers: {max_workers})..."
        )

        # Setup Parallelism
        data_queue, stop_event, writer_threads = self._start_writer_pool(num_writers=4)

        self._execute_thread_pool(
            pending_files, data_queue, processed_files, manifest_key, max_workers
        )

        # Cleanup
        self._stop_writer_pool(data_queue, stop_event, writer_threads)
        self._save_manifest(manifest_key, processed_files)

    def list_quarter_dirs(self, raw_path: str, prefix: str = "Kraken_OHLCVT_") -> list:
        """Find quarterly Kraken folders under data/raw."""
        out = []
        if not os.path.isdir(raw_path):
            return out
        for name in os.listdir(raw_path):
            full = os.path.join(raw_path, name)
            if os.path.isdir(full) and name.startswith(prefix):
                out.append(full)
        out.sort()
        return out

    def sync_local_data(self, root_dir: str, force: bool = False, **kwargs) -> None:
        """
        Intelligently sync new raw CSV directories into the PostgreSQL dataset.
        Uses a manifest to avoid re-processing directories.

        Args:
            root_dir: The root project directory containing 'data/raw'
            force: If True, ignore manifest and sync all directories
        """
        raw_path = os.path.join(root_dir, "data", "raw")
        manifest_key = "ingest_processed_dirs"

        processed_dirs = self._load_manifest(manifest_key) if not force else set()

        all_dirs = self.list_quarter_dirs(raw_path=raw_path)

        if force:
            new_dirs = all_dirs
            print(f"Force sync: processing all {len(new_dirs)} directories")
        else:
            new_dirs = [d for d in all_dirs if os.path.basename(d) not in processed_dirs]

        if not new_dirs:
            print("No new data directories found to sync.")
            return

        print(
            f"Found {len(new_dirs)} new directories to sync: "
            f"{[os.path.basename(d) for d in new_dirs]}"
        )

        for d in tqdm(new_dirs, desc="Syncing Directories", unit="dir"):
            self.ingest_dir(d)
            processed_dirs.add(os.path.basename(d))
            self._save_manifest(manifest_key, processed_dirs)

        print("Data synchronization complete.")
