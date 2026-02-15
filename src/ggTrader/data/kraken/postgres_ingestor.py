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

from .constants import kraken_map


class KrakenPostgresIngestor:
    """
    Ingests Kraken OHLCV data into a PostgreSQL database (TimescaleDB optimized).
    Uses a Producer-Consumer pattern to decouple parsing and database writing.
    """

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string (str): SQLAlchemy connection string
        """
        # We still use SQLAlchemy for init and high-level ops, but psycopg2 for bulk
        self.engine = create_engine(connection_string, pool_size=20, max_overflow=10)
        self.connection_string = connection_string
        self.intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]
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
                    PRIMARY KEY (timestamp, symbol, interval)
                );
                """
                )
            )
            # Convert to Hypertable (TimescaleDB)
            try:
                conn.execute(
                    text(
                        "SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);"
                    )
                )
            except Exception:
                pass
            conn.commit()

    def _clean_ccy(self, ccy: str) -> str:
        return kraken_map.get(ccy.upper(), ccy.upper())

    def _split_pair(self, filename_stem: str) -> str:
        p = filename_stem.upper()
        quotes = [
            "ZUSD",
            "ZEUR",
            "ZGBP",
            "ZJPY",
            "ZCAD",
            "ZAUD",
            "USDT",
            "USDC",
            "USD",
            "EUR",
            "GBP",
            "CAD",
            "AUD",
            "JPY",
            "XBT",
            "ETH",
        ]
        quotes.sort(key=len, reverse=True)

        for q in quotes:
            if p.endswith(q):
                raw_base = p[: -len(q)]
                if not raw_base:
                    continue
                base = self._clean_ccy(raw_base)
                quote = self._clean_ccy(q)
                return f"{base}-{quote}"
        return self._clean_ccy(p)

    def _db_writer_worker(
        self, data_queue: queue.Queue, stop_event: threading.Event
    ) -> None:
        """
        Dedicated thread for consuming record batches and writing to Postgres.
        Uses psycopg2 execute_values for maximum throughput.
        """
        # Create a dedicated connection for the writer
        conn = psycopg2.connect(
            self.connection_string.replace("postgresql+psycopg2://", "postgresql://")
        )
        conn.autocommit = True

        upsert_query = """
            INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
            VALUES %s
            ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades;
        """

        while not stop_event.is_set() or not data_queue.empty():
            try:
                # Use a timeout to allow checking stop_event periodically
                batch = data_queue.get(timeout=1.0)
                if batch:
                    with conn.cursor() as cur:
                        execute_values(cur, upsert_query, batch)
                data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nWriter Error: {e}")

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

            # Use raw tuples for execute_values
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["symbol"] = symbol
            df["interval"] = interval_str

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
                    ]
                ].itertuples(index=False, name=None)
            )
            return records
        except Exception as e:
            print(f"Error parsing OHLC file {file_path}: {e}")
            return []

    def _parse_raw_trades_file(self, file_path: str, symbol: str) -> list:
        """Parse raw trades, resample, and return list of tuples."""
        try:
            df = pd.read_csv(
                file_path, names=["timestamp", "price", "volume"], header=None
            )
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
                rule = interval.replace("m", "min").replace("d", "D")
                ohlc = df["price"].resample(rule).ohlc()
                vol = df["volume"].resample(rule).sum()
                trades = df["price"].resample(rule).count()

                final = pd.concat([ohlc, vol, trades], axis=1)
                final.columns = ["open", "high", "low", "close", "volume", "trades"]
                final.dropna(inplace=True)
                if final.empty:
                    continue

                final["symbol"] = symbol
                final["interval"] = interval
                final.reset_index(inplace=True)

                recs = list(
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
                        ]
                    ].itertuples(index=False, name=None)
                )
                all_records.extend(recs)
            return all_records
        except Exception as e:
            print(f"Error parsing raw trades {file_path}: {e}")
            return []

    def ingest_dir(self, raw_dir: str, max_workers: int = 16) -> None:
        """
        Ingest files using Producer-Consumer pattern.
        """
        csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            return

        manifest_path = os.path.join(raw_dir, ".processed_files.json")
        processed_files = set()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    processed_files = set(json.load(f))
            except Exception:
                pass

        pending_files = [
            f for f in csv_files if os.path.basename(f) not in processed_files
        ]
        if not pending_files:
            return

        print(
            f"Ingesting {len(pending_files)} files in {os.path.basename(raw_dir)} (Workers: {max_workers})..."
        )

        # Setup Parallelism
        data_queue = queue.Queue(maxsize=100)  # Flow control
        stop_event = threading.Event()
        writer_thread = threading.Thread(
            target=self._db_writer_worker, args=(data_queue, stop_event)
        )
        writer_thread.start()

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

        def parse_and_enqueue(file_path):
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

        pbar = tqdm(
            total=len(pending_files), desc="Processing Files", unit="file", leave=False
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(parse_and_enqueue, fp): fp for fp in pending_files
            }
            files_since_manifest = 0
            for future in as_completed(futures):
                fname = future.result()
                if fname:
                    processed_files.add(fname)
                    files_since_manifest += 1

                # Checkpointing
                if files_since_manifest >= 100:
                    with open(manifest_path, "w") as f:
                        json.dump(list(processed_files), f)
                    files_since_manifest = 0

                pbar.update(1)

        # Cleanup
        stop_event.set()
        writer_thread.join()

        with open(manifest_path, "w") as f:
            json.dump(list(processed_files), f)

        pbar.close()
