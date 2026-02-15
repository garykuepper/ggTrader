import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from .constants import kraken_map


# python scripts/data/ingest_kraken_data.py --sync


class KrakenPostgresIngestor:
    """
    Ingests Kraken OHLCV data into a PostgreSQL database (TimescaleDB optimized).
    """

    def __init__(self, connection_string):
        """
        Args:
            connection_string (str): SQLAlchemy connection string
            e.g. postgresql://user:pass@localhost:5432/ggtrader
        """
        # Increase pool size to support high concurrency
        self.engine = create_engine(connection_string, pool_size=20, max_overflow=10)
        self.intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]
        self._init_db()

        # Define Table object for SQLAlchemy Core operations (needed for insert().on_conflict...)
        from sqlalchemy import (
            MetaData,
            Table,
            Column,
            String,
            TIMESTAMP,
            Float,
            Integer,
        )

        metadata = MetaData()
        self.ohlcv_table = Table(
            "ohlcv",
            metadata,
            Column("timestamp", TIMESTAMP, primary_key=True),
            Column("symbol", String, primary_key=True),
            Column("interval", String, primary_key=True),
            Column("open", Float),
            Column("high", Float),
            Column("low", Float),
            Column("close", Float),
            Column("volume", Float),
            Column("trades", Integer),
        )

    def _init_db(self):
        """Creates the OHLCV hypertable."""
        with self.engine.connect() as conn:
            # Create standard table (using raw SQL for specific TimescaleDB needs if any,
            # though Table.create() could also work. Keeping raw SQL for now to ensure exact schema control)
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
                # We check if it's already a hypertable to avoid errors
                # Or just use if_not_exists logic
                conn.execute(
                    text(
                        "SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);"
                    )
                )
            except Exception as e:
                # print(f"TimescaleDB hypertable creation skipped: {e}")
                pass

            conn.commit()

    def _clean_ccy(self, ccy):
        return kraken_map.get(ccy.upper(), ccy.upper())

    def _split_pair(self, filename_stem, quote_only="USD"):
        p = filename_stem.upper()

        # Robust quote detection
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
        quotes.sort(
            key=len, reverse=True
        )  # Check longer suffixes first (e.g. ZUSD before USD)

        for q in quotes:
            if p.endswith(q):
                raw_base = p[: -len(q)]
                # If raw_base is empty, it means the symbol IS the quote (unlikely but possible)
                if not raw_base:
                    continue

                base = self._clean_ccy(raw_base)
                quote = self._clean_ccy(q)
                return f"{base}-{quote}"

        # Fallback: just return the stem if no standard quote found
        return self._clean_ccy(p)

    def _bulk_upsert(self, records):
        """Execute a bulk upsert for a batch of records."""
        if not records:
            return

        from sqlalchemy.dialects.postgresql import insert

        # Batch DB operations locally to keep memory sane inside this flush
        chunk_size = 5000
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = insert(self.ohlcv_table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp", "symbol", "interval"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "trades": stmt.excluded.trades,
                },
            )
            with self.engine.begin() as conn:
                conn.execute(stmt)

    def _parse_ohlc_file(self, file_path, symbol, interval_str):
        """
        Parse a pre-aggregated OHLC file and return list of records.
        """
        try:
            # Kraken OHLC might have no header
            df = pd.read_csv(
                file_path,
                names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
                header=None,
            )

            if df.empty:
                return []

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["symbol"] = symbol
            df["interval"] = interval_str

            # Postgres needs: timestamp, symbol, interval, open, high, low, close, volume, trades
            # Ensure order
            cols = [
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
            df = df[cols]
            return df.to_dict(orient="records")

        except Exception as e:
            # We log error but return empty list to keep process running
            print(f"Error parsing OHLC file {file_path}: {e}")
            return []

    def ingest_dir(self, raw_dir, max_workers=16):
        """
        Ingest files using Producer-Consumer pattern:
        Parallel Workers (Parse) -> Main Thread (Accumulate & Batch Output).
        """
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            return

        # Resumability
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
            print(
                f"All {len(csv_files)} files in {os.path.basename(raw_dir)} already processed."
            )
            return

        print(
            f"Ingesting {len(pending_files)}/{len(csv_files)} files in {os.path.basename(raw_dir)} using {max_workers} worker threads with batching..."
        )

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

        # INTERNAL HELPER: Purely CPU/IO work, returns Data Records
        def parse_file(file_path):
            try:
                filename = os.path.basename(file_path)
                stem = os.path.splitext(filename)[0].strip()

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

                return filename, records, None
            except Exception as e:
                return os.path.basename(file_path), [], str(e)

        # MAIN LOOP
        buffer_records = []
        BUFFER_SIZE = 50000

        pbar = tqdm(
            total=len(pending_files), desc="Processing Files", unit="file", leave=False
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(parse_file, fp): fp for fp in pending_files}

            for future in as_completed(futures):
                fname, records, error = future.result()

                if error:
                    print(f"\nError processing {fname}: {error}")
                else:
                    buffer_records.extend(records)
                    processed_files.add(fname)

                # Flush buffer if full
                if len(buffer_records) >= BUFFER_SIZE:
                    self._bulk_upsert(buffer_records)
                    buffer_records = []  # clear buffer

                    # Update manifest periodically on flush
                    with open(manifest_path, "w") as f:
                        json.dump(list(processed_files), f)

                pbar.update(1)

        # Final flush
        if buffer_records:
            self._bulk_upsert(buffer_records)

        # Final manifest
        with open(manifest_path, "w") as f:
            json.dump(list(processed_files), f)

        pbar.close()

    def _parse_raw_trades_file(self, file_path, symbol):
        """
        Parse raw trades, resample, and return list of records.
        """
        try:
            # Read Trades CSV
            # Use header=None, forcing names.
            # CAUTION: If file HAS header, first row will be junk strings.
            # But Kraken raw files usually don't.
            df = pd.read_csv(
                file_path, names=["timestamp", "price", "volume"], header=None
            )

            if df.empty:
                return []

            # Simple check: if first row 'timestamp' is not numeric, drop it (it might be a header)
            if (
                isinstance(df.iloc[0]["timestamp"], str)
                and not df.iloc[0]["timestamp"].replace(".", "", 1).isdigit()
            ):
                df = df.iloc[1:].copy()

            # Ensure numeric
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

            # Filter out bad dates (e.g. 1970). Kraken started ~2013?
            # Let's say anything < 2010 is bad
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

                all_records.extend(final.to_dict(orient="records"))

            return all_records

        except Exception as e:
            print(f"Error parsing raw trades {file_path}: {e}")
            return []
