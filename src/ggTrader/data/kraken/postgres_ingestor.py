import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from .constants import kraken_map


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
        self.engine = create_engine(connection_string)
        self.intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]
        self._init_db()

    def _init_db(self):
        """Creates the OHLCV hypertable."""
        with self.engine.connect() as conn:
            # Create standard table
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
        if p.endswith(quote_only):
            raw_base = p[: -len(quote_only)]
            base = self._clean_ccy(raw_base)
            quote = quote_only
            pair_std = f"{base}-{quote}"
            return pair_std
        return filename_stem

    def ingest_dir(self, raw_dir):
        """
        Ingest all CSV files in a directory into PostgreSQL/TimescaleDB.
        Assumes raw CSVs contain Trades (timestamp, price, volume) and need resampling.
        """
        csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)

        if not csv_files:
            return

        print(f"Found {len(csv_files)} CSV files. Starting ingestion...")

        for file_path in tqdm(csv_files):
            filename = os.path.basename(file_path)
            stem = filename.split(".")[0]
            symbol = self._split_pair(stem)

            try:
                # Read Trades CSV
                # Assumes columns: timestamp, price, volume
                df = pd.read_csv(
                    file_path, names=["timestamp", "price", "volume"], header=None
                )

                if df.empty:
                    continue

                # Convert timestamp
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

                # Set index for resampling
                df.set_index("timestamp", inplace=True)

                for interval in self.intervals:
                    # Map interval string to pandas offset alias
                    # 1m -> 1T, 1h -> 1H, 1d -> 1D
                    rule = interval.replace("m", "T").replace("d", "D")
                    # 'h' works as is 1h -> 1h? No, Pandas uses 'h' or 'H'. '1h' -> '1h' works usually.

                    # Resample to OHLCV
                    ohlc = df["price"].resample(rule).ohlc()
                    vol = df["volume"].resample(rule).sum()
                    trades = df["price"].resample(rule).count()

                    final = pd.concat([ohlc, vol, trades], axis=1)
                    final.columns = ["open", "high", "low", "close", "volume", "trades"]

                    # Drop rows with no trades (if volume is 0 or NaN)
                    final.dropna(inplace=True)

                    if final.empty:
                        continue

                    final["symbol"] = symbol
                    final["interval"] = interval
                    final.reset_index(inplace=True)

                    # Insert
                    final.to_sql(
                        "ohlcv",
                        self.engine,
                        if_exists="append",
                        index=False,
                        method="multi",  # batch insert
                        chunksize=5000,
                    )

            except Exception as e:
                print(f"Error processing {filename}: {e}")
