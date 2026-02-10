import os
import duckdb
import json
from tqdm import tqdm
from .constants import kraken_map


class KrakenDuckDBIngestor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]

    def _clean_ccy(self, ccy):
        return kraken_map.get(ccy.upper(), ccy.upper())

    def _split_pair(self, filename_stem, quote_only="USD"):
        p = filename_stem.upper()
        if p.endswith(quote_only):
            raw_base = p[: -len(quote_only)]
            base = self._clean_ccy(raw_base)
            base = self._clean_ccy(base)
            quote = quote_only
            pair_std = f"{base}-{quote}"
            return base, quote, pair_std
        return None, None, None

    def ingest_dir(self, raw_dir):
        """
        Ingest all CSV files in a directory into DuckDB.
        """
        conn = duckdb.connect(self.db_path)

        all_files = []
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f.endswith(".csv"):
                    all_files.append(os.path.join(root, f))

        if not all_files:
            return

        for file_path in all_files:
            filename = os.path.basename(file_path)
            stem = filename.split(".")[0]
            base, quote, pair_std = self._split_pair(stem)

            if not base:
                continue

            try:
                temp_table = f"temp_trades_{stem.replace('-', '_')}"
                conn.execute(
                    f"CREATE TEMP TABLE {temp_table} (timestamp DOUBLE, price DOUBLE, volume DOUBLE)"
                )
                conn.execute(
                    f"COPY {temp_table} FROM '{file_path.replace('\\', '/')}' (DELIMITER ',', HEADER FALSE)"
                )

                for interval in self.intervals:
                    bucket_sql = ""
                    if interval.endswith("m"):
                        bucket_sql = f"INTERVAL {interval[:-1]} MINUTE"
                    elif interval.endswith("h"):
                        bucket_sql = f"INTERVAL {interval[:-1]} HOUR"
                    elif interval.endswith("d"):
                        bucket_sql = f"INTERVAL {interval[:-1]} DAY"

                    sql = f"""
                        INSERT INTO ohlcv
                        SELECT 
                            time_bucket({bucket_sql}, to_timestamp(timestamp)) as timestamp,
                            first(price) as open,
                            max(price) as high,
                            min(price) as low,
                            last(price) as close,
                            sum(volume * price) as volume,
                            count(*) as trades,
                            '{base}' as base,
                            '{quote}' as quote,
                            '{pair_std}' as pair,
                            '{interval}' as interval
                        FROM {temp_table}
                        GROUP BY 1
                        ON CONFLICT (pair, interval, timestamp) DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close,
                            volume = excluded.volume,
                            trades = excluded.trades;
                    """
                    conn.execute(sql)

                conn.execute(f"DROP TABLE {temp_table}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        conn.close()
