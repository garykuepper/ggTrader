import os
import sys
from sqlalchemy import create_engine, text


def enable_compression():
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
    )

    print(f"Connecting to DB...")
    engine = create_engine(connection_string)

    with engine.begin() as conn:
        print("--- Enabling TimescaleDB Compression ---")

        # 1. Enable compression on the hypertable
        # Compress by 'symbol' and 'interval' (segmentby) and order by 'timestamp' (orderby)
        print("1. Configuring compression settings (Segment by: symbol, interval)...")
        try:
            conn.execute(
                text(
                    """
                ALTER TABLE ohlcv SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol, interval',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """
                )
            )
            print("   Success.")
        except Exception as e:
            print(f"   Note: {e}")

        # 2. Add compression policy
        # Compress chunks older than 7 days
        print("2. Adding compression policy (Compress chunks > 7 days old)...")
        try:
            conn.execute(
                text("SELECT add_compression_policy('ohlcv', INTERVAL '7 days');")
            )
            print("   Success.")
        except Exception as e:
            # Policy might already exist
            print(f"   Note: {e}")

        # 3. Force compression on existing chunks logic?
        # Usually policy handles it, but we can manually compress current chunks if needed.
        # But 'compress_chunk' requires identifying chunks.
        # Easier to just let the policy run or wait.

        # However, for immediate effect on historical data:
        # We can look for all chunks.

        print("3. Compression enabled. The background worker will compress old data.")
        print("   To force verify, check back in a few minutes or run:")
        print("   SELECT * FROM hypertable_compression_stats('ohlcv');")


if __name__ == "__main__":
    enable_compression()
