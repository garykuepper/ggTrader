import os
from sqlalchemy import create_engine, text


def enable_compression() -> None:
    """
    Configures and enables TimescaleDB compression on the OHLCV table.
    """
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader",
    )

    print("Connecting to database...")
    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    with engine.begin() as conn:
        print("--- Configuring TimescaleDB Compression ---")

        # 1. Configure compression on the hypertable
        print("1. Setting compression policy (Segment by: symbol, interval)...")
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
        print("2. Adding automatic compression policy (Chunks > 7 days)...")
        try:
            conn.execute(
                text("SELECT add_compression_policy('ohlcv', INTERVAL '7 days');")
            )
            print("   Success.")
        except Exception as e:
            print(f"   Note: {e}")

        print("\nCompression enabled. Background workers will process old data.")
        print(
            "To verify status, run: SELECT * FROM hypertable_compression_stats('ohlcv');"
        )


if __name__ == "__main__":
    enable_compression()
