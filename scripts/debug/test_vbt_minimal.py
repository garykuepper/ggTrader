import argparse
import traceback

import numpy as np
import pandas as pd
import vectorbt as vbt


def test() -> None:
    """
    Minimal test of vectorbt with MultiIndex DataFrames.
    """
    print("Testing vbt with MultiIndex...")
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    # 2 levels
    cols = pd.MultiIndex.from_tuples(
        [("BTC", "close"), ("BTC", "open")], names=["symbol", "ohlcv"]
    )
    df = pd.DataFrame(np.random.rand(10, 2), index=idx, columns=cols)
    print(f"DF cols: {df.columns}")
    print(f"DF cols names: {df.columns.names}")

    try:
        # Try vbt portfolio from signals
        print("Running vbt.Portfolio.from_signals...")
        pf = vbt.Portfolio.from_signals(
            df.xs("close", axis=1, level=1, drop_level=False),
            df.xs("close", axis=1, level=1, drop_level=False) > 0.5,
            df.xs("close", axis=1, level=1, drop_level=False) < 0.2,
        )
        print("vbt success!")
        print(pf.total_return())
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()


def main() -> None:
    """
    Main orchestration for VBT minimal test.
    """
    parser = argparse.ArgumentParser(description="Minimal VectorBT MultiIndex Test")
    parser.parse_args()  # Allow --help
    test()


if __name__ == "__main__":
    main()
