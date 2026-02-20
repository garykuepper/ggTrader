import sys
import os
import pandas as pd
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

try:
    from ggTrader.utils.plotting import plot_wfo_splits

    print("SUCCESS: Imported plot_wfo_splits")
except ImportError as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)

# Create dummy data
idx = pd.date_range("2023-01-01", periods=100, freq="4h")
data = {"close": np.random.rand(100)}
df = pd.DataFrame(data, index=idx)


class MockRM:
    def __init__(self):
        self.run_dir = "test_run"
        self.plots_dir = "test_run/plots"
        os.makedirs(self.plots_dir, exist_ok=True)


rm = MockRM()

print("Calling plot_wfo_splits...")
# 100 bars, 2 splits, 0.5 ratio => 66 bar window
plot_wfo_splits(df, window_len=66, set_lens=0.5, n_splits=2, results_manager=rm)

plot_file = "test_run/plots/wfo_splits.png"
if os.path.exists(plot_file):
    print(f"FINAL SUCCESS: Plot found at {plot_file}")
    print(f"File size: {os.path.getsize(plot_file)} bytes")
else:
    print("FINAL FAILURE: Plot file not found")
