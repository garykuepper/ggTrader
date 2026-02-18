import os
import sys
import numpy as np

# Ensure project root is in path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.orchestrator import run_sensitivity_orchestrator

CONSTANTS = {
    "SYMBOLS": ["BTC-USD"],
    "START_DATE": "2023-01-01",
    "END_DATE": "2023-01-31",
    "INTERVAL": "1h",
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.1,
    "FEES": 0.0,
    "MIN_TRADES": 1000,  # Ridiculously high for verification
}

params = {
    "adx_threshold": [25, 30],
}

try:
    print("Running sensitivity with MIN_TRADES=1000...")
    results = run_sensitivity_orchestrator(
        config=CONSTANTS,
        param_grid=params,
        save_results=False,
        show_progress=False,
    )

    # Check if results_df has Sharpe Ratio as NaN
    df = results["results_df"]
    print("\nResults DataFrame:")
    print(df)

    if df["Sharpe Ratio"].isna().all():
        print("\nSUCCESS: All results filtered out as expected.")
    else:
        print("\nFAILURE: Some results were not filtered out.")

except Exception as e:
    print(f"\nCaught expected or unexpected error: {e}")

# Test with reasonable trades
CONSTANTS["MIN_TRADES"] = 1
print("\nRunning sensitivity with MIN_TRADES=1...")
results = run_sensitivity_orchestrator(
    config=CONSTANTS,
    param_grid=params,
    save_results=False,
    show_progress=False,
)
df = results["results_df"]
print("\nResults DataFrame:")
print(df)
if not df["Sharpe Ratio"].isna().any():
    print("\nSUCCESS: Results present with reasonable threshold.")
else:
    print("\nFAILURE: Results filtered out unexpectedly.")
