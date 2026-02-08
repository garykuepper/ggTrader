import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ggTrader.core.portfolio import Portfolio

def test_sortino_realistic():
    print("Testing Sortino Ratio calculation with noisy data...")
    p = Portfolio(cash=10000)
    np.random.seed(42) # For reproducible results
    
    # Simulate 100 days
    n_days = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Generate random daily returns: 0.1% expected mean, 1.5% volatility
    daily_returns = np.random.normal(loc=0.001, scale=0.015, size=n_days)
    
    # Calculate equity curve
    current_val = 10000
    equity_values = []
    for ret in daily_returns:
        current_val *= (1 + ret)
        equity_values.append(current_val)
        
    for date, val in zip(dates, equity_values):
        p.equity_curve[pd.Timestamp(date)] = float(val)
        
    sortino = p.sortino_ratio()
    sharpe = p.sharpe_ratio()
    total_ret = (equity_values[-1] / 10000 - 1) * 100
    
    print(f"Total Period: {n_days} days")
    print(f"Final Value: ${equity_values[-1]:.2f} ({total_ret:.2f}%)")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    
    # Sanity checks for noisy data
    assert not np.isnan(sortino)
    assert not np.isnan(sharpe)
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_sortino_realistic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
