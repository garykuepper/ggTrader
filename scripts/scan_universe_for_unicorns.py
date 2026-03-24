"""
Scan the Top 50 universe to identify 'Unicorns' (high volatility, high return assets).
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ggTrader.utils.setup import load_data_and_setup

def main():
    config = {
        "SYMBOLS_FILE": "data/top_50_USD_2023-01-01_2025-12-31.json",
        "INTERVAL": "4h",
        "START_DATE": "2024-01-01",
        "END_DATE": "2025-12-31"  # Focus on the most recent bull/chop cycle
    }
    
    print(f"Loading Top 50 OHLCV for {config['START_DATE']} to {config['END_DATE']}...")
    try:
        ohlcv = load_data_and_setup(config)
    except Exception as e:
        print(f"Error: {e}")
        return

    close = ohlcv.xs("close", axis=1, level=1)
    returns = close.pct_change()
    
    # Calculate Metrics
    # Annualized Volatility (assuming 4h bars: 6 bars/day * 365 = 2190 bars/year)
    vol = returns.std() * np.sqrt(2190)
    
    # Total Return
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max) - 1
    max_dd = drawdown.min()
    
    # Unicorn Score: Volatility * Total Return (rewarding tokens that move fast and up)
    unicorn_score = vol * total_return
    
    stats = pd.DataFrame({
        "Total Return (%)": total_return * 100,
        "Ann. Volatility (%)": vol * 100,
        "Max Drawdown (%)": max_dd * 100,
        "Unicorn Score": unicorn_score
    }).sort_values("Unicorn Score", ascending=False)
    
    print("\n" + "="*80)
    print("TOP 50 UNIVERSE SCAN (2024-2025)")
    print("="*80)
    print(stats.head(20).to_markdown())
    print("="*80)
    
    # Identify which ones are NOT in our Top 25
    top_25 = [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "SUI", "ADA", "ZEC", "PEPE", "LINK",
        "AVAX", "LTC", "MATIC", "WIF", "DOT", "XLM", "NEAR", "FTM", "TIA", "SEI",
        "FET", "CRV", "ICP", "XMR", "SHIB"
    ]
    
    new_unicorns = stats[~stats.index.str.split('-').str[0].isin(top_25)].head(10)
    
    print("\nPROPOSED 'NEW' UNICORNS (from Ranks 26-50):")
    print(new_unicorns.to_markdown())

if __name__ == "__main__":
    main()
