from backtest_simulator import BacktestSimulator
from data_manager import CryptoDataManager
from datetime import datetime, timedelta, timezone
from trading_strategy import EMAStrategy

def get_top_binance_usdt_symbols(top_n=10, min_change=0.0, min_trades=0, min_volume=100000):
    """
    Return a deduped list of the top USDT pairs on Binance.US by 24h quote volume.
    Filters out leveraged/ETF-like products and obvious non-tradables for this setup.
    """
    try:
        top_pairs = CryptoDataManager.get_24hr_top_binance(
            top_n=top_n * 2,  # overfetch to allow filtering
            quote='USDT',
            min_change=min_change,
            min_trades=min_trades,
            min_volume=min_volume
        )
        raw_symbols = [p.get('symbol', '') for p in top_pairs if isinstance(p, dict)]
        # Filter out leveraged/ETF-like tickers and stablecoin-to-stablecoin
        banned_fragments = ('UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT', 'USDCUSDT')
        filtered = [s for s in raw_symbols if s.endswith('USDT') and not any(b in s for b in banned_fragments)]
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(filtered))
        return deduped[:top_n]
    except Exception as e:
        print(f"⚠️ Failed to fetch top Binance.US pairs: {e}")
        return []

# Symbols (dynamic, with safe fallback)
symbols = get_top_binance_usdt_symbols(top_n=10)
if not symbols:
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'SOLUSDT',
        'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LINKUSDT', 'TRXUSDT',
    ]
symbols = list(dict.fromkeys(symbols))  # Deduplicate while preserving order

# Use one set of params across all symbols (consistent with optuna script approach)
EMA_FAST = 46
EMA_SLOW = 58
TRAILING_PCT = 0.0816

# Backtest configuration
INTERVAL = '4h'
end_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
start_date = end_date - timedelta(days=30 * 2)
initial_cash = 1000.0

# Prepare data
cm = CryptoDataManager()
price_data_map = {sym: cm.get_crypto_data(sym, INTERVAL, start_date, end_date) for sym in symbols}

# Build strategy and position maps
pos_pct = 1.0 / len(symbols) if symbols else 0.0
if pos_pct == 0.0:
    raise ValueError("symbols list is empty; cannot compute equal position size.")

symbol_strategy_map = {}
position_pct_map = {}
for sym in symbols:
    params = {'ema_fast': EMA_FAST, 'ema_slow': EMA_SLOW}
    strategy_name = f"{sym} EMA({EMA_FAST},{EMA_SLOW})"
    symbol_strategy_map[sym] = EMAStrategy(strategy_name, params, trailing_pct=TRAILING_PCT)
    position_pct_map[sym] = pos_pct

# Run simulation
simulator = BacktestSimulator(
    symbol_strategy_map=symbol_strategy_map,
    price_data_map=price_data_map,
    position_pct_map=position_pct_map,
    initial_cash=initial_cash
)
simulator.run()
# Optional: detailed trades
# simulator.print_trade_history()

# Performance summaries aligned with optuna printing
simulator.print_performance_summary()

# Optional: plot a symbol
if symbols:
    # Plot a few of the selected symbols if available
    for sym in symbols[:3]:
        simulator.plot_symbol(sym, start_date, end_date)
