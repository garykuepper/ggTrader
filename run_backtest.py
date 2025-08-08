from backtest_simulator import BacktestSimulator
from datetime import datetime, timedelta, timezone
from data_manager import CryptoDataManager, DataManager
from utils import align_end_to_interval  # NEW

# Symbols (dynamic, with safe fallback)
symbols = CryptoDataManager.get_top_binance_usdt_symbols(top_n=10, min_change=0.0, min_trades=0, min_volume=100000)
if not symbols:
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'SOLUSDT',
        'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LINKUSDT', 'TRXUSDT',
    ]
symbols = list(dict.fromkeys(symbols))  # Deduplicate while preserving order

# Backtest configuration
INTERVAL = '4h'
now_utc = datetime.now(timezone.utc)
end_date = align_end_to_interval(now_utc, INTERVAL)  # aligned to closed bar
start_date = end_date - timedelta(days=30*3)
initial_cash = 1000.0

# Default params (used if DB has nothing)
DEFAULT_EMA_FAST = 10
DEFAULT_EMA_SLOW = 30
DEFAULT_TRAILING_PCT = 0.04

# Try to pull the latest optimized parameters
strategy_name = 'EMA_trailing'
dm = DataManager()
latest = dm.get_latest_strategy_params(strategy=strategy_name, interval=INTERVAL)

if latest and all(k in latest for k in ('ema_fast', 'ema_slow', 'trailing_pct')):
    EMA_FAST = latest['ema_fast']
    EMA_SLOW = latest['ema_slow']
    TRAILING_PCT = latest['trailing_pct']
    print(f"Using latest optimized params from MongoDB for {strategy_name} @ {INTERVAL}: "
          f"ema_fast={EMA_FAST}, ema_slow={EMA_SLOW}, trailing_pct={TRAILING_PCT}")
else:
    EMA_FAST = DEFAULT_EMA_FAST
    EMA_SLOW = DEFAULT_EMA_SLOW
    TRAILING_PCT = DEFAULT_TRAILING_PCT
    print(f"Using default params: ema_fast={EMA_FAST}, ema_slow={EMA_SLOW}, trailing_pct={TRAILING_PCT}")

# Prepare data
cm = CryptoDataManager()
price_data_map = cm.build_price_data(symbols, INTERVAL, start_date, end_date)

# Optionally include fees (if you added fee support in the simulator)
FEE_PCT = 0.001

# Run simulation via shared helper
simulator = BacktestSimulator.run_with_uniform_params(
    symbols=symbols,
    price_data_map=price_data_map,
    ema_fast=EMA_FAST,
    ema_slow=EMA_SLOW,
    trailing_pct=TRAILING_PCT,
    initial_cash=initial_cash,
    interval=INTERVAL,   # ensure interval-aware metrics
    fee_pct=FEE_PCT,     # apply transaction costs
    verbose=True
)

# Performance summaries
simulator.print_performance_summary()

# Optional: plot a few selected symbols
if symbols:
    for sym in symbols[:3]:
        simulator.plot_symbol(sym, start_date, end_date)
