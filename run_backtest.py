from backtest_simulator import BacktestSimulator
from data_manager import CryptoDataManager
from datetime import datetime, timedelta, timezone
from trading_strategy import EMAStrategy

# Symbols (deduplicated while preserving order)
symbols = [
    'BTCUSDT',
    'ETHUSDT',
    'LTCUSDT',
    'ADAUSDT',
    'SOLUSDT',
    'DOGEUSDT',
    'XRPUSDT',
    'BNBUSDT',
]
symbols = list(dict.fromkeys(symbols))

# Use one set of params across all symbols (consistent with optuna script approach)
EMA_FAST = 14
EMA_SLOW = 21
TRAILING_PCT = 0.045

# Backtest configuration
INTERVAL = '1h'
end_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
start_date = end_date - timedelta(days=30 )
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
simulator.plot_symbol('LTCUSDT', start_date, end_date)
simulator.plot_symbol('ADAUSDT', start_date, end_date)
simulator.plot_symbol('BTCUSDT', start_date, end_date)
