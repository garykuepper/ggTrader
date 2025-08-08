from backtest_simulator import BacktestSimulator
from data_manager import CryptoDataManager
from datetime import datetime, timedelta
from trading_strategy import TradingStrategy, EMAStrategy

def to_binance_symbol(sym: str) -> str:
    """
    Normalize a symbol to Binance format:
    - Remove hyphens
    - Convert ...USD to ...USDT
    - Leave already-correct symbols (e.g., BTCUSDT) unchanged
    """
    s = sym.strip().upper()
    if '-' in s:
        s = s.replace('-', '')
    if s.endswith('USD') and not s.endswith('USDT'):
        s = s[:-3] + 'USDT'
    return s

def extract_base(sym: str) -> str:
    """
    Extract the base asset from a symbol in various formats:
    BTC-USD -> BTC, BTCUSDT -> BTC, BTCUSD -> BTC
    """
    s = sym.strip().upper().replace('-', '')
    for suffix in ('USDT', 'USD'):
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s  # fallback if no known suffix

# Change symbols here only
symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']

# Normalize symbols for internal use
symbols = [to_binance_symbol(s) for s in symbols]

# Per-symbol strategy parameters and position sizing defaults.
# Add or adjust entries here if you want custom settings per coin.
EMA_PARAMS = {
    'BTC': {'ema_fast': 5, 'ema_slow': 20},
    'ETH': {'ema_fast': 10, 'ema_slow': 25},
    'LTC': {'ema_fast': 10, 'ema_slow': 25},
}
TRAILING_PCT = {
    'BTC': 0.03,
    'ETH': 0.05,
    'LTC': 0.05,
}
POSITION_PCT = {
    'BTC': 0.10,
    'ETH': 0.20,
    'LTC': 0.10,
}

# Sensible fallbacks if a symbol isn't in maps above
DEFAULT_EMA = {'ema_fast': 10, 'ema_slow': 25}
DEFAULT_TRAILING = 0.05
DEFAULT_POSITION_PCT = 0.10

# Prepare data
price_data_map = {}
cm = CryptoDataManager()
end_date = datetime(2025, 8, 1)
start_date = end_date - timedelta(days=365)
for symbol in symbols:
    price_data_map[symbol] = cm.get_crypto_data(symbol, "1d", start_date, end_date)

# Build strategy and position maps programmatically from the symbols list
symbol_strategy_map = {}
position_pct_map = {}

for sym in symbols:
    base = extract_base(sym)
    ema_cfg = EMA_PARAMS.get(base, DEFAULT_EMA)
    trailing = TRAILING_PCT.get(base, DEFAULT_TRAILING)
    # Strategy name updates automatically with the (normalized) symbol
    strategy_name = f"{sym} EMA"
    symbol_strategy_map[sym] = EMAStrategy(strategy_name, ema_cfg, trailing_pct=trailing)
    position_pct_map[sym] = POSITION_PCT.get(base, DEFAULT_POSITION_PCT)

# Run simulation
simulator = BacktestSimulator(
    symbol_strategy_map,
    price_data_map,
    position_pct_map,
    initial_cash=1000
)
simulator.run()
simulator.print_trade_history()