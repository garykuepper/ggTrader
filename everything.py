from backtest_simulator import BacktestSimulator
from data_manager import CryptoDataManager
from datetime import datetime, timedelta
from trading_strategy import TradingStrategy, EMAStrategy

# Download data for multiple symbols
symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD']
price_data_map = {}
cm = CryptoDataManager()
end_date = datetime(2025,8,1)
start_date = end_date - timedelta(days=365)
for symbol in symbols:
    price_data_map[symbol] = cm.get_crypto_data(symbol, "1d", start_date, end_date)

# Define different strategies for each symbol
symbol_strategy_map = {
    'BTC-USD': EMAStrategy('BTC-USD EMA', {'ema_fast': 5, 'ema_slow': 20}, trailing_pct=0.03),
    'ETH-USD': EMAStrategy('ETH-USD EMA', {'ema_fast': 10, 'ema_slow': 25}, trailing_pct=0.05),
    'LTC-USD': EMAStrategy('LTC-USD EMA', {'ema_fast': 10, 'ema_slow': 25}, trailing_pct=0.05),
}
position_pct_map = {
    'BTC-USD': 0.10,
    'ETH-USD': 0.20,
    'LTC-USD': 0.10,
}
# Run simulation
simulator = BacktestSimulator(
    symbol_strategy_map,
    price_data_map,
    position_pct_map,
    initial_cash=1000
)
simulator.run()
simulator.print_trade_history()