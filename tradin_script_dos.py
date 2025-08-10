# Python
from trading_strategy import EMAStrategy
from data_manager import CryptoDataManager
from datetime import datetime, timedelta, timezone
from tabulate import tabulate

def align_to_binance_4h(dt):
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=(dt.hour // 4) * 4)

cm = CryptoDataManager()
symbol, interval = "BTCUSDT", "4h"
end_date = align_to_binance_4h(datetime.now(timezone.utc))
start_date = end_date - timedelta(days=60)
df = cm.get_crypto_data(symbol, interval, start_date, end_date)

strategy = EMAStrategy(
    name="ema_crossover",
    params={'ema_fast': 10, 'ema_slow': 50},
    trailing_pct=0.05  # 5% trailing
)

result = strategy.backtest(df, starting_cash=1000, min_hold_bars=2, use_trailing=True)

print(f"Final cash: {result['final_cash']:.2f} (Return: {result['total_return_pct']:.2f}%)")
# Inspect signals or trades
signals = result['signal_df']
print(tabulate(signals.tail(10), headers='keys', tablefmt='github'))
print(f"Trades: {len(result['trades'])}")
for t in result['trades'][-3:]:
    print(f"{t.buy_time} buy {t.buy_price:.4f} -> {t.exit_time} {t.exit_price:.4f} ({t.exit_reason})")
