from swing_trader.data.mr_data import MrData
from datetime import datetime, timedelta
from swing_trader.trading.portfolio import Portfolio
from tabulate import tabulate
from swing_trader.strategy.swing_strategy import SwingStrategy

mr_data = MrData()

# Create datetime objects
start_dt = datetime(2016, 1, 5)
# end_dt = datetime(2024, 4, 1)
end_dt = datetime.today() - timedelta(days=1)

# Convert datetime objects to strings in YYYY-MM-DD format
start_date = start_dt.strftime("%Y-%m-%d")
end_date = end_dt.strftime("%Y-%m-%d")

# Download stock data for SSO, SH, and SPY
for ticker in ["SSO", "SH", "SPY"]:
    mr_data.get_stock_data(ticker, start_date, end_date)

# Enrich the tickers with indicators
# print("Enriching tickers with indicators...")
# mr_data.enrich_ticker_with_indicators(ticker="SSO")
# mr_data.enrich_ticker_with_indicators(ticker="SH")
mr_data.enrich_ticker_with_indicators(ticker="SPY")

spy_data = mr_data.get_stock_data_collection(ticker="SPY")

# Display the last 10 rows of SPY data
print(spy_data[['Ticker', 'Date', 'Close', 'momentum_rsi', 'trend_macd']].tail(10))

strat = SwingStrategy(
    db=mr_data.db,
    signal_ticker="SPY",
    long_ticker="SSO",
    short_ticker="SH"
)
print("Generating swing signals...")
strat.generate_signals()

sigs = strat.get_signals()

start = start_date
end = end_date

# Filter rows within the date range
filtered = sigs[(sigs['Date'] >= start) & (sigs['Date'] <= end)]

# print(filtered)

# Buy and Hold Portfolio
bh_sp_portfolio = Portfolio(cash=10000.0, name="Buy and Hold Portfolio")
spy_price = mr_data.get_stock_price('SPY', start_date)
qty = bh_sp_portfolio.cash / spy_price
bh_sp_portfolio.add_position(ticker='SPY',
                             quantity=qty,
                             bought_price=spy_price)

# Initialize the portfolio for trading based on signals
portfolio = Portfolio(cash=10000.0, name="Test Portfolio")
# Add positions based on the signals
prev_signal = None

transactions = []

print("Processing signals and executing trades...")
for _, row in filtered.iterrows():
    cur_signal = row['strat_swing_signal']
    if prev_signal == cur_signal:
        continue  # Skip repeated signals

    if cur_signal == 'HOLD':
        continue  # Skip HOLD signals

    if cur_signal == 'BUY':

        # if 'SH' in portfolio.positions:
        #     qty = portfolio.positions['SH']['quantity']
        #     price = mr_data.get_stock_price('SH', row['Date'])
        #     portfolio.remove_position(ticker='SH', quantity=qty, price=price)
        #     value = qty * price
        #     transactions.append(['SELL','SH', row['Date'], qty, f"${price:,.2f}", f"${value:,.2f}"])

        price = mr_data.get_stock_price('SSO', row['Date'])
        qty = portfolio.cash / price
        portfolio.add_position(ticker='SSO', quantity=qty, bought_price=price)
        value = qty * price
        transactions.append(['BUY', 'SSO', row['Date'], qty, f"${price:,.2f}", f"${value:,.2f}"])

    elif cur_signal == 'SELL':

        if 'SSO' in portfolio.positions:
            qty = portfolio.positions['SSO']['quantity']
            price = mr_data.get_stock_price('SSO', row['Date'])
            portfolio.remove_position(ticker='SSO', quantity=qty, price=price)
            value = qty * price
            transactions.append(['SELL', 'SSO', row['Date'], qty, f"${price:,.2f}", f"${value:,.2f}"])

        # Add SH position
        # price = mr_data.get_stock_price('SH', row['Date'])
        # qty = portfolio.cash / price
        # portfolio.add_position(ticker='SH', quantity=qty, bought_price=price)
        # value = qty * price
        # transactions.append(['BUY', 'SH', row['Date'], qty, f"${price:,.2f}", f"${value:,.2f}"])

    prev_signal = cur_signal

# Print transaction table
print(tabulate(
    transactions[-10:],
    headers=['Action', 'Ticker', 'Date', 'Quantity', 'Price', 'Value'],
    floatfmt=('.0f', '', '.2f', '.2f', '.2f'),
    tablefmt='github'
))

portfolio.update_prices('SSO', mr_data.get_stock_price('SSO', end_date))
bh_sp_portfolio.update_prices('SPY', mr_data.get_stock_price('SPY', end_date))

final_value = portfolio.total_portfolio_value()
bh_value = bh_sp_portfolio.total_portfolio_value()
performance = (final_value / bh_value - 1) * 100

print(f"\nFinal Portfolio Value: ${final_value:,.2f}")
print(f"Buy and Hold Portfolio Value: ${bh_value:,.2f}")
print(f"Relative Performance: {performance:.2f}%")
