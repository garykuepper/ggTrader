import yfinance as yf
from yfinance import EquityQuery
from tabulate import tabulate
from yfinance.utils import auto_adjust
import pandas as pd
from datetime import datetime, timedelta

def get_ticker(ticker):
    dat = yf.Ticker(ticker)

    for key in dat.info.keys():
        print(F"{key}: {dat.info[key]}")


def download_yf(ticker, interval, csv_file, period=None, start_date=None, end_date=None):
    if period is not None:
        df = yf.download(ticker, period=period, interval=interval, multi_level_index=False, auto_adjust=False)
    elif start_date is not None and end_date is not None:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, multi_level_index=False, auto_adjust=False)
    else:
        raise ValueError("Either period or start_date and end_date must be provided")

    df.columns = df.columns.str.lower()
    df.index.name = 'date'
    df.to_csv(csv_file)


def read_print_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.set_index('date')
    # print(df)
    print(tabulate(df, headers='keys', tablefmt='github'))


def get_most_active_stocks():
    # Example query (commented out)
    # q = EquityQuery('and', [
    #        EquityQuery('gt', ['percentchange', 3]),
    #        EquityQuery('eq', ['region', 'us'])
    # ])
    # response = yf.screen(q, sortField='percentchange', sortAsc=True)

    # Using predefined screen for most active stocks
    response = yf.screen("most_actives")

    # Select the columns you want to display
    table_data = []
    stocks = response['quotes']
    # stock = stocks[0]
    # for key in stock.keys():
    #
    #     print(f"{key}:{stock[key]}")

    for stock in stocks:
        table_data.append([
            stock.get('symbol', ''),
            stock.get('regularMarketPrice', ''),
            stock.get('regularMarketVolume', '')
        ])

    # Define headers for the table
    headers = ['Symbol', 'Price', 'Volume']
    table_data = sorted(table_data, key=lambda x: x[2] if x[2] != '' else 0, reverse=True)
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt='github'))


def main():
    # get_most_active_stocks()
    # get_ticker("SPY")
    csv_file = "../../dl_data/yf_ltc_5y.csv"
    end_date = datetime(2025, 8, 3)
    start_date = end_date - timedelta(days=365*5)
    download_yf("BTC-USD", "1d",  csv_file, start_date=start_date,end_date=end_date)
    read_print_csv(csv_file)

if __name__ == '__main__':
    main()
