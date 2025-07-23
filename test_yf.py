from swing_trader.data.data_loader import DataLoader
import yfinance as yf
import matplotlib.pyplot as plt  # Add this for plotting
from pymongo import MongoClient

ticker = 'SPY'

def main():
    data_loader = DataLoader()
    df = yf.download(ticker, start='2018-01-01',auto_adjust=True)

    print(df.columns)
    # Remove the second level of the columns MultiIndex
    df.columns = df.columns.droplevel('Ticker')
    df.columns.name = None
    df = df.reset_index()
    df['Date'] = df['Date'].astype(str)
    df['Ticker'] = ticker
    df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print(df.columns)
    print(df)
    # df['Close'].plot()
    # plt.show()  # To actually display the plot

    # Insert into MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['ggTrader']          # <--- matches your screenshot!
    collection = db['stock_data']    # <--- matches your screenshot!

    data = df.to_dict(orient='records')
    collection.insert_many(data)
    print("Inserted", len(data), "records into ggTrader.stock_data.")


if __name__ == "__main__":
    main()