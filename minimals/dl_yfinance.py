import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient
from tabulate import tabulate
import pandas_market_calendars as mcal
import pandas as pd


def yf_download_df(symbol, interval, start_date, end_date):
    df = yf.download(symbol,
                     start=start_date,
                     end=end_date,
                     interval=interval,
                     progress=False,
                     multi_level_index=False,
                     auto_adjust=True)

    df['symbol'] = symbol
    df['interval'] = interval
    df.columns = df.columns.str.lower()
    return df


def mongodb_to_df(results):
    if not results:
        return pd.DataFrame().set_index(pd.DatetimeIndex([]))
    df = pd.DataFrame(results)
    # If 'date' is already the index, access it accordingly
    if 'date' in df.columns:
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize('UTC')
        df = df.set_index('date')
    elif isinstance(df.index, pd.DatetimeIndex):
        # If index is DatetimeIndex and naive, localize it
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    else:
        raise KeyError("'date' column or index not found in DataFrame")

    # Drop _id if exists
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    return df


def get_market_days(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    market_days = nyse.schedule(start_date=start_date, end_date=end_date)

    return market_days


def df_to_mongodb_format(df):
    df['date'] = df.index
    records = df.to_dict('records')
    return records


def find_missing_dates(request_dates, available_dates):

    if request_dates.empty:
        missing_list = [[]]
        dates_only = []
        return missing_list, dates_only

    # normalize timezone
    available_dates = available_dates.index.tz_convert(
        'UTC') if available_dates.index.tz else available_dates.index.tz_localize('UTC')
    request_dates = request_dates.index.tz_convert(
        'UTC') if request_dates.index.tz else request_dates.index.tz_localize('UTC')
    # normalize
    available_dates = available_dates.normalize()
    request_dates = request_dates.normalize()

    missing_dates = request_dates.difference(available_dates)
    missing_list = [[d.date()] for d in missing_dates]
    dates_only = [d[0] for d in missing_list]

    return missing_list, dates_only


def fetch_from_mongodb(symbol, interval, start_date, end_date):
    """
    return as df
    """

    query = {
        'symbol': symbol,
        'interval': interval,
        'date': {
            '$gte': start_date,
            '$lte': end_date
        }
    }
    results = list(collection.find(query).sort('date', 1))

    # missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
    return mongodb_to_df(results)


def get_missing_dates(symbol, interval, start_date, end_date, dates_only):
    dfs = []
    for date in dates_only:
        df = yf_download_df(symbol, interval, date, date + timedelta(days=1))
        dfs.append(df)

    missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
    # For missing_df (downloaded)
    if missing_df.index.tz is None:
        missing_df.index = missing_df.index.tz_localize('UTC')
    return missing_df


def fetch_data(symbol, interval, start_date, end_date):
    from_mongodb_df = fetch_from_mongodb(symbol, interval, start_date, end_date)
    market_days = get_market_days(start_date, end_date)
    missing_list, dates_only = find_missing_dates(market_days, from_mongodb_df)

    # For df with timezone-naive index
    if from_mongodb_df.index.tz is None:
        from_mongodb_df.index = from_mongodb_df.index.tz_localize('UTC')

    # if none missing just return
    if len(missing_list) <= 0:
        return from_mongodb_df

    missing_df = get_missing_dates(symbol, interval, start_date, end_date, dates_only)

    # insert missing into db
    insert_into_db(missing_df)
    # join and sort
    combined_df = (pd.concat([from_mongodb_df, missing_df]).sort_index())

    return remove_duplicates(combined_df)


def remove_duplicates(combined_df):
    combined_df.index.name = 'date'  # Name the index 'date'
    if 'date' in combined_df.columns:
        combined_df = combined_df.drop(columns=['date'])
    combined_df = combined_df.reset_index()  # 'date' becomes a column
    combined_df = combined_df.drop_duplicates(subset=['symbol', 'date', 'interval'])
    combined_df = combined_df.set_index('date')
    return combined_df


def insert_into_db(df):
    records = df_to_mongodb_format(df)
    for record in records:
        collection.update_one(
            {'symbol': record['symbol'], 'date': record['date'], 'interval': record['interval']},  # unique key filter
            {'$set': record},  # update with new data
            upsert=True  # insert if not exists
        )
    print(f"Upserted {len(records)} documents.")


def get_collection():
    # Setup mongodb
    client = MongoClient('mongodb://localhost:27017/')
    db = client['learning']
    collection = db['stock_market_data']
    indexes = collection.index_information()
    if 'symbol_1_date_1_interval_1' not in indexes:
        collection.create_index([('symbol', 1), ('date', 1), ('interval', 1)], unique=True)
    return collection


# Stock Data
symbol = "SPY"
interval = "15m"
end_date = datetime(2025, 7, 11)
start_date = end_date - timedelta(days=1)

collection = get_collection()

df = fetch_from_mongodb(symbol, interval, start_date, end_date)
market_days = get_market_days(start_date, end_date)
missing_list, _ = find_missing_dates(market_days, df)
market_data = fetch_data(symbol, interval, start_date, end_date)

# print tables

print("Data in Mongodb")
print(tabulate(df, headers='keys', tablefmt='github'))

print("Market Days")
print(tabulate(market_days, headers='keys', tablefmt='github'))

print(F"Missing Dates between {start_date}:{end_date} ")
print(tabulate(missing_list, headers=['Missing Dates'], tablefmt='github'))

print(F"MY BIG BEAUTIFUL TABLE FUNCTION")
print(tabulate(market_data, headers='keys', tablefmt='github'))
