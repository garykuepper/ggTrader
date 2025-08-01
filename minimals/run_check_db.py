from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta, date, time

# Connect to MongoDB
client = MongoClient("mongodb://flynn:qu0rra@192.168.50.44:27017")
db = client["market_data"]
collection = db["crypto_market_data"]

# Query for XRPUSDT data
symbol = "XRPUSDT"
interval = "5m"

# Check what's in the database
query = {
    "symbol": symbol,
    "interval": interval
}

# Get total count
total_count = collection.count_documents(query)
print(f"Total {symbol} {interval} records in database: {total_count}")

if total_count > 0:
    # Get date range
    oldest = collection.find(query).sort("datetime", 1).limit(1)
    newest = collection.find(query).sort("datetime", -1).limit(1)

    oldest_record = list(oldest)[0]
    newest_record = list(newest)[0]

    print(f"Date range: {oldest_record['datetime']} to {newest_record['datetime']}")

    # Show sample records
    print("\nFirst 5 records:")
    sample_records = collection.find(query).sort("datetime", 1).limit(5)
    for record in sample_records:
        print(f"  {record['datetime']}: close={record['close']}")

    print("\nLast 5 records:")
    sample_records = collection.find(query).sort("datetime", -1).limit(5)
    for record in sample_records:
        print(f"  {record['datetime']}: close={record['close']}")
else:
    print(f"No {symbol} {interval} data found in database")

# Close connection
client.close()