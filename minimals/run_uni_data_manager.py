import os
from dotenv import load_dotenv
from ggTrader.data_manager.universal_data_manager import UniversalDataManager
import matplotlib.pyplot as plt
import mplfinance as mpf

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

dm = UniversalDataManager(mongo_uri=mongo_uri)

# df = dm.load_or_fetch("BTCUSDT",
#                       "1h",
#                       "2025-07-28",
#                       "2025-07-29",
#                       market="crypto")
# # print(df.tail())

df = dm.load_or_fetch("XRPUSDT", "1d", "2025-07-01", "2025-07-29", market="crypto")
print(df.tail())
print(df.columns)
# Plot all columns
# mpf.plot(df, type='candle', style='charles', volume=True)

# Check decimal precision for each column
# for col in df.columns:
#     # Get max decimal places in the column
#     decimal_places = df[col].astype(str).str.split('.').str[1].str.len().max()
#     print(f"{col}: {decimal_places} decimal places")
#
# # Check data types
# print(f"\nData types:\n{df.dtypes}")
#
# # Check specific values with full precision
# print(f"\nFull precision sample:")
# print(df.iloc[0].to_string())