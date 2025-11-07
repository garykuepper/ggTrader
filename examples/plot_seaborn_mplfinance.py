import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import pandas as pd

# 1. Load data (replace with your data loading)
# Assume 'df' is a pandas DataFrame with a DatetimeIndex and OHLCV columns
# For the example, we'll create dummy data
data = {
    'Open': [100, 102, 101, 105, 103],
    'High': [105, 106, 103, 108, 105],
    'Low': [98, 100, 99, 104, 101],
    'Close': [102, 104, 102, 106, 104],
    'Volume': [1000, 1200, 800, 1500, 1100]
}
dates = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'])
df = pd.DataFrame(data, index=dates)
df.index.name = 'Date'


# 2. Create the Figure and Axes objects using Matplotlib
# We want 2 rows, 1 column, sharing the x-axis for alignment
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(10, 8),
                         gridspec_kw={'height_ratios': [3, 1]}) # Top panel is 3x height of bottom

# 3. Plot the mplfinance data onto the FIRST axis (axes[0])
# Use the 'ax' argument to specify the target axis.
# Note: 'volume=True' is NOT supported in external axes mode; you must handle volume manually if needed
mpf.plot(df, type='candle', ax=axes[0], axtitle='Stock Price', show_nontrading=True)

# 4. Plot the Seaborn data onto the SECOND axis (axes[1])
# Use the 'ax' argument to specify the target axis for seaborn.
sns.histplot(data=df, x=df.index, y='Close', ax=axes[1], kde=True, bins=len(df))
axes[1].set_title('Distribution of Closing Prices (Seaborn)')
axes[1].set_xlabel('Date')

# 5. Adjust layout and display the plot
plt.tight_layout()
plt.show()

