from datetime import datetime, timedelta, timezone
import pandas as pd


date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
date2 = datetime.now(timezone.utc)

start_date = date - timedelta(days=4)
end_date = date
date_range = pd.date_range(start=start_date, end=end_date, freq='1d')
print(date)
print(date2)

print(date_range)

