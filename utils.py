# utils.py
from datetime import datetime, timedelta, timezone
import pandas as pd

def align_end_to_interval(now_utc: datetime, interval: str) -> datetime:
    """
    Align to the last fully closed bar end time for the given interval.
    Example: for 4h bars, returns the most recent 0,4,8,12,16,20 hour mark strictly before 'now_utc'.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    interval = str(interval).lower()
    # Map to pandas offset
    freq_map = {
        '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': 'H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
        '1d': 'D'
    }
    freq = freq_map.get(interval, 'H')
    floored = pd.Timestamp(now_utc).floor(freq).to_pydatetime().replace(tzinfo=timezone.utc)
    # Step back one full interval to ensure fully closed bar
    delta_map = {
        'T': timedelta(minutes=1),
        '5T': timedelta(minutes=5),
        '15T': timedelta(minutes=15),
        '30T': timedelta(minutes=30),
        'H': timedelta(hours=1),
        '2H': timedelta(hours=2),
        '4H': timedelta(hours=4),
        '6H': timedelta(hours=6),
        '12H': timedelta(hours=12),
        'D': timedelta(days=1),
    }
    delta = delta_map.get(freq, timedelta(hours=1))
    return floored - delta
