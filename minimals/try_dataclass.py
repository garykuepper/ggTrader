from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Ticker:
    symbol: str
    ohlc_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    trailing_stop_pct: int = 0
    cooldown_period: int = 0
    hold_min_period: int = 0
    position_share_pct: int = 1


print(Ticker("SPY"))