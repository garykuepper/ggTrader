from dataclasses import dataclass
from typing import List
from enum import Enum

class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class StrategyParameters:
    ema_windows: List[int]
    trailing_stop_pct: float
    stop_loss_pct: float
    interval: str
    lookback_days: int

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
