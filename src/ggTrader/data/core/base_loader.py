from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders (live and historical).
    Ensures a unified interface and output format for the execution engine.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbols: List[str],
        interval: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbols: List of trading pairs/symbols.
            interval: Timeframe (e.g., '1h', '4h', '1d').
            start_date: Optional start datetime (timezone aware).
            end_date: Optional end datetime (timezone aware).
            limit: Maximum number of rows to return.

        Returns:
            A MultiIndex Pandas DataFrame where columns are (symbol, dimension)
            e.g., (BTC/USD, close). Output must be fully compatible with VectorBT.
        """
        pass

    @abstractmethod
    def list_symbols(self) -> List[str]:
        """Returns a list of available symbols from this source."""
        pass
