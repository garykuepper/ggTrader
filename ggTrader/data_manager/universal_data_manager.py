# universal_data_manager.py
from ggTrader.data_manager.binance_data_manager import BinanceDataManager
from ggTrader.data_manager.yahoo_data_manager import YahooDataManager
from ggTrader.utils.config import PROVIDER_CAPABILITIES

class UniversalDataManager:
    def __init__(self, mongo_uri="mongodb://localhost:27017/"):
        self.mongo_uri = mongo_uri

    def get_manager(self, symbol, interval='1d', market='stock'):
        market = market.lower()
        provider = 'binance' if market == 'crypto' else 'yahoo'
        if interval not in PROVIDER_CAPABILITIES[provider]['intervals']:
            raise ValueError(f"Interval '{interval}' not supported by {provider}")

        if market == 'crypto':
            return BinanceDataManager(symbol, interval, mongo_uri=self.mongo_uri)
        elif market == 'stock':
            return YahooDataManager(symbol, interval, mongo_uri=self.mongo_uri)
        else:
            raise ValueError(f"Unsupported market type: {market}")

    def load_or_fetch(self, symbol, interval, start_date, end_date, market='stock'):
        manager = self.get_manager(symbol, interval, market)
        return manager.load_or_fetch(start_date, end_date)

    def force_update(self, symbol, interval, start_date, end_date, market='stock'):
        manager = self.get_manager(symbol, interval, market)
        return manager.force_update(start_date, end_date)
