# universal_data_manager.py
from ggTrader.data_manager.binance_data_manager import BinanceDataManager
from ggTrader.data_manager.yahoo_data_manager import YahooDataManager
from ggTrader.utils.config import PROVIDER_CAPABILITIES
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

class UniversalDataManager:
    def __init__(self, mongo_uri=None):
        if mongo_uri is None:
            load_dotenv()
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

        self.mongo_uri = mongo_uri
        self.client = MongoClient(mongo_uri)
        db_name = os.getenv('DB_NAME', 'market_data')
        self.db = self.client[db_name]
        self.optimization_collection = self.db["optimization_parameters"]

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

    def save_optimization_parameters(self, symbol, strategy_name, interval, start_date, end_date, parameters):
        """
        Save or update optimization parameters for a strategy.

        Args:
            symbol: Trading symbol (e.g., 'XRPUSDT')
            strategy_name: Name of the strategy (e.g., 'ema_crossover')
            interval: Time interval (e.g., '5m', '1h')
            start_date: Start date of optimization period
            end_date: End date of optimization period
            parameters: Dict of parameters (e.g., {'ema_fast': 8, 'ema_slow': 21})
        """
        document = {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'parameters': parameters,
            'timestamp': datetime.now()
        }

        # Upsert based on unique combination
        filter_criteria = {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date
        }

        self.optimization_collection.replace_one(
            filter_criteria,
            document,
            upsert=True
        )
        print(f"Saved optimization parameters for {symbol} {strategy_name} {interval}")

    def get_latest_optimization_parameters(self, symbol, strategy_name, interval):
        """
        Get the latest optimization parameters for a symbol/strategy/interval combination.

        Returns:
            Dict with parameters or None if not found
        """
        result = self.optimization_collection.find({
            'symbol': symbol,
            'strategy_name': strategy_name,
            'interval': interval
        }).sort('timestamp', -1).limit(1)

        result_list = list(result)
        if result_list:
            return result_list[0]
        return None

    def get_optimization_parameters_by_period(self, symbol, strategy_name, interval, start_date, end_date):
        """
        Get optimization parameters for a specific time period.
        """
        result = self.optimization_collection.find_one({
            'symbol': symbol,
            'strategy_name': strategy_name,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date
        })
        return result

    def list_optimization_records(self, symbol=None, strategy_name=None, interval=None):
        """
        List all optimization records with optional filtering.
        """
        query = {}
        if symbol:
            query['symbol'] = symbol
        if strategy_name:
            query['strategy_name'] = strategy_name
        if interval:
            query['interval'] = interval

        results = self.optimization_collection.find(query).sort('timestamp', -1)
        return list(results)

    def __del__(self):
        try:
            self.close()
        except:
            pass  # Ignore any errors during cleanup

    def close(self):
        """Explicitly close the MongoDB connection"""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass