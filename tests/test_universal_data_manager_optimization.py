# tests/test_universal_data_manager_optimization.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from ggTrader.data_manager.universal_data_manager import UniversalDataManager


@pytest.fixture
def mock_mongo_setup():
    """Mock MongoDB setup for optimization tests"""
    with patch('ggTrader.data_manager.universal_data_manager.MongoClient') as mock_client:
        mock_db = MagicMock()
        mock_optimization_collection = MagicMock()

        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_optimization_collection

        yield mock_client, mock_db, mock_optimization_collection


@pytest.fixture
def universal_dm(mock_mongo_setup):
    """Create UniversalDataManager instance for testing"""
    mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

    with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017/', 'DB_NAME': 'test_db'}):
        dm = UniversalDataManager()
        dm.optimization_collection = mock_optimization_collection
        return dm


class TestOptimizationParameterSaving:
    """Test saving optimization parameters"""

    def test_save_optimization_parameters_success(self, universal_dm, mock_mongo_setup):
        """Test successful saving of optimization parameters"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        # Test data
        symbol = "XRPUSDT"
        strategy_name = "ema_crossover"
        interval = "5m"
        start_date = "2023-01-01"
        end_date = "2023-01-07"
        parameters = {"ema_fast": 8, "ema_slow": 21}

        # Execute
        universal_dm.save_optimization_parameters(
            symbol, strategy_name, interval, start_date, end_date, parameters
        )

        # Verify replace_one was called with correct parameters
        mock_optimization_collection.replace_one.assert_called_once()
        args = mock_optimization_collection.replace_one.call_args

        filter_criteria = args[0][0]
        document = args[0][1]

        # Check filter criteria
        assert filter_criteria['symbol'] == symbol
        assert filter_criteria['strategy_name'] == strategy_name
        assert filter_criteria['interval'] == interval
        assert filter_criteria['start_date'] == start_date
        assert filter_criteria['end_date'] == end_date

        # Check document
        assert document['symbol'] == symbol
        assert document['strategy_name'] == strategy_name
        assert document['parameters'] == parameters
        assert 'timestamp' in document
        assert args[1]['upsert'] is True

    def test_save_optimization_parameters_different_strategies(self, universal_dm, mock_mongo_setup):
        """Test saving parameters for different strategies"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        # Save EMA parameters
        universal_dm.save_optimization_parameters(
            "BTCUSDT", "ema_crossover", "1h", "2023-01-01", "2023-01-07",
            {"ema_fast": 12, "ema_slow": 26}
        )

        # Save RSI parameters
        universal_dm.save_optimization_parameters(
            "BTCUSDT", "rsi_strategy", "1h", "2023-01-01", "2023-01-07",
            {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30}
        )

        # Should be called twice
        assert mock_optimization_collection.replace_one.call_count == 2


class TestOptimizationParameterRetrieval:
    """Test retrieving optimization parameters"""

    def test_get_latest_optimization_parameters_found(self, universal_dm, mock_mongo_setup):
        """Test getting latest parameters when they exist"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        # Mock return data
        mock_result = {
            'symbol': 'XRPUSDT',
            'strategy_name': 'ema_crossover',
            'interval': '5m',
            'parameters': {'ema_fast': 8, 'ema_slow': 21},
            'timestamp': datetime.now()
        }

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [mock_result]
        mock_optimization_collection.find.return_value = mock_cursor

        # Execute
        result = universal_dm.get_latest_optimization_parameters('XRPUSDT', 'ema_crossover', '5m')

        # Verify query
        expected_query = {
            'symbol': 'XRPUSDT',
            'strategy_name': 'ema_crossover',
            'interval': '5m'
        }
        mock_optimization_collection.find.assert_called_with(expected_query)
        mock_cursor.sort.assert_called_with('timestamp', -1)
        mock_cursor.sort.return_value.limit.assert_called_with(1)

        # Check result
        assert result == mock_result
        assert result['parameters']['ema_fast'] == 8

    def test_get_latest_optimization_parameters_not_found(self, universal_dm, mock_mongo_setup):
        """Test getting latest parameters when none exist"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = []
        mock_optimization_collection.find.return_value = mock_cursor

        result = universal_dm.get_latest_optimization_parameters('NONEXISTENT', 'strategy', '1h')

        assert result is None

    def test_get_optimization_parameters_by_period_found(self, universal_dm, mock_mongo_setup):
        """Test getting parameters by specific period"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_result = {
            'symbol': 'BTCUSDT',
            'strategy_name': 'ema_crossover',
            'interval': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-01-07',
            'parameters': {'ema_fast': 12, 'ema_slow': 26}
        }
        mock_optimization_collection.find_one.return_value = mock_result

        result = universal_dm.get_optimization_parameters_by_period(
            'BTCUSDT', 'ema_crossover', '1h', '2023-01-01', '2023-01-07'
        )

        expected_query = {
            'symbol': 'BTCUSDT',
            'strategy_name': 'ema_crossover',
            'interval': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-01-07'
        }
        mock_optimization_collection.find_one.assert_called_with(expected_query)
        assert result == mock_result

    def test_get_optimization_parameters_by_period_not_found(self, universal_dm, mock_mongo_setup):
        """Test getting parameters by period when none exist"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_optimization_collection.find_one.return_value = None

        result = universal_dm.get_optimization_parameters_by_period(
            'NONEXISTENT', 'strategy', '1h', '2023-01-01', '2023-01-07'
        )

        assert result is None


class TestOptimizationParameterListing:
    """Test listing optimization parameters"""

    def test_list_optimization_records_no_filter(self, universal_dm, mock_mongo_setup):
        """Test listing all optimization records"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_records = [
            {'symbol': 'XRPUSDT', 'strategy_name': 'ema_crossover', 'interval': '5m'},
            {'symbol': 'BTCUSDT', 'strategy_name': 'rsi_strategy', 'interval': '1h'}
        ]

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_records
        mock_optimization_collection.find.return_value = mock_cursor

        result = universal_dm.list_optimization_records()

        mock_optimization_collection.find.assert_called_with({})
        mock_cursor.sort.assert_called_with('timestamp', -1)
        assert result == mock_records

    def test_list_optimization_records_with_symbol_filter(self, universal_dm, mock_mongo_setup):
        """Test listing records filtered by symbol"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_records = [
            {'symbol': 'XRPUSDT', 'strategy_name': 'ema_crossover', 'interval': '5m'}
        ]

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_records
        mock_optimization_collection.find.return_value = mock_cursor

        result = universal_dm.list_optimization_records(symbol='XRPUSDT')

        expected_query = {'symbol': 'XRPUSDT'}
        mock_optimization_collection.find.assert_called_with(expected_query)
        assert result == mock_records

    def test_list_optimization_records_with_multiple_filters(self, universal_dm, mock_mongo_setup):
        """Test listing records with multiple filters"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = []
        mock_optimization_collection.find.return_value = mock_cursor

        universal_dm.list_optimization_records(
            symbol='BTCUSDT',
            strategy_name='ema_crossover',
            interval='1h'
        )

        expected_query = {
            'symbol': 'BTCUSDT',
            'strategy_name': 'ema_crossover',
            'interval': '1h'
        }
        mock_optimization_collection.find.assert_called_with(expected_query)


class TestOptimizationIntegration:
    """Integration tests for optimization functionality"""

    def test_save_and_retrieve_workflow(self, universal_dm, mock_mongo_setup):
        """Test complete save and retrieve workflow"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        # Step 1: Save parameters
        symbol = "ETHUSDT"
        strategy_name = "ema_crossover"
        interval = "15m"
        start_date = "2023-01-01"
        end_date = "2023-01-07"
        parameters = {"ema_fast": 10, "ema_slow": 30}

        universal_dm.save_optimization_parameters(
            symbol, strategy_name, interval, start_date, end_date, parameters
        )

        # Verify save was called
        mock_optimization_collection.replace_one.assert_called_once()

        # Step 2: Mock retrieval
        mock_result = {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'interval': interval,
            'parameters': parameters,
            'timestamp': datetime.now()
        }

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [mock_result]
        mock_optimization_collection.find.return_value = mock_cursor

        # Step 3: Retrieve parameters
        retrieved = universal_dm.get_latest_optimization_parameters(symbol, strategy_name, interval)

        assert retrieved['parameters'] == parameters

    def test_multiple_strategies_same_symbol(self, universal_dm, mock_mongo_setup):
        """Test handling multiple strategies for same symbol"""
        mock_client, mock_db, mock_optimization_collection = mock_mongo_setup

        symbol = "ADAUSDT"
        interval = "1h"

        # Save EMA strategy
        universal_dm.save_optimization_parameters(
            symbol, "ema_crossover", interval, "2023-01-01", "2023-01-07",
            {"ema_fast": 8, "ema_slow": 21}
        )

        # Save RSI strategy
        universal_dm.save_optimization_parameters(
            symbol, "rsi_strategy", interval, "2023-01-01", "2023-01-07",
            {"rsi_period": 14, "rsi_threshold": 30}
        )

        # Both should be saved independently
        assert mock_optimization_collection.replace_one.call_count == 2


# Standalone test script you can run directly
def test_optimization_manually():
    """Manual test script you can run to test with real database"""
    from ggTrader.data_manager.universal_data_manager import UniversalDataManager

    print("Testing optimization parameter functionality...")

    # Initialize manager
    dm = UniversalDataManager()

    # Test data
    symbol = "XRPUSDT"
    strategy_name = "ema_crossover"
    interval = "5m"
    start_date = "2023-01-01"
    end_date = "2023-01-07"
    parameters = {"ema_fast": 8, "ema_slow": 21}

    # Save parameters
    print(f"Saving parameters for {symbol}...")
    dm.save_optimization_parameters(
        symbol, strategy_name, interval, start_date, end_date, parameters
    )

    # Retrieve latest parameters
    print(f"Retrieving latest parameters for {symbol}...")
    latest = dm.get_latest_optimization_parameters(symbol, strategy_name, interval)
    print(f"Latest parameters: {latest}")

    # Retrieve by period
    print(f"Retrieving parameters by period...")
    by_period = dm.get_optimization_parameters_by_period(
        symbol, strategy_name, interval, start_date, end_date
    )
    print(f"Parameters by period: {by_period}")

    # List all records for symbol
    print(f"Listing all records for {symbol}...")
    records = dm.list_optimization_records(symbol=symbol)
    print(f"Found {len(records)} records")
    for record in records:
        print(f"  {record['strategy_name']} - {record['parameters']}")


if __name__ == "__main__":
    test_optimization_manually()