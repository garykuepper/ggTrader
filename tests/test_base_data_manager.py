import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch
from old.ggTrader_old.data_manager.base_data_manager import DataManager


class ConcreteDataManager(DataManager):
    """Concrete implementation for testing the abstract DataManager class"""

    def fetch(self, start_date, end_date):
        """Mock implementation of the abstract fetch method"""
        # Return sample data for testing
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        data = {
            'open': [100.0] * len(dates),
            'high': [105.0] * len(dates),
            'low': [95.0] * len(dates),
            'close': [102.0] * len(dates),
            'volume': [1000.0] * len(dates)
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'datetime'
        return df


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client to avoid actual database connections"""
    with patch('ggTrader_old.data_manager.base_data_manager.MongoClient') as mock_client:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        yield mock_client, mock_db, mock_collection


@pytest.fixture
def data_manager(mock_mongo_client):
    """Create a test data manager instance"""
    mock_client, mock_db, mock_collection = mock_mongo_client
    dm = ConcreteDataManager(
        symbol="BTCUSDT",
        interval="1h",
        market="crypto",
        provider="test_provider"
    )
    return dm


class TestDataManagerInitialization:
    """Test DataManager initialization and setup"""

    def test_initialization_with_defaults(self, mock_mongo_client):
        """Test DataManager initialization with default parameters"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        dm = ConcreteDataManager("btcusdt", "1h", "crypto", "test")

        assert dm.symbol == "BTCUSDT"  # Should be uppercase
        assert dm.interval == "1h"
        assert dm.market == "crypto"
        assert dm.provider == "test"

        # Verify MongoDB setup
        mock_client.assert_called_once_with("mongodb://localhost:27017/")
        mock_collection.create_index.assert_called_once()

    def test_initialization_with_custom_mongo_uri(self, mock_mongo_client):
        """Test DataManager initialization with custom MongoDB URI"""
        mock_client, mock_db, mock_collection = mock_mongo_client
        custom_uri = "mongodb://custom:27017/"

        dm = ConcreteDataManager("ETH", "5m", "crypto", "test", custom_uri)

        mock_client.assert_called_once_with(custom_uri)

    def test_symbol_case_conversion(self, mock_mongo_client):
        """Test that symbols are converted to uppercase"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        dm = ConcreteDataManager("btcusdt", "1h", "crypto", "test")
        assert dm.symbol == "BTCUSDT"


class TestDatabaseOperations:
    """Test database save and load operations"""

    def test_save_to_db_with_datetime_index(self, data_manager, mock_mongo_client):
        """Test saving DataFrame with datetime index"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Create test DataFrame with datetime index
        dates = pd.date_range('2023-01-01', periods=3, freq='1h')
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=dates)
        df.index.name = 'datetime'

        data_manager.save_to_db(df)

        # Verify insert_many was called
        mock_collection.insert_many.assert_called_once()
        args = mock_collection.insert_many.call_args[0][0]

        # Check that records have required fields
        assert len(args) == 3
        for record in args:
            assert 'symbol' in record
            assert 'interval' in record
            assert 'provider' in record
            assert 'datetime' in record

    def test_save_to_db_with_multiindex_columns(self, data_manager, mock_mongo_client):
        """Test saving DataFrame with MultiIndex columns"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Create DataFrame with MultiIndex columns (like some financial data)
        dates = pd.date_range('2023-01-01', periods=2, freq='1h')
        columns = pd.MultiIndex.from_tuples([('OHLCV', 'open'), ('OHLCV', 'close')])
        df = pd.DataFrame([[100, 102], [101, 103]], index=dates, columns=columns)
        df.index.name = 'datetime'

        data_manager.save_to_db(df)

        mock_collection.insert_many.assert_called_once()

    def test_save_to_db_empty_dataframe(self, data_manager, mock_mongo_client):
        """Test saving empty DataFrame"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        df = pd.DataFrame()
        data_manager.save_to_db(df)

        # Should not call insert_many for empty DataFrame
        mock_collection.insert_many.assert_not_called()

    def test_save_to_db_handles_date_column(self, data_manager, mock_mongo_client):
        """Test saving DataFrame with 'Date' column (Yahoo Finance format)"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Create DataFrame that mimics Yahoo Finance data after reset_index
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'open': [100, 101],
            'close': [102, 103]
        })
        df['Date'] = pd.to_datetime(df['Date'])

        data_manager.save_to_db(df)

        mock_collection.insert_many.assert_called_once()
        args = mock_collection.insert_many.call_args[0][0]

        # Check that Date was converted to datetime
        for record in args:
            assert 'datetime' in record
            assert 'Date' not in record

    def test_load_from_db(self, data_manager, mock_mongo_client):
        """Test loading data from database"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Mock database response
        mock_data = [
            {
                'datetime': datetime(2023, 1, 1, 10, 0),
                'open': 100,
                'high': 105,
                'low': 95,
                'close': 102,
                'volume': 1000,
                'symbol': 'BTCUSDT',
                'interval': '1h'
            }
        ]
        mock_collection.find.return_value = mock_data

        start_date = "2023-01-01"
        end_date = "2023-01-02"

        df = data_manager.load_from_db(start_date, end_date)

        # Verify query was made correctly
        expected_query = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "datetime": {
                "$gte": pd.to_datetime(start_date),
                "$lte": pd.to_datetime(end_date)
            }
        }
        mock_collection.find.assert_called_once_with(expected_query)

        # Verify DataFrame structure
        assert not df.empty
        assert df.index.name == 'datetime'
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_load_from_db_empty_result(self, data_manager, mock_mongo_client):
        """Test loading from database when no data exists"""
        mock_client, mock_db, mock_collection = mock_mongo_client
        mock_collection.find.return_value = []

        df = data_manager.load_from_db("2023-01-01", "2023-01-02")

        assert df.empty


class TestTradingDays:
    """Test trading days calculation"""

    @patch('ggTrader_old.data_manager.base_data_manager.mcal')
    def test_get_trading_days_stock_market(self, mock_mcal, data_manager):
        """Test getting trading days for stock market"""
        # Mock NYSE calendar
        mock_calendar = MagicMock()
        mock_schedule = pd.DataFrame(index=pd.date_range('2023-01-02', '2023-01-04'))  # Weekdays only
        mock_calendar.schedule.return_value = mock_schedule
        mock_mcal.get_calendar.return_value = mock_calendar

        data_manager.market = 'stock'

        trading_days = data_manager._get_trading_days('2023-01-01', '2023-01-05')

        mock_mcal.get_calendar.assert_called_once_with('NYSE')
        mock_calendar.schedule.assert_called_once_with(
            start_date='2023-01-01',
            end_date='2023-01-05'
        )

        assert len(trading_days) == 3  # 3 weekdays

    def test_get_trading_days_crypto_market(self, data_manager):
        """Test getting trading days for crypto market (all days)"""
        data_manager.market = 'crypto'

        trading_days = data_manager._get_trading_days('2023-01-01', '2023-01-03')

        # Crypto markets trade every day
        assert len(trading_days) == 3


class TestMissingRanges:
    """Test missing data range detection"""

    def test_get_missing_ranges_no_existing_data(self, data_manager, mock_mongo_client):
        """Test missing ranges when no data exists"""
        mock_client, mock_db, mock_collection = mock_mongo_client
        mock_collection.find.return_value = []

        data_manager.market = 'crypto'
        data_manager.interval = '1h'

        missing_ranges = data_manager.get_missing_ranges('2023-01-01', '2023-01-02')

        assert len(missing_ranges) == 1
        start, end = missing_ranges[0]
        assert start == pd.to_datetime('2023-01-01')
        assert end == pd.to_datetime('2023-01-02')

    def test_get_missing_ranges_no_missing_data(self, data_manager, mock_mongo_client):
        """Test missing ranges when all data exists"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Mock complete existing data
        dates = pd.date_range('2023-01-01', '2023-01-02', freq='1h')
        mock_data = [
            {'datetime': date, 'open': 100, 'high': 105, 'low': 95, 'close': 102, 'volume': 1000}
            for date in dates
        ]
        mock_collection.find.return_value = mock_data

        data_manager.interval = '1h'

        missing_ranges = data_manager.get_missing_ranges('2023-01-01', '2023-01-02')

        assert len(missing_ranges) == 0

    @patch('ggTrader_old.data_manager.base_data_manager.mcal')
    def test_get_missing_ranges_stock_daily_no_trading_days(self, mock_mcal, data_manager, mock_mongo_client):
        """Test missing ranges for stock daily data when no trading days exist"""
        mock_client, mock_db, mock_collection = mock_mongo_client
        mock_collection.find.return_value = []

        # Mock no trading days (weekend)
        mock_calendar = MagicMock()
        mock_schedule = pd.DataFrame(index=pd.DatetimeIndex([]))  # No trading days
        mock_calendar.schedule.return_value = mock_schedule
        mock_mcal.get_calendar.return_value = mock_calendar

        data_manager.market = 'stock'
        data_manager.interval = '1d'

        missing_ranges = data_manager.get_missing_ranges('2023-01-07', '2023-01-08')  # Weekend

        assert len(missing_ranges) == 0  # No trading days, so no missing ranges


class TestStaticMethods:
    """Test static utility methods"""

    def test_convert_to_datetime_string_with_time(self):
        """Test datetime conversion from string with time"""
        result = DataManager._convert_to_datetime("2023-01-01 10:30:00")
        expected = datetime(2023, 1, 1, 10, 30, 0)
        assert result == expected

    def test_convert_to_datetime_string_date_only(self):
        """Test datetime conversion from date-only string"""
        result = DataManager._convert_to_datetime("2023-01-01")
        expected = datetime(2023, 1, 1, 0, 0, 0)
        assert result == expected

    def test_convert_to_datetime_datetime_object(self):
        """Test datetime conversion from datetime object"""
        input_dt = datetime(2023, 1, 1, 10, 30, 0)
        result = DataManager._convert_to_datetime(input_dt)
        assert result == input_dt

    def test_convert_to_datetime_invalid_format(self):
        """Test datetime conversion with invalid input"""
        with pytest.raises(ValueError, match="Invalid date format"):
            DataManager._convert_to_datetime(12345)

    def test_parse_interval_to_minutes_minutes(self):
        """Test interval parsing for minutes"""
        assert DataManager._parse_interval_to_minutes("5m") == 5
        assert DataManager._parse_interval_to_minutes("15m") == 15

    def test_parse_interval_to_minutes_hours(self):
        """Test interval parsing for hours"""
        assert DataManager._parse_interval_to_minutes("1h") == 60
        assert DataManager._parse_interval_to_minutes("4h") == 240

    def test_parse_interval_to_minutes_days(self):
        """Test interval parsing for days"""
        assert DataManager._parse_interval_to_minutes("1d") == 1440
        assert DataManager._parse_interval_to_minutes("7d") == 10080

    def test_parse_interval_to_minutes_weeks(self):
        """Test interval parsing for weeks"""
        assert DataManager._parse_interval_to_minutes("1w") == 10080
        assert DataManager._parse_interval_to_minutes("2w") == 20160

    def test_parse_interval_to_minutes_invalid(self):
        """Test interval parsing with invalid format"""
        with pytest.raises(ValueError, match="Unsupported interval format"):
            DataManager._parse_interval_to_minutes("invalid")


class TestLoadOrFetch:
    """Test the main load_or_fetch functionality"""

    def test_load_or_fetch_with_missing_data(self, data_manager, mock_mongo_client):
        """Test load_or_fetch when some data is missing"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Mock partial existing data
        existing_date = datetime(2023, 1, 1, 10, 0)
        mock_data = [{
            'datetime': existing_date,
            'open': 100, 'high': 105, 'low': 95, 'close': 102, 'volume': 1000
        }]
        mock_collection.find.return_value = mock_data

        data_manager.interval = '1h'

        result = data_manager.load_or_fetch('2023-01-01', '2023-01-01 12:00:00')

        # Should return combined existing and fetched data
        assert not result.empty
        assert len(result) > 1  # Should include both existing and newly fetched data

    def test_force_update(self, data_manager):
        """Test force update functionality"""
        result = data_manager.force_update('2023-01-01', '2023-01-02')

        assert not result.empty
        assert len(result.columns) == 5  # OHLCV columns


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_save_to_db_handles_duplicate_key_error(self, data_manager, mock_mongo_client):
        """Test that duplicate key errors are handled gracefully"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Mock duplicate key error
        duplicate_error = Exception("duplicate key error")
        mock_collection.insert_many.side_effect = duplicate_error

        dates = pd.date_range('2023-01-01', periods=1, freq='1h')
        df = pd.DataFrame({'open': [100], 'close': [102]}, index=dates)
        df.index.name = 'datetime'

        # Should not raise exception
        data_manager.save_to_db(df)

    def test_save_to_db_handles_other_errors(self, data_manager, mock_mongo_client):
        """Test that non-duplicate errors are handled properly"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Mock other type of error
        other_error = Exception("connection error")
        mock_collection.insert_many.side_effect = other_error

        dates = pd.date_range('2023-01-01', periods=1, freq='1h')
        df = pd.DataFrame({'open': [100], 'close': [102]}, index=dates)
        df.index.name = 'datetime'

        # Should not raise exception but should print error
        with patch('builtins.print') as mock_print:
            data_manager.save_to_db(df)
            mock_print.assert_called_once()


class TestFrequencyMapping:
    """Test the frequency mapping for deprecated pandas aliases"""

    def test_frequency_mapping_in_missing_ranges(self, data_manager, mock_mongo_client):
        """Test that deprecated frequency aliases are properly mapped"""
        mock_client, mock_db, mock_collection = mock_mongo_client
        mock_collection.find.return_value = []

        data_manager.market = 'crypto'  # Not stock
        data_manager.interval = 'm'  # Deprecated alias

        missing_ranges = data_manager.get_missing_ranges('2023-01-01', '2023-01-02')

        # For empty data with non-stock market, should return the full range
        assert len(missing_ranges) == 1
        start, end = missing_ranges[0]
        assert start == pd.to_datetime('2023-01-01')
        assert end == pd.to_datetime('2023-01-02')


# Integration test
class TestIntegration:
    """Integration tests that test multiple components together"""

    def test_complete_workflow(self, mock_mongo_client):
        """Test complete workflow from initialization to data retrieval"""
        mock_client, mock_db, mock_collection = mock_mongo_client

        # Setup
        dm = ConcreteDataManager("BTCUSDT", "1h", "crypto", "test")

        # Mock no existing data initially
        mock_collection.find.return_value = []

        # Execute load_or_fetch
        result = dm.load_or_fetch('2023-01-01', '2023-01-01 02:00:00')

        # Verify results
        assert not result.empty
        assert result.index.name == 'datetime'
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']

        # Verify database operations were called
        mock_collection.find.assert_called()
        mock_collection.insert_many.assert_called()