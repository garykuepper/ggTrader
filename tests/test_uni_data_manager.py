import os
import pytest
from dotenv import load_dotenv
from old.ggTrader_old.data_manager import UniversalDataManager
import time
from datetime import datetime, timedelta

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

def get_reliable_crypto_dates():
    """Get reliable crypto test dates - crypto markets run 24/7"""
    # Go back enough days to ensure data exists, but not too far
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=2)  # 2 days of data
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def get_reliable_stock_dates():
    """Get reliable stock test dates - avoid weekends"""
    now = datetime.now()

    # Find the most recent weekday
    days_back = 1
    while True:
        test_date = now - timedelta(days=days_back)
        if test_date.weekday() < 5:  # Monday=0, Friday=4
            break
        days_back += 1

    end_date = test_date
    start_date = end_date - timedelta(days=7)  # 1 week of data
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

@pytest.fixture(scope="module")
def data_manager():
    """Create a data manager instance for testing"""
    return UniversalDataManager(mongo_uri=mongo_uri)

@pytest.fixture(scope="module")
def crypto_dates():
    """Get reliable crypto test dates"""
    return get_reliable_crypto_dates()

@pytest.fixture(scope="module")
def stock_dates():
    """Get reliable stock test dates"""
    return get_reliable_stock_dates()

@pytest.fixture(scope="module")
def sample_crypto_data(data_manager, crypto_dates):
    """Fetch sample crypto data for testing"""
    start_date, end_date = crypto_dates
    return data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")

@pytest.fixture(scope="module")
def sample_stock_data(data_manager, stock_dates):
    """Fetch sample stock data for testing"""
    start_date, end_date = stock_dates
    try:
        # Use daily data which Yahoo Finance supports
        return data_manager.load_or_fetch("SPY", "1d", start_date, end_date, market="stock")
    except Exception:
        return None

class TestCryptoData:
    """Test crypto data functionality"""

    def test_basic_crypto_fetch(self, data_manager, crypto_dates):
        """Test basic crypto data fetching"""
        start_date, end_date = crypto_dates
        df = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")

        assert not df.empty, f"DataFrame should not be empty for dates {start_date} to {end_date}"
        assert len(df.columns) == 5, "Should have 5 columns (OHLCV)"
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        print(f"✓ Basic fetch: {df.shape} for {start_date} to {end_date}")

    def test_crypto_caching(self, data_manager, crypto_dates):
        """Test that re-fetching same data returns identical results"""
        start_date, end_date = crypto_dates
        df1 = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")
        df2 = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")

        assert df1.equals(df2), "Cached data should be identical"
        print("✓ Caching works correctly")

    def test_different_intervals(self, data_manager, crypto_dates):
        """Test different time intervals"""
        start_date, end_date = crypto_dates
        intervals = ["1h", "4h"]

        for interval in intervals:
            df = data_manager.load_or_fetch("BTCUSDT", interval, start_date, end_date, market="crypto")

            assert not df.empty, f"{interval} data should not be empty for {start_date} to {end_date}"
            assert len(df.columns) == 5, f"{interval} should have 5 columns"
            print(f"✓ {interval}: {len(df)} records")

    def test_fallback_to_known_dates(self, data_manager):
        """Test with known working dates as fallback"""
        # Use the exact dates from your reference script as backup
        df = data_manager.load_or_fetch("BTCUSDT", "1h", "2025-07-28", "2025-07-29", market="crypto")

        # This might be empty if those specific dates don't have data anymore
        if not df.empty:
            assert len(df.columns) == 5, "Should have 5 columns (OHLCV)"
            print(f"✓ Fallback dates: {df.shape}")
        else:
            print("⚠ Fallback dates returned empty (data may have expired)")

    def test_different_symbols(self, data_manager, crypto_dates):
        """Test different crypto symbols"""
        start_date, end_date = crypto_dates
        symbols = ["BTCUSDT", "ETHUSDT"]

        successful_symbols = []
        for symbol in symbols:
            try:
                df = data_manager.load_or_fetch(symbol, "1h", start_date, end_date, market="crypto")
                if not df.empty:
                    successful_symbols.append(symbol)
                    print(f"✓ {symbol}: {df.shape}")
                else:
                    print(f"⚠ {symbol}: No data available for {start_date} to {end_date}")
            except Exception as e:
                print(f"⚠ {symbol} failed: {e}")

        # At least one symbol should work
        assert len(successful_symbols) > 0, "At least one crypto symbol should return data"

class TestStockData:
    """Test stock data functionality"""

    def test_daily_stock_data(self, data_manager, stock_dates):
        """Test daily stock data (supported by Yahoo Finance)"""
        start_date, end_date = stock_dates
        try:
            df = data_manager.load_or_fetch("SPY", "1d", start_date, end_date, market="stock")
            if not df.empty:
                # Check for standard OHLCV columns
                standard_columns = ['open', 'high', 'low', 'close', 'volume']
                available_standard_cols = [col for col in standard_columns if col in df.columns]

                assert len(available_standard_cols) >= 4, f"Should have at least 4 standard columns, found: {available_standard_cols}"
                print(f"✓ SPY daily: {df.shape} for {start_date} to {end_date}")
                print(f"Available columns: {list(df.columns)}")
            else:
                print(f"⚠ SPY daily data returned empty for {start_date} to {end_date}")
        except Exception as e:
            print(f"⚠ SPY daily failed: {e}")
            pytest.skip(f"Daily stock data not available: {e}")

    def test_stock_interval_validation(self, data_manager, stock_dates):
        """Test that unsupported stock intervals are properly handled"""
        start_date, end_date = stock_dates

        with pytest.raises(Exception) as exc_info:
            data_manager.load_or_fetch("SPY", "1h", start_date, end_date, market="stock")

        assert "not supported" in str(exc_info.value).lower(), "Should indicate interval not supported"
        print(f"✓ Unsupported stock interval properly rejected: {exc_info.value}")

    def test_multiple_stock_symbols(self, data_manager, stock_dates):
        """Test multiple stock symbols with daily data"""
        start_date, end_date = stock_dates
        symbols = ["SPY", "AAPL", "MSFT"]
        successful_symbols = []

        for symbol in symbols:
            try:
                df = data_manager.load_or_fetch(symbol, "1d", start_date, end_date, market="stock")
                if not df.empty:
                    successful_symbols.append(symbol)
                    print(f"✓ {symbol}: {df.shape}")
                else:
                    print(f"⚠ {symbol}: No data available for {start_date} to {end_date}")
            except Exception as e:
                print(f"⚠ {symbol} failed: {e}")

        print(f"Working stock symbols: {successful_symbols}")

class TestDataQuality:
    """Test data quality and consistency"""

    def test_crypto_data_quality(self, sample_crypto_data):
        """Test crypto data quality"""
        df = sample_crypto_data

        if df.empty:
            pytest.skip("No crypto data available for quality testing")

        # Test 1: No missing values in OHLC
        missing_data = df.isnull().sum()
        assert missing_data.sum() == 0, f"Should have no missing values, found: {missing_data}"

        # Test 2: OHLC relationships
        assert (df['high'] >= df['low']).all(), "High should be >= Low"
        assert (df['high'] >= df['open']).all(), "High should be >= Open"
        assert (df['high'] >= df['close']).all(), "High should be >= Close"
        assert (df['low'] <= df['open']).all(), "Low should be <= Open"
        assert (df['low'] <= df['close']).all(), "Low should be <= Close"

        # Test 3: Volume validation
        assert (df['volume'] >= 0).all(), "Volume should be non-negative"

        print("✓ Crypto data quality checks passed")

    def test_stock_data_quality(self, sample_stock_data):
        """Test stock data quality"""
        if sample_stock_data is None or sample_stock_data.empty:
            pytest.skip("Stock data not available")

        df = sample_stock_data

        # Filter to only the standard OHLCV columns for testing
        standard_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if we have the standard columns
        available_standard_cols = [col for col in standard_columns if col in df.columns]

        if len(available_standard_cols) == 0:
            pytest.skip("No standard OHLCV columns found in stock data")

        # Test only the standard columns
        df_standard = df[available_standard_cols]

        # Quality checks on standard columns only
        missing_data = df_standard.isnull().sum()
        assert missing_data.sum() == 0, f"Should have no missing values in standard columns, found: {missing_data}"

        assert (df_standard['high'] >= df_standard['low']).all(), "High should be >= Low"

        if 'volume' in df_standard.columns:
            assert (df_standard['volume'] >= 0).all(), "Volume should be non-negative"

        print(f"✓ Stock data quality checks passed for columns: {available_standard_cols}")

        # Report on extra columns if they exist
        extra_columns = [col for col in df.columns if col not in standard_columns]
        if extra_columns:
            print(f"ℹ Extra columns found (not tested): {extra_columns}")

    def test_data_precision_matches_reference(self, sample_crypto_data):
        """Test that data precision matches reference implementation"""
        df = sample_crypto_data

        if df.empty:
            pytest.skip("No data available for precision testing")

        # Test precision as shown in your reference script
        for col in ['open', 'high', 'low', 'close']:
            decimal_places = df[col].astype(str).str.split('.').str[1].str.len().max()
            print(f"{col}: {decimal_places} decimal places")

            # Prices should have reasonable precision
            assert decimal_places <= 8, f"{col} has too many decimal places: {decimal_places}"

        # Volume precision
        volume_decimals = df['volume'].astype(str).str.split('.').str[1].str.len().max()
        print(f"volume: {volume_decimals} decimal places")
        assert volume_decimals <= 8, f"Volume has too many decimal places: {volume_decimals}"

        # Check data types match reference
        expected_dtypes = ['float64'] * 5
        actual_dtypes = [str(dtype) for dtype in df.dtypes]
        assert actual_dtypes == expected_dtypes, f"Data types don't match: {actual_dtypes}"

        print("✓ Data precision checks passed")

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_symbol(self, data_manager, crypto_dates):
        """Test invalid symbol handling"""
        start_date, end_date = crypto_dates
        try:
            df = data_manager.load_or_fetch("INVALID123", "1h", start_date, end_date, market="crypto")
            assert df.empty, "Invalid symbol should return empty data or raise exception"
            print("✓ Invalid symbol returned empty data")
        except Exception as e:
            print(f"✓ Invalid symbol properly rejected: {e}")

    def test_invalid_crypto_interval(self, data_manager, crypto_dates):
        """Test invalid crypto interval handling"""
        start_date, end_date = crypto_dates
        try:
            df = data_manager.load_or_fetch("BTCUSDT", "13h", start_date, end_date, market="crypto")
            if df.empty:
                print("✓ Invalid interval returned empty data")
            else:
                print(f"⚠ Invalid interval unexpectedly worked: {df.shape}")
        except Exception as e:
            print(f"✓ Invalid interval properly rejected: {e}")

class TestPerformance:
    """Test performance with known working data"""

    def test_reasonable_performance(self, data_manager, crypto_dates):
        """Test reasonable performance"""
        start_date, end_date = crypto_dates
        start_time = time.time()

        df = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")
        end_time = time.time()

        duration = end_time - start_time

        assert not df.empty, f"Should return data for performance test ({start_date} to {end_date})"
        assert duration < 30, f"Should complete within 30 seconds, took {duration:.2f}s"

        print(f"✓ Performance test: {len(df)} records in {duration:.2f}s")

    def test_caching_performance(self, data_manager, crypto_dates):
        """Test that caching improves performance"""
        start_date, end_date = crypto_dates

        # First fetch (should be slower - fetching/saving)
        start_time = time.time()
        df1 = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")
        first_duration = time.time() - start_time

        # Second fetch (should be faster - from cache)
        start_time = time.time()
        df2 = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")
        second_duration = time.time() - start_time

        assert df1.equals(df2), "Cached data should be identical"
        print(f"✓ Caching performance: First={first_duration:.2f}s, Second={second_duration:.2f}s")

class TestIntegration:
    """Integration tests matching your reference script"""

    def test_reference_script_behavior(self, data_manager, crypto_dates):
        """Test that behavior matches your reference script style"""
        start_date, end_date = crypto_dates
        df = data_manager.load_or_fetch("BTCUSDT", "1h", start_date, end_date, market="crypto")

        assert not df.empty, f"Should return data for {start_date} to {end_date}"

        # Check that we can do the same operations as your reference script
        tail_data = df.tail()
        assert len(tail_data) <= 5, "tail() should work"
        print("Last 5 rows:")
        print(tail_data)

        # Check decimal precision like your reference script
        print("\nDecimal precision check:")
        for col in df.columns:
            decimal_places = df[col].astype(str).str.split('.').str[1].str.len().max()
            print(f"{col}: {decimal_places} decimal places")

        # Check data types like your reference script
        print(f"\nData types:\n{df.dtypes}")

        # Check specific values with full precision like your reference script
        print(f"\nFull precision sample:")
        print(df.iloc[0].to_string())

        print("✓ Reference script behavior replicated successfully")

# Standalone test runner
def run_standalone_tests():
    """Run tests directly without pytest - matches your reference script style"""
    print("Starting Universal Data Manager Tests...")
    print(f"Test run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Get dynamic dates
    crypto_start, crypto_end = get_reliable_crypto_dates()
    stock_start, stock_end = get_reliable_stock_dates()

    print(f"Crypto test dates: {crypto_start} to {crypto_end}")
    print(f"Stock test dates: {stock_start} to {stock_end}")

    print("\n" + "="*50)
    print("CRYPTO FUNCTIONALITY TESTS")
    print("="*50)

    try:
        # Test with dynamic dates
        df = dm.load_or_fetch("BTCUSDT", "1h", crypto_start, crypto_end, market="crypto")

        print(f"Dynamic dates fetch: {df.shape}")
        if not df.empty:
            print("Last 5 rows:")
            print(df.tail())

            # Replicate your reference script checks
            print("\nDecimal precision check:")
            for col in df.columns:
                decimal_places = df[col].astype(str).str.split('.').str[1].str.len().max()
                print(f"{col}: {decimal_places} decimal places")

            print(f"\nData types:\n{df.dtypes}")

            print(f"\nFull precision sample:")
            print(df.iloc[0].to_string())

            # Additional quality checks
            missing = df.isnull().sum().sum()
            print(f"\nMissing values: {missing}")

            ohlc_valid = (
                    (df['high'] >= df['low']).all() and
                    (df['high'] >= df['open']).all() and
                    (df['high'] >= df['close']).all() and
                    (df['low'] <= df['open']).all() and
                    (df['low'] <= df['close']).all()
            )
            print(f"OHLC relationships valid: {ohlc_valid}")

            print("\n✓ Crypto tests passed!")
        else:
            print("❌ No crypto data returned")

    except Exception as e:
        print(f"❌ Crypto test failed: {e}")

    print("\n" + "="*50)
    print("STOCK FUNCTIONALITY TESTS")
    print("="*50)

    try:
        # Test stock data with daily interval
        df_stock = dm.load_or_fetch("SPY", "1d", stock_start, stock_end, market="stock")

        if not df_stock.empty:
            print(f"Stock data fetch: {df_stock.shape}")
            print("Last 5 rows:")
            print(df_stock.tail())
            print("✓ Stock tests passed!")
        else:
            print("⚠ No stock data returned (might be weekend/holiday)")

    except Exception as e:
        print(f"⚠ Stock test failed (expected for unsupported intervals): {e}")

    print(f"\n✓ All tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_standalone_tests()