
import unittest
from unittest.mock import patch
import pandas as pd
from swing_trader.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    @patch('swing_trader.data.data_loader.MongoDBClient')
    @patch('swing_trader.data.data_loader.yf.download')
    def test_get_stock_data_with_missing_dates(self, mock_yf_download, mock_mongo_client):
        # Initial DB data
        initial_data = [{'data': [{'Date': '2023-01-01'}, {'Date': '2023-01-02'}]}]
        # After insert, DB should have all 5 records
        all_data = [{'data': [
            {'Date': '2023-01-01'}, {'Date': '2023-01-02'},
            {'Date': '2023-01-03', 'Close': 100},
            {'Date': '2023-01-04', 'Close': 101},
            {'Date': '2023-01-05', 'Close': 102}
        ]}]

        mock_client_instance = mock_mongo_client.return_value
        # Switch return value after insert_stock_data is called
        def find_stock_data_side_effect(*args, **kwargs):
            if mock_client_instance.insert_stock_data.called:
                return all_data
            return initial_data
        mock_client_instance.find_stock_data.side_effect = find_stock_data_side_effect

        # Mock yfinance DataFrame
        missing_dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])
        df = pd.DataFrame({'Close': [100, 101, 102]}, index=missing_dates)
        df.index.name = 'Date'
        mock_yf_download.return_value = df

        loader = DataLoader()
        result = loader.get_stock_data('AAPL', '2023-01-01', '2023-01-05')

        self.assertEqual(len(result), 5)  # 2 from DB + 3 from yfinance

    @patch('swing_trader.data.data_loader.MongoDBClient')
    def test_get_stock_data_no_missing_dates(self, mock_mongo_client):
        mock_client_instance = mock_mongo_client.return_value
        mock_client_instance.find_stock_data.return_value = [
            {'data': [{'Date': '2023-01-01'}, {'Date': '2023-01-02'}, {'Date': '2023-01-03'}]}
        ]
        loader = DataLoader()
        result = loader.get_stock_data('AAPL', '2023-01-01', '2023-01-03')
        self.assertEqual(len(result), 3)
        mock_client_instance.insert_stock_data.assert_not_called()

if __name__ == '__main__':
    unittest.main()