import unittest
from unittest.mock import patch, MagicMock
from swing_trader.db.mongodb import MongoDBClient

class TestMongoDBClient(unittest.TestCase):
    @patch('swing_trader.db.mongodb.MongoClient')
    def test_insert_stock_data(self, mock_mongo_client):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = {'ggTrader': mock_db}

        client = MongoDBClient()
        client.db = mock_db  # override db with mock

        client.insert_stock_data('AAPL', [{'Date': '2023-01-01'}], '2023-01-01', '2023-01-01')
        mock_collection.insert_one.assert_called_once()

    @patch('swing_trader.db.mongodb.MongoClient')
    def test_find_stock_data(self, mock_mongo_client):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{'_id': 1, 'symbol': 'AAPL'}]
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = {'ggTrader': mock_db}

        client = MongoDBClient()
        client.db = mock_db  # override db with mock

        result = client.find_stock_data('AAPL')
        mock_collection.find.assert_called_once()
        self.assertEqual(result, [{'_id': 1, 'symbol': 'AAPL'}])

if __name__ == '__main__':
    unittest.main()