import time
import hmac
import hashlib
import requests


class BinanceClient:
    BASE_URL = 'https://api.binance.us'

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode()

    def _sign(self, params: dict) -> dict:
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
        signature = hmac.new(self.api_secret, query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        return params

    def _headers(self) -> dict:
        return {
            'X-MBX-APIKEY': self.api_key
        }

    def _get(self, endpoint: str, params=None):
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        signed_params = self._sign(params)
        response = requests.get(self.BASE_URL + endpoint, headers=self._headers(), params=signed_params)
        return response.json()

    def _post(self, endpoint: str, params: dict):
        params['timestamp'] = int(time.time() * 1000)
        signed_params = self._sign(params)
        response = requests.post(self.BASE_URL + endpoint, headers=self._headers(), params=signed_params)
        return response.json()

    def get_account_info(self):
        return self._get('/api/v3/account')

    def get_open_orders(self, symbol: str):
        return self._get('/api/v3/openOrders', {'symbol': symbol})

    def market_buy(self, symbol: str, quantity: float):
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': quantity
        }
        return self._post('/api/v3/order', params)

    def market_sell(self, symbol: str, quantity: float):
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': quantity
        }
        return self._post('/api/v3/order', params)

    def limit_buy(self, symbol: str, quantity: float, price: float, time_in_force='GTC'):
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': format(price, '.8f')
        }
        return self._post('/api/v3/order', params)

    def limit_sell(self, symbol: str, quantity: float, price: float, time_in_force='GTC'):
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': format(price, '.8f')
        }
        return self._post('/api/v3/order', params)

    def cancel_order(self, symbol: str, order_id: int):
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._post('/api/v3/order', params)
