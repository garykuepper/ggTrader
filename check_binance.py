import requests
import time
import hmac
import hashlib
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv("../.env")
# Load from environment variables (recommended)
API_KEY = os.getenv("BINANCE_API_LIVE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_LIVE_KEY")

BASE_URL = "https://api.binance.us"

def sign_request(params: dict):
    query_string = urlencode(params)
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return f"{query_string}&signature={signature}"

def account_status():
    params = {
        "timestamp": int(time.time() * 1000)
    }
    signed_params = sign_request(params)
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    r = requests.get(f"{BASE_URL}/api/v3/account?{signed_params}", headers=headers)
    return r.json()

if __name__ == "__main__":
    data = account_status()
    for key in data.keys():
        print(f"{key}: {data[key]}")
