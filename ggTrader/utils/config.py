import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_US_API_KEY = os.getenv("BINANCE_US_API_KEY")
BINANCE_US_API_SECRET = os.getenv("BINANCE_US_API_SECRET")

PROVIDER_CAPABILITIES = {
    "binance": {
        "intervals": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        "supports_crypto": True,
        "supports_stock": False,
        "api_key": BINANCE_US_API_KEY,
        "api_secret": BINANCE_US_API_SECRET
    },
    "yahoo": {
        "intervals": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"],
        "supports_crypto": False,
        "supports_stock": True
    }
}
