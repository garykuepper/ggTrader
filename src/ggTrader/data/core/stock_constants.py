"""Constants for stock trading (NYSE/NASDAQ)."""

# yfinance interval mapping (project canonical -> yfinance accepted values)
# yfinance accepts: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
YFINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}

# NYSE market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = "US/Eastern"

# S&P 500 constituents (Top 50 by market cap as a starting point)
# Note: In production, use scripts/update_universe_stocks.py to fetch dynamically.
SP500_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "V",
    "JPM", "XOM", "LLY", "AVGO", "MA", "PG", "HD", "JNJ", "CVX", "ADBE",
    "MRK", "COST", "PEP", "ABBV", "KO", "TMO", "WMT", "BAC", "CRM", "AVGO",
    "MCD", "CSCO", "ACN", "ABT", "LIN", "ORCL", "INTC", "AMD", "WFC", "VZ",
    "CMCSA", "DIS", "PM", "PFE", "COP", "TXN", "DHR", "NFLX", "CAT", "AMGN"
]
