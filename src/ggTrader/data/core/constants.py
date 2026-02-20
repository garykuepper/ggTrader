"""Canonical Kraken symbol mappings, stable bases, and interval constants."""

# Mapping for Kraken-specific legacy symbols (ISO 4217 prefixes)
# 'X' prefix = Crypto, 'Z' prefix = Fiat
SYMBOL_MAPPING = {
    # Cryptocurrencies
    "XETC": "ETC",
    "XETH": "ETH",
    "XLTC": "LTC",
    "XMLN": "MLN",
    "XREP": "REP",
    "XXBT": "BTC",
    "XXDG": "DOGE",
    "XXLM": "XLM",
    "XXMR": "XMR",
    "XXRP": "XRP",
    "XZEC": "ZEC",
    # Fiat currencies
    "ZAUD": "AUD",
    "ZCAD": "CAD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZJPY": "JPY",
    "ZUSD": "USD",
    # Short-form aliases
    "XBT": "BTC",
    "XDG": "DOGE",
}

# Backward-compatible alias
kraken_map = SYMBOL_MAPPING

STABLE_BASES = {
    "USDT",
    "USDC",
    "DAI",
    "USDP",
    "TUSD",
    "EUR",
    "GBP",
    "AUD",
    "USD",
    "JPY",
    "CAD",
    "MKR",
    "USDG",
}

INTERVAL_MAP = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "240": "4h",
    "720": "12h",
    "1440": "1d",
    "10080": "1w",
}

# Backward-compatible alias
interval_map = INTERVAL_MAP

QUOTES = [
    "ZUSD",
    "ZEUR",
    "ZGBP",
    "ZJPY",
    "ZCAD",
    "ZAUD",
    "USDT",
    "USDC",
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "JPY",
    "XBT",
    "ETH",
]
