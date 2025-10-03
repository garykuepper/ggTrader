# pip install ccxt
import ccxt

ex = ccxt.kraken()
ex.load_markets()

def to_common_currency(code: str) -> str:
    # ccxt handles all the quirks internally (no hardcoded mapping on your side)
    return ex.safe_currency_code(code)


print(to_common_currency("XBTUSD"))