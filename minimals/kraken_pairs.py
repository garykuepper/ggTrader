import os
import requests
from tabulate import tabulate
from dotenv import load_dotenv


# TODO:  Add result to mongodb
# ----------------------------
# Config
# ----------------------------
QUOTE_CCY = os.getenv("KRAKEN_QUOTE", "USD")  # change to 'USDT' if you want tether pairs
EXCLUDE_BASES = {"USDT", "USDC", "DAI", "USDP", "TUSD", "EUR", "GBP", "AUD" }  # stablecoins to drop from BASE side
TOP_N = 30

# ----------------------------
# CoinMarketCap helpers

# ----------------------------
def setup_cmc_headers():
    load_dotenv()
    api_key = os.getenv('CMC_API_KEY')
    if not api_key:
        raise RuntimeError("Set CMC_API_KEY in your environment or .env")
    return {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }

def get_cmc_data(headers):
    cmc_url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    params = {'limit': '500', 'convert': 'USD'}
    r = requests.get(cmc_url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()['data']
    # Map symbol -> metrics
    return {
        c['symbol']: {
            'rank': c['cmc_rank'],
            'volume_24h': c['quote']['USD']['volume_24h'],
            'volume_change_24h': c['quote']['USD']['volume_change_24h'],
            'percent_change_24h': c['quote']['USD']['percent_change_24h'],
            'percent_change_7d': c['quote']['USD']['percent_change_7d'],
        }
        for c in data
    }

# ----------------------------
# Kraken helpers
# ----------------------------
def _kraken_quote_codes(q):
    """
    Kraken's AssetPairs 'quote' field uses Z-prefixed codes for fiat (e.g., ZUSD)
    and bare codes for crypto (e.g., USDT). Handle both just in case.
    """
    q = q.upper()
    fiat_pref = f"Z{q}"
    crypto_pref = f"X{q}"
    return {q, fiat_pref, crypto_pref}

def _normalize_for_cmc(base_symbol):
    """
    Map Kraken quirks to CMC symbols.
    """
    m = {
        "XBT": "BTC",   # Kraken REST still uses XBT in many places
        "XDG": "DOGE",  # older alias; typically DOGE nowadays, but keep for safety
    }
    return m.get(base_symbol.upper(), base_symbol.upper())

def get_kraken_asset_pairs(quote_currency="USD"):
    """Fetch AssetPairs and keep only those with the desired quote currency."""
    url = "https://api.kraken.com/0/public/AssetPairs"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {result['error']}")
    pairs = result["result"]

    quote_set = _kraken_quote_codes(quote_currency)
    selected = []
    for pair_code, info in pairs.items():
        alt = info.get("altname")        # e.g., XBTUSD, ETHUSDT
        wsname = info.get("wsname")      # e.g., BTC/USD (v2 WS), may be None for some
        base = info.get("base")          # e.g., XXBT, XETH
        quote = info.get("quote")        # e.g., ZUSD, USDT

        if quote not in quote_set:
            continue

        # Prefer altname to extract a clean base (e.g., XBT from XBTUSD, ETH from ETHUSDT)
        base_for_cmc = None
        if alt and alt.endswith(quote_currency.upper()):
            base_for_cmc = alt[:-len(quote_currency)]
        elif wsname and "/" in wsname and wsname.split("/")[1].upper() == quote_currency.upper():
            base_for_cmc = wsname.split("/")[0]
        else:
            # Fallback: strip leading X/Z from 'base' like XXBT -> XBT or XETH -> ETH
            b = base or ""
            base_for_cmc = b[1:] if b.startswith(("X", "Z")) else b

        base_for_cmc = base_for_cmc.upper()

        if base_for_cmc in EXCLUDE_BASES:
            continue

        selected.append({
            "pair_code": pair_code,  # key used in Ticker result
            "altname": alt or pair_code,
            "base_symbol": base_for_cmc,
            "quote_currency": quote_currency.upper(),
        })
    return selected

def get_kraken_tickers():
    """
    Leave 'pair' blank to fetch ALL tickers once, then subset locally.
    Kraken returns:
      - 'v': [today, last_24h] base volume
      - 'p': [today, last_24h] VWAP (quote terms)
      - 'c': [last_price, lot_size]
    """
    url = "https://api.kraken.com/0/public/Ticker"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken Ticker error: {data['error']}")
    return data["result"]  # dict keyed by pair_code

# ----------------------------
# Table creation
# ----------------------------
def format_table_row(index, pair_info, tickers, cmc_data):
    code = pair_info["pair_code"]
    alt = pair_info["altname"]
    base = pair_info["base_symbol"]
    quote = pair_info["quote_currency"]

    t = tickers.get(code, {})
    v_base_24h = float(t.get("v", [0, 0])[1]) if t else 0.0
    vwap_24h = float(t.get("p", [0, 0])[1]) if t else 0.0
    quote_volume_24h = v_base_24h * vwap_24h  # approximate quote-volume over 24h

    cmc_key = _normalize_for_cmc(base)
    c = cmc_data.get(cmc_key, {})

    return [
        index + 1,
        f"{alt}",  # e.g., XBTUSD / ETHUSDT
        cmc_key,
        c.get("rank", "N/A"),
        f"{quote_volume_24h:,.2f} {quote}",
        f"{c.get('volume_24h', 0):,.2f}" if c else 'N/A',
        f"{c.get('volume_change_24h', 0):,.2f}%" if c else 'N/A',
        f"{c.get('percent_change_24h', 0):,.2f}%" if c else 'N/A',
        f"{c.get('percent_change_7d', 0):,.2f}%" if c else 'N/A',
        ]

def create_table(rows, tickers, cmc_data, limit=TOP_N):
    table = []
    for i, p in enumerate(rows[:limit]):
        table.append(format_table_row(i, p, tickers, cmc_data))
    return table

def display_table(table, quote_ccy):
    headers = [
        'Rank', f'Pair (quote={quote_ccy})', 'CMC Symbol', 'CMC Rank',
        f'Kraken 24h Vol (quote)', 'CMC Vol (USD)',
        'Vol Δ 24h %', 'Price Δ 24h %', 'Price Δ 7d %'
    ]
    alignments = ('right', 'left', 'left', 'right', 'right', 'right', 'right', 'right', 'right')
    print(tabulate(table, headers=headers, tablefmt='github', colalign=alignments))

# ----------------------------
# Main
# ----------------------------
def main():
    try:
        cmc_headers = setup_cmc_headers()
        cmc_data = get_cmc_data(cmc_headers)
    except Exception as e:
        print(f"Error fetching CMC data: {e}")
        cmc_data = {}

    try:
        pairs = get_kraken_asset_pairs(quote_currency=QUOTE_CCY)
        tickers = get_kraken_tickers()
    except Exception as e:
        print(f"Error fetching Kraken data: {e}")
        return

    # Sort by computed quote-volume descending
    def quote_vol(p):
        t = tickers.get(p["pair_code"], {})
        v_base_24h = float(t.get("v", [0, 0])[1]) if t else 0.0
        vwap_24h = float(t.get("p", [0, 0])[1]) if t else 0.0
        return v_base_24h * vwap_24h

    pairs_sorted = sorted(pairs, key=quote_vol, reverse=True)
    table = create_table(pairs_sorted, tickers, cmc_data, limit=TOP_N)
    display_table(table, QUOTE_CCY)

if __name__ == "__main__":
    main()
