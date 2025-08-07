import os
import requests
from tabulate import tabulate
from dotenv import load_dotenv

def setup_cmc_headers():
    load_dotenv()
    api_key = os.getenv('CMC_API_KEY')
    return {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }

def get_cmc_data(headers):
    cmc_url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    params = {
        'limit': '100',
        'convert': 'USD'
    }
    response = requests.get(cmc_url, headers=headers, params=params)
    return {coin['symbol']: {'rank': coin['cmc_rank'],
                            'volume_24h': coin['quote']['USD']['volume_24h'],
                            'volume_change_24h': coin['quote']['USD']['volume_change_24h'],
                            'percent_change_24h': coin['quote']['USD']['percent_change_24h'],
                            'percent_change_7d': coin['quote']['USD']['percent_change_7d']
                             }
            for coin in response.json()['data']}

def get_binance_data():
    url = "https://api.binance.us/api/v3/ticker/24hr"
    response = requests.get(url)
    return response.json()

def filter_usdt_pairs(data, exclude_symbols=['USDC']):
    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
    return [
        item for item in usdt_pairs
        if not any(exclude in item['symbol'] for exclude in exclude_symbols)
    ]

def format_table_row(index, pair, cmc_data):
    symbol_without_usdt = pair['symbol'].replace('USDT', '')
    cmc_info = cmc_data.get(symbol_without_usdt, {})
    
    return [
        index + 1,
        pair['symbol'],
        cmc_info.get('rank', 'N/A'),
        f"{float(pair['quoteVolume']):,.2f}",
        f"{cmc_info.get('volume_24h', 0):,.2f}" if cmc_info else 'N/A',
        f"{cmc_info.get('volume_change_24h', 0):,.2f}%" if cmc_info else 'N/A',
        f"{cmc_info.get('percent_change_24h', 0):,.2f}%" if cmc_info else 'N/A',
        f"{cmc_info.get('percent_change_7d', 0):,.2f}%" if cmc_info else 'N/A'
    ]

def create_table(sorted_pairs, cmc_data, limit=15):
    return [
        format_table_row(i, pair, cmc_data)
        for i, pair in enumerate(sorted_pairs[:limit])
    ]

def display_table(table):
    headers = [
        'Rank', 'Pair', 'CMC Rank', 'Binance Volume (USDT)',
        'CMC Volume (USD)', 'Vol Change 24h %', '24h Change %', '7d Change %'
    ]
    alignments = ('right', 'left', 'right', 'right', 'right', 'right', 'right', 'right')
    
    print(tabulate(table,
                  headers=headers,
                  tablefmt='github',
                  colalign=alignments))

# Main execution flow
cmc_headers = setup_cmc_headers()

try:
    cmc_data = get_cmc_data(cmc_headers)
except Exception as e:
    print(f"Error fetching CMC data: {e}")
    cmc_data = {}

binance_data = get_binance_data()
filtered_pairs = filter_usdt_pairs(binance_data)
sorted_pairs = sorted(filtered_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
table_data = create_table(sorted_pairs, cmc_data)
display_table(table_data)
