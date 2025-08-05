import os
from dotenv import load_dotenv
load_dotenv()
import requests
from tabulate import tabulate

CMC_API_KEY = os.getenv('CMC_API_KEY')
headers = {'X-CMC_PRO_API_KEY': CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com"
STABLES = {'USDT', 'USDC', 'DAI', 'FDUSD', 'TUSD', 'USDP', 'USDD', 'EURS'}
top_n = 15
# Step 1: Get top non-stablecoins by 24h volume
url = f"{BASE_URL}/v1/cryptocurrency/listings/latest"
params = {
    'start': 1,
    'limit': 100,
    'sort': 'volume_24h',
    'sort_dir': 'desc',
    'convert': 'USD'
}
resp = requests.get(url, headers=headers, params=params)
if resp.status_code != 200:
    print("Error:", resp.text)
    exit()
data = resp.json()['data']
data = [coin for coin in data if coin['symbol'] not in STABLES]

table_data = []

for coin in data[:top_n]:  # Only top 5
    table_data.append({
        'Rank': coin['cmc_rank'],
        'Symbol': coin['symbol'],
        'Name': coin['name'],
        'Price': f"${coin['quote']['USD']['price']:,.4f}",
        'Volume (24h)': f"${coin['quote']['USD']['volume_24h']:,.2f}",
        '% Change (24h)': f"{coin['quote']['USD']['percent_change_24h']:.2f}%",
        '% Change (7d)': f"{coin['quote']['USD']['percent_change_7d']:.2f}%"
    })

print(f"\nTop {top_n} Coins by 24h Volume (excluding stables):\n")
print(tabulate(table_data, headers="keys", tablefmt="github"))


# Filter coins for swing trading candidates
swing_candidates = [
    coin for coin in data
    if coin['quote']['USD']['percent_change_24h'] > 0.5
       and coin['quote']['USD']['percent_change_7d'] > 0
       and coin['quote']['USD']['volume_24h'] > 10_000_000
]

table_data = []
for coin in swing_candidates[:top_n]:
    table_data.append({
        'Rank': coin['cmc_rank'],
        'Symbol': coin['symbol'],
        'Name': coin['name'],
        'Price': f"${coin['quote']['USD']['price']:,.4f}",
        'Volume (24h)': f"${coin['quote']['USD']['volume_24h']:,.2f}",
        '% Change (24h)': f"{coin['quote']['USD']['percent_change_24h']:.2f}%",
        '% Change (7d)': f"{coin['quote']['USD']['percent_change_7d']:.2f}%"
    })

print("\nSwing Trade Candidates (Filtered by Momentum and Volume):\n")
print(tabulate(table_data, headers="keys", tablefmt="github"))
