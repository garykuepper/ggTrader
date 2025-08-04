import os
from dotenv import load_dotenv
load_dotenv()
import requests
from tabulate import tabulate

CMC_API_KEY = os.getenv('CMC_API_KEY')
headers = {'X-CMC_PRO_API_KEY': CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com"
STABLES = {'USDT', 'USDC', 'DAI', 'FDUSD', 'TUSD', 'USDP', 'USDD', 'EURS'}

# Step 1: Get top non-stablecoins by 24h volume
url = f"{BASE_URL}/v1/cryptocurrency/listings/latest"
params = {
    'start': 1,
    'limit': 20,
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

for coin in data[:15]:  # Only top 5
    table_data.append({
        'Rank': coin['cmc_rank'],
        'Symbol': coin['symbol'],
        'Name': coin['name'],
        'Price': f"${coin['quote']['USD']['price']:,.4f}",
        'Volume (24h)': f"${coin['quote']['USD']['volume_24h']:,.2f}",
        '% Change (24h)': f"{coin['quote']['USD']['percent_change_24h']:.2f}%",
    })

print("\nTop 5 Coins by 24h Volume (excluding stables):\n")
print(tabulate(table_data, headers="keys", tablefmt="github"))
