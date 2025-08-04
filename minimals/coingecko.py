import os
from dotenv import load_dotenv
load_dotenv()
import aiohttp
import asyncio
from tabulate import tabulate

COIN_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    # Add more as needed!
}
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')

async def fetch_usdt_pairs(session, coin, coin_id, top_exchanges=5):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/tickers"
    headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}
    try:
        async with session.get(url, headers=headers, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                tickers = data.get('tickers', [])
                usdt_pairs = [t for t in tickers if t['target'] == 'USDT']
                total_volume = sum(float(t['volume']) for t in usdt_pairs)
                binance_pair = next((t for t in usdt_pairs if t['market']['name'].lower() == 'binance'), None)
                binance_volume = float(binance_pair['volume']) if binance_pair else 0.0
                binance_pct = (binance_volume / total_volume * 100) if total_volume > 0 else 0

                # Get top exchanges by USDT volume for this coin
                top_usdt = sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)[:top_exchanges]
                exchanges_table = [
                    {
                        'Exchange': t['market']['name'],
                        'Pair': f"{t['base']}/{t['target']}",
                        'Volume (24h)': f"{float(t['volume']):,.2f}",
                        'Price': f"${float(t['last']):,.4f}"
                    }
                    for t in top_usdt
                ]

                return {
                    'Coin': coin,
                    'Total Volume': f"{total_volume:,.2f}" if total_volume else "N/A",
                    'Binance Volume': f"{binance_volume:,.2f}",
                    'Binance %': f"{binance_pct:.2f}%",
                    'Exchanges Table': exchanges_table
                }
            else:
                return {
                    'Coin': coin,
                    'Total Volume': 'N/A',
                    'Binance Volume': 'N/A',
                    'Binance %': 'N/A',
                    'Exchanges Table': []
                }
    except Exception as e:
        print(f"Error fetching {coin_id}: {e}")
        return {
            'Coin': coin,
            'Total Volume': 'N/A',
            'Binance Volume': 'N/A',
            'Binance %': 'N/A',
            'Exchanges Table': []
        }

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for coin, coin_id in COIN_ID_MAP.items():
            tasks.append(fetch_usdt_pairs(session, coin, coin_id))
        results = await asyncio.gather(*tasks)

        # Print summary table
        print(tabulate(
            [{k: v for k, v in r.items() if k != 'Exchanges Table'} for r in results],
            headers="keys", tablefmt="github")
        )

        # Print exchange breakdowns for each coin
        for r in results:
            print(f"\n=== {r['Coin']} - Top 5 Exchanges by USDT Volume ===")
            if r['Exchanges Table']:
                print(tabulate(r['Exchanges Table'], headers="keys", tablefmt="github"))
            else:
                print("No USDT pairs found or error.")

if __name__ == "__main__":
    asyncio.run(main())
