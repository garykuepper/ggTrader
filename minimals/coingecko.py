import requests
from tabulate import tabulate
import os
from dotenv import load_dotenv
load_dotenv()
import aiohttp
import asyncio


COIN_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    # Add more as needed!
}
COINGECKO_API_KEY = "your-api-key-here"

async def fetch_usdt_pairs(session, coin, coin_id):
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
                return {
                    'Coin': coin,
                    'Total Volume': f"{total_volume:,.2f}" if total_volume else "N/A",
                    'Binance Volume': f"{binance_volume:,.2f}",
                    'Binance %': f"{binance_pct:.2f}%"
                }
            else:
                return {'Coin': coin, 'Total Volume': 'N/A', 'Binance Volume': 'N/A', 'Binance %': 'N/A'}
    except Exception as e:
        print(f"Error fetching {coin_id}: {e}")
        return {'Coin': coin, 'Total Volume': 'N/A', 'Binance Volume': 'N/A', 'Binance %': 'N/A'}

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for coin, coin_id in COIN_ID_MAP.items():
            tasks.append(fetch_usdt_pairs(session, coin, coin_id))
        results = await asyncio.gather(*tasks)
        print(tabulate(results, headers="keys", tablefmt="github"))

if __name__ == "__main__":
    asyncio.run(main())
