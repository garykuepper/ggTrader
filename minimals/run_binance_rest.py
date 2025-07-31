import requests

def get_binance_pair(symbol):
    url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"⚠️ Error fetching data for {symbol}: {e}")
        return None

def get_top_binance_us_pairs(top_n=10):
    url = "https://api.binance.us/api/v3/ticker/24hr"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Filter for only USDT pairs
        usdt_pairs = [pair for pair in data if pair['symbol'].endswith('USDT')]
        # Sort by quoteVolume (24h volume in quote currency, usually USDT or USD)
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)

        print(f"\n🔝 Top {top_n} most active Binance.US trading pairs (by 24h volume):\n")
        for i, pair in enumerate(sorted_pairs[:top_n]):
            symbol = pair['symbol']
            volume = float(pair['quoteVolume'])
            price_change = float(pair['priceChangePercent'])
            print(f"{i+1}. {symbol:<10} | Volume: ${volume:,.2f} | Change: {price_change:.2f}%")
    except Exception as e:
        print(f"⚠️ Error fetching Binance.US data: {e}")

def get_binance_kline(symbol, interval='1m', limit=1000):
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"⚠️ Error fetching kline data for {symbol}: {e}")
        return None

def get_top_active_coins(top_n=10, vs_currency='usd'):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': vs_currency,
        'order': 'volume_desc',
        'per_page': top_n,
        'page': 1,
        'sparkline': False
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        print(f"\n🔝 Top {top_n} active coins by 24h volume on CoinGecko:\n")
        for i, coin in enumerate(data):
            name = coin['name']
            symbol = coin['symbol'].upper()
            volume = coin['total_volume']
            price = coin['current_price']
            change = coin['price_change_percentage_24h']
            print(f"{i+1}. {name:<15} ({symbol}) | Price: ${price:.2f} | Volume: ${volume:,.2f} | Change: {change:.2f}%")
    except Exception as e:
        print(f"⚠️ Error fetching data from CoinGecko: {e}")



# Run it
get_top_binance_us_pairs()
pair_data = get_binance_pair("BTCUSDT")
kline_data = get_binance_kline(symbol='BTCUSDT', interval='1d', limit=1)
for row in pair_data.items():
    print(f"{row[0]}: {row[1]}")

if kline_data:
    kline = kline_data[0]
    print(f"\nLatest Kline Data for BTCUSDT:")
    print(f"Timestamp: {kline[0]}")
    print(f"Open: {kline[1]}")
    print(f"High: {kline[2]}")
    print(f"Low: {kline[3]}")
    print(f"Close: {kline[4]}")
    print(f"Volume (Base): {kline[5]}")
    print(f"Quote Volume (USDT/USD): {kline[7]}")


get_top_active_coins()