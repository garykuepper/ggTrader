import asyncio
import websockets
import json
from datetime import datetime

# 1-min kline WebSocket URL for BTC/USDT on Binance.US
BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@kline_1m"

async def run_stream():
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print(f"✅ Connected to Binance.US WebSocket for BTC/USDT")

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                if "k" in data:
                    k = data["k"]
                    candle_time = datetime.fromtimestamp(k["t"] / 1000).strftime("%Y-%m-%d %H:%M")
                    print(f"[{candle_time}] open={k['o']} high={k['h']} low={k['l']} close={k['c']} volume={k['v']}")
            except Exception as e:
                print(f"❌ Error: {e}")
                break
if __name__ == "__main__":

    asyncio.run(run_stream())
