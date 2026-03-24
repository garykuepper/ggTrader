import json
import os

data_dir = r"c:\Users\gkuep\PycharmProjects\ggTrader\data"
symbols = set()

for filename in os.listdir(data_dir):
    if filename.endswith(".json") and filename != ".processed_dirs.json":
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            symbols.add(item["symbol"])
        except Exception as e:
            pass

for symbol in sorted(list(symbols)):
    print(symbol)
