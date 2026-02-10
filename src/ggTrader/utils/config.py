import os
import json


def load_symbols_from_json(file_path):
    """Loads symbols from a JSON file expecting a list of objects with a 'symbol' key."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return [item["symbol"] for item in data if "symbol" in item]
    except Exception as e:
        print(f"Error loading symbols from {file_path}: {e}")
        return None
