
# verify_env.py
import os
from dotenv import load_dotenv

# By default, load_dotenv() looks for a .env file in the current working directory.
# If your .env is elsewhere, pass load_dotenv(dotenv_path="path/to/.env")
loaded = load_dotenv()
print(f".env loaded: {loaded}")

def mask(val: str, show=4):
    if not val:
        return "<missing>"
    if len(val) <= show * 2:
        return "*" * len(val)
    return val[:show] + "*" * (len(val) - show * 2) + val[-show:]

# Replace with the exact variable names you expect
vars_to_check = [
    "BINANCE_API_LIVE_KEY",
    "BINANCE_SECRET_LIVE_KEY",
    "MONGO_URI",
    "MATRIX_HOMESERVER",
    "MATRIX_ROOM_ID",
]

for k in vars_to_check:
    v = os.getenv(k)
    print(f"{k} present: {v is not None}, value: {mask(v)}")