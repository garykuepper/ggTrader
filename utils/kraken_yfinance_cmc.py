#!/usr/bin/env python3
"""
Kraken x Yahoo Finance x CMC (USD-only, refactored)

- Rank Kraken USD pairs by last-24h notional (base_vol * 24h VWAP)
- Map Kraken symbols to common names using 'wsname' (e.g., XDGUSD -> DOGE/USD)
- Yahoo Finance: map to BASE-USD; if not found, mark '-'
- CMC: enrich via base_common symbol (BTC, ETH, DOGE, ...)
- Public API: get_top_kraken_usd_pairs(top_n=30, require_yf=False) -> pandas.DataFrame
"""

from __future__ import annotations

import os
import time
import requests
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from tabulate import tabulate

# yfinance is optional — if unavailable, we still run and mark YF column as '-'
try:
    import yfinance as yf  # recent versions need websockets>=12
    YF_AVAILABLE = True
    YF_IMPORT_ERROR = None
except Exception as _e:
    YF_AVAILABLE = False
    YF_IMPORT_ERROR = _e

# ----------------------------
# Config
# ----------------------------
TOP_N = int(os.getenv("TOP_N", "30"))
QUOTE_CCY = "USD"  # script is USD-only
EXCLUDE_BASES = {"USDT", "USDC", "DAI", "USDP", "TUSD", "EUR", "GBP", "AUD", "USDG"}

# Kraken→Common base aliases (keep tiny; wsname covers most)
KRAKEN_BASE_ALIASES: Dict[str, str] = {
    "XBT": "BTC",
    "XDG": "DOGE",

}

# ----------------------------
# Datamodel
# ----------------------------
@dataclass
class Row:
    rank: int
    kraken_pair: str         # e.g., XDGUSD
    base_common: str         # e.g., DOGE
    yf_ticker: str           # e.g., DOGE-USD or '-'
    kraken_vol_usd: float    # v[1] * p[1]
    cmc_rank: int | str
    cmc_vol_usd: float | str
    cmc_vol_change_24h: float | str
    cmc_price_change_24h: float | str
    cmc_price_change_7d: float | str

# ----------------------------
# CMC helpers
# ----------------------------
def setup_cmc_headers() -> Dict[str, str]:
    load_dotenv()
    api_key = os.getenv('CMC_API_KEY')
    if not api_key:
        raise RuntimeError("Set CMC_API_KEY in your environment or .env")
    return {'X-CMC_PRO_API_KEY': api_key, 'Accept': 'application/json'}

def get_cmc_data(headers: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    params = {'limit': '500', 'convert': 'USD'}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()['data']
    return {
        c['symbol'].upper(): {
            'rank': c['cmc_rank'],
            'volume_24h': c['quote']['USD']['volume_24h'],
            'volume_change_24h': c['quote']['USD']['volume_change_24h'],
            'percent_change_24h': c['quote']['USD']['percent_change_24h'],
            'percent_change_7d': c['quote']['USD']['percent_change_7d'],
        }
        for c in data
    }

# ----------------------------
# Kraken helpers
# ----------------------------
def _alias_to_common(sym: str) -> str:
    return KRAKEN_BASE_ALIASES.get(sym.upper(), sym.upper())

def extract_base_quote_common(info: dict) -> tuple[str, str] | None:
    """
    Prefer 'wsname' which already has human/common symbols (e.g., 'DOGE/USD').
    Fallback to 'altname' suffix parsing. Last resort: base/quote fields.
    """
    # 1) wsname 'BASE/QUOTE'
    ws = (info.get("wsname") or "").upper()
    if ws and "/" in ws:
        base_ws, quote_ws = ws.split("/", 1)
        return _alias_to_common(base_ws), _alias_to_common(quote_ws)

    # 2) altname like 'XBTUSD', 'XRPUSD', 'ETHUSDT'
    alt = (info.get("altname") or "").upper()
    if alt:
        for q in ("USD", "USDT", "USDC", "EUR", "GBP", "AUD"):
            if alt.endswith(q):
                base = alt[:-len(q)]
                return _alias_to_common(base), _alias_to_common(q)

    # 3) base/quote fields with possible leading X/Z prefixes
    base_raw = (info.get("base") or "").upper()
    quote_raw = (info.get("quote") or "").upper()

    def _strip_xz(s: str) -> str:
        # common Kraken asset ids look like 'XXBT', 'XETH', 'ZUSD'
        if len(s) >= 4 and s[0] in "XZ":
            return s[1:]
        return s

    base = _alias_to_common(_strip_xz(base_raw))
    quote = _alias_to_common(_strip_xz(quote_raw))
    if base and quote:
        return base, quote
    return None

def get_kraken_asset_pairs_usd() -> List[Dict[str, str]]:
    url = "https://api.kraken.com/0/public/AssetPairs"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {payload['error']}")

    out: List[Dict[str, str]] = []
    for pair_code, info in payload["result"].items():
        bq = extract_base_quote_common(info)
        if not bq:
            continue
        base_common, quote_common = bq
        if quote_common != "USD":
            continue
        if base_common in EXCLUDE_BASES:
            continue
        out.append({
            "pair_code": pair_code,
            "altname": info.get("altname") or pair_code,  # e.g., XDGUSD
            "base_common": base_common,                    # e.g., DOGE
            "quote_common": quote_common,                  # USD
        })
    return out

def get_kraken_tickers() -> Dict[str, Any]:
    url = "https://api.kraken.com/0/public/Ticker"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken Ticker error: {payload['error']}")
    return payload["result"]

def kraken_quote_volume_usd(tickers: dict, pair_code: str) -> float:
    """
    Kraken ticker fields:
      'v': [today, last_24h]  (base volume)
      'p': [today, last_24h]  (VWAP in quote terms)
    Notional USD ≈ v[1] * p[1] for USD-quoted pairs.
    """
    t = tickers.get(pair_code, {})
    v_base_24h = float(t.get("v", [0, 0])[1]) if t else 0.0
    vwap_24h   = float(t.get("p", [0, 0])[1]) if t else 0.0
    return v_base_24h * vwap_24h

# ----------------------------
# Yahoo Finance helpers (USD only)
# ----------------------------
def yf_usd_symbol(base_common: str) -> str:
    return f"{base_common}-USD"

def yf_symbol_exists(sym: str, pause: float = 0.0) -> bool:
    """Return False if yfinance unavailable, or if download is empty."""
    if not YF_AVAILABLE:
        return False
    try:
        df = yf.download(sym, period="2d", interval="1h", progress=False, auto_adjust=True)
        if pause:
            time.sleep(pause)
        return not df.empty
    except Exception:
        return False

# ----------------------------
# Core pipeline
# ----------------------------
def build_ranked_rows(top_n: int = TOP_N) -> List[Row]:
    # CMC (optional)
    try:
        cmc = get_cmc_data(setup_cmc_headers())
    except Exception as e:
        print(f"Warning: CMC unavailable: {e}")
        cmc = {}

    pairs = get_kraken_asset_pairs_usd()
    tickers = get_kraken_tickers()

    # Rank by Kraken last-24h notional USD
    pairs_sorted = sorted(
        pairs,
        key=lambda p: kraken_quote_volume_usd(tickers, p["pair_code"]),
        reverse=True
    )

    rows: List[Row] = []
    for i, p in enumerate(pairs_sorted, start=1):
        base = p["base_common"]
        kr_alt = p["altname"]
        k_usd = kraken_quote_volume_usd(tickers, p["pair_code"])

        # Map to Yahoo (BASE-USD); mark '-' if missing
        yf_sym = yf_usd_symbol(base)
        if not yf_symbol_exists(yf_sym):
            yf_sym = "-"

        # CMC stats using common base symbol
        c = cmc.get(base, {})
        rows.append(Row(
            rank=i,
            kraken_pair=kr_alt,
            base_common=base,
            yf_ticker=yf_sym,
            kraken_vol_usd=k_usd,
            cmc_rank=c.get("rank", "N/A"),
            cmc_vol_usd=c.get("volume_24h", "N/A"),
            cmc_vol_change_24h=c.get("volume_change_24h", "N/A"),
            cmc_price_change_24h=c.get("percent_change_24h", "N/A"),
            cmc_price_change_7d=c.get("percent_change_7d", "N/A"),
        ))
        if len(rows) >= top_n:
            break
    return rows

def rows_to_dataframe(rows: List[Row]) -> pd.DataFrame:
    """Convert ranked rows to a pandas DataFrame."""
    df = pd.DataFrame([{
        "Rank": r.rank,
        "Kraken Pair": r.kraken_pair,
        "Common": r.base_common,
        "YF Ticker": r.yf_ticker,
        "Kraken 24h Vol (USD)": r.kraken_vol_usd,
        "CMC Vol (USD)": r.cmc_vol_usd,
        "CMC Rank": r.cmc_rank,
        "Vol Δ 24h %": r.cmc_vol_change_24h,
        "Price Δ 24h %": r.cmc_price_change_24h,
        "Price Δ 7d %": r.cmc_price_change_7d,
    } for r in rows])
    # Make numeric where applicable
    for col in ["Rank", "Kraken 24h Vol (USD)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def print_rows(rows: List[Row]) -> None:
    headers = [
        "Rank", "Kraken Pair", "Common", "YF Ticker",
        "Kraken 24h Vol (USD)", "CMC Vol (USD)", "CMC Rank",
        "Vol Δ 24h %", "Price Δ 24h %", "Price Δ 7d %"
    ]
    table = []
    for r in rows:
        table.append([
            r.rank,
            r.kraken_pair,
            r.base_common,
            r.yf_ticker,
            f"{r.kraken_vol_usd:,.2f}",
            (f"{r.cmc_vol_usd:,.2f}" if isinstance(r.cmc_vol_usd, (int, float)) else r.cmc_vol_usd),
            r.cmc_rank,
            (f"{r.cmc_vol_change_24h:,.2f}%" if isinstance(r.cmc_vol_change_24h, (int, float)) else r.cmc_vol_change_24h),
            (f"{r.cmc_price_change_24h:,.2f}%" if isinstance(r.cmc_price_change_24h, (int, float)) else r.cmc_price_change_24h),
            (f"{r.cmc_price_change_7d:,.2f}%" if isinstance(r.cmc_price_change_7d, (int, float)) else r.cmc_price_change_7d),
        ])
    print(tabulate(
        table, headers=headers, tablefmt="github",
        colalign=("right","left","left","left","right","right","right","right","right","right")
    ))

# ----------------------------
# Public API: return DataFrame
# ----------------------------
def get_top_kraken_usd_pairs(top_n: int = 30, require_yf: bool = False) -> pd.DataFrame:
    """
    Return a pandas DataFrame of the top Kraken USD pairs with CMC/Yahoo columns.
    If require_yf=True, filter to rows where 'YF Ticker' != '-'.
    """
    rows = build_ranked_rows(top_n=top_n)
    if require_yf:
        rows = [r for r in rows if r.yf_ticker != "-"]
    return rows_to_dataframe(rows)

# ----------------------------
# Main (demo)
# ----------------------------
def main():
    if not YF_AVAILABLE and YF_IMPORT_ERROR:
        print(f"Note: yfinance import failed ({YF_IMPORT_ERROR}). "
              f"YF Ticker column will be '-' for all rows.\n")

    rows = build_ranked_rows(top_n=TOP_N)
    print_rows(rows)

    # Example: get a DataFrame for downstream analysis
    df = rows_to_dataframe(rows)
    # print("\nDataFrame:")
    # print(tabulate(df, headers='keys', tablefmt='github'))  # uncomment to preview

if __name__ == "__main__":
    main()
