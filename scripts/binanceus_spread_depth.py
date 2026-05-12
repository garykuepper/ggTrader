"""Cross-exchange spread + depth + 24h volume comparison (Binance.US vs Kraken).

Public data only — no API keys required.

Outputs:
  - Human-readable table to stdout (also goes to log file via cron redirect)
  - One JSON line per pair × venue appended to results/binanceus_smoke/snapshots.jsonl
    (downstream consolidation reads from this file)

Designed for cron scheduling — multiple snapshots accumulate in the JSON-lines file.
"""

import datetime as dt
import json
from pathlib import Path

import ccxt

REPO = Path("/home/flynn/ggTrader")
OUT_DIR = REPO / "results" / "binanceus_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "snapshots.jsonl"

PAIRS = ["BTC/USD", "ETH/USD", "DOGE/USD", "TRX/USD"]

kraken = ccxt.kraken({"enableRateLimit": True})
bus = ccxt.binanceus({"enableRateLimit": True})


def order_book_stats(book):
    bid = book["bids"][0][0]
    ask = book["asks"][0][0]
    spread_bp = 10000 * (ask - bid) / bid
    # Kraken returns [price, amount, timestamp]; Binance.US [price, amount].
    bid_depth = sum(level[0] * level[1] for level in book["bids"][:5])
    ask_depth = sum(level[0] * level[1] for level in book["asks"][:5])
    return spread_bp, bid_depth, ask_depth


def ticker_volume_usd(ticker):
    """Return 24h quote-currency (USD) volume.

    Prefer quoteVolume (in USD). Fall back to baseVolume × last if quoteVolume
    is missing (some venues populate only one).
    """
    qv = ticker.get("quoteVolume")
    if qv is not None and qv > 0:
        return float(qv)
    bv = ticker.get("baseVolume")
    last = ticker.get("last")
    if bv is not None and last is not None and bv > 0 and last > 0:
        return float(bv) * float(last)
    return None


def fetch_pair_snapshot(venue_name, exchange, pair):
    book = exchange.fetch_order_book(pair, limit=10)
    ticker = exchange.fetch_ticker(pair)
    spread_bp, bid_d, ask_d = order_book_stats(book)
    vol_usd = ticker_volume_usd(ticker)
    return {
        "venue": venue_name,
        "pair": pair,
        "spread_bp": round(spread_bp, 2),
        "top5_bid_usd": round(bid_d, 0),
        "top5_ask_usd": round(ask_d, 0),
        "vol_24h_usd": round(vol_usd, 0) if vol_usd is not None else None,
        "best_bid": book["bids"][0][0],
        "best_ask": book["asks"][0][0],
    }


def main():
    ts_utc = dt.datetime.now(dt.timezone.utc)
    ts_iso = ts_utc.isoformat()
    print(f"Snapshot: {ts_iso}")
    print(
        f"{'Pair':<10} {'Venue':<12} {'spread_bp':>10} "
        f"{'top5_bid_$':>12} {'top5_ask_$':>12} {'vol_24h_$':>14}"
    )

    rows = []
    for pair in PAIRS:
        bus_row = fetch_pair_snapshot("Binance.US", bus, pair)
        kr_row = fetch_pair_snapshot("Kraken", kraken, pair)
        bus_row["snapshot_ts"] = ts_iso
        kr_row["snapshot_ts"] = ts_iso
        rows.extend([bus_row, kr_row])

        def fmt_vol(v):
            return f"{v:>13,.0f}" if v is not None else "          n/a"

        print(
            f"{pair:<10} {'Binance.US':<12} {bus_row['spread_bp']:>10.1f} "
            f"{bus_row['top5_bid_usd']:>12,.0f} {bus_row['top5_ask_usd']:>12,.0f} "
            f"{fmt_vol(bus_row['vol_24h_usd'])}"
        )
        print(
            f"{pair:<10} {'Kraken':<12} {kr_row['spread_bp']:>10.1f} "
            f"{kr_row['top5_bid_usd']:>12,.0f} {kr_row['top5_ask_usd']:>12,.0f} "
            f"{fmt_vol(kr_row['vol_24h_usd'])}"
        )
        print(
            f"{pair:<10} {'Δ (BUS-K)':<12} "
            f"{bus_row['spread_bp'] - kr_row['spread_bp']:>+10.1f} "
            f"{bus_row['top5_bid_usd'] - kr_row['top5_bid_usd']:>+12,.0f} "
            f"{bus_row['top5_ask_usd'] - kr_row['top5_ask_usd']:>+12,.0f}"
        )
        print()

    # Append JSON lines (one per pair × venue)
    with open(OUT_FILE, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Appended {len(rows)} rows → {OUT_FILE}")


if __name__ == "__main__":
    main()
