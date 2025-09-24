# language: python
#!/usr/bin/env python3
"""
get_top_kraken_volume.py

Fetch Kraken tickers via ccxt and return the top markets ranked by 24h quote-volume (notional).
Writes a CSV if --out is provided and prints a small table to stdout.

Requirements:
  - ccxt
  - pandas
  - tabulate (optional but recommended)

Example:
  python get_top_kraken_volume.py --limit 25 --out top25.csv --verbose
"""

from __future__ import annotations
import argparse
import time
import math
from typing import List, Dict, Optional
import ccxt
import pandas as pd
from tabulate import tabulate

DEFAULT_LIMIT = 20
DEFAULT_USD = "USD"
FETCH_TICKERS_TIMEOUT = 30.0


def _safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def estimate_quote_volume_from_ticker(ticker: dict) -> float:
    """
    Try to obtain a reliable 24h quote-volume (notional) for one ticker.
    Prefers these in order:
      - ticker['quoteVolume'] (already quote notional)
      - ticker['info'] with Kraken 'v'[1] (base 24h) and 'p'[1] (vwap 24h)
      - ticker['baseVolume'] * last
    Returns 0.0 if no usable values found.
    """
    # direct ccxt field
    qv = ticker.get("quoteVolume")
    if qv is not None:
        return _safe_float(qv, 0.0)

    # try Kraken-specific raw info
    info = ticker.get("info") or {}
    if isinstance(info, dict):
        v = info.get("v")  # base volume [today, 24h]
        p = info.get("p")  # vwap [today, 24h]
        try:
            base24 = float(v[1]) if isinstance(v, (list, tuple)) and len(v) > 1 else None
        except Exception:
            base24 = None
        try:
            vwap24 = float(p[1]) if isinstance(p, (list, tuple)) and len(p) > 1 else None
        except Exception:
            vwap24 = None
        if base24 is not None and vwap24 is not None:
            return base24 * vwap24
        # fallback: base24 * last
        last = ticker.get("last") or ticker.get("close")
        if base24 is not None and last is not None:
            return base24 * _safe_float(last, 0.0)

    # last resort: baseVolume * last
    base_vol = ticker.get("baseVolume")
    last = ticker.get("last") or ticker.get("close")
    if base_vol is not None and last is not None:
        return _safe_float(base_vol, 0.0) * _safe_float(last, 0.0)

    return 0.0


def get_candidate_symbols(exchange: ccxt.Exchange, only_usd: bool = True) -> List[str]:
    """
    Get candidate market symbols from the exchange markets dict.
    If only_usd is True, restrict to symbols whose quote is USD (market['quote'] == 'USD')
    or whose string endswith '/USD'.
    """
    markets = exchange.markets  # loaded markets required
    symbols = []
    for sym, m in markets.items():
        if only_usd:
            quote = (m.get("quote") or "").upper()
            if quote == DEFAULT_USD:
                symbols.append(sym)
                continue
            # fallback: string suffix check
            if sym.upper().endswith(f"/{DEFAULT_USD}"):
                symbols.append(sym)
                continue
        else:
            symbols.append(sym)
    return symbols


def fetch_tickers_safe(exchange: ccxt.Exchange, symbols: Optional[List[str]] = None, verbose: bool = False) -> Dict[str, dict]:
    """
    Attempt to use fetch_tickers for speed. If it fails, fall back to per-symbol fetch_ticker with
    a polite per-request sleep based on exchange.rateLimit to avoid triggering ccxt internal throttling.

    Returns a dict symbol -> ticker.

    Notes:
      - exchange.rateLimit is expected to be in milliseconds (ccxt convention). We convert to seconds.
      - We handle KeyboardInterrupt gracefully so user can cancel long-running fetches.
      - If symbols is None we iterate over exchange.markets.keys(); consider passing a filtered list
        (e.g., only USD pairs) to avoid fetching every market.
    """
    tickers = {}
    # try bulk fetch first
    try:
        if verbose:
            print("Attempting bulk fetch_tickers() ...")
        # Many exchanges accept a symbols list; pass it when provided.
        bulk = exchange.fetch_tickers(symbols or list(exchange.markets.keys()), timeout=FETCH_TICKERS_TIMEOUT)
        if isinstance(bulk, dict):
            tickers.update(bulk)
    except Exception as e:
        if verbose:
            print(f"fetch_tickers() failed or unsupported: {e}. Falling back to per-symbol fetching.")

    # fetch missing individually
    if symbols is None:
        symbols = list(exchange.markets.keys())
    missing = [s for s in symbols if s not in tickers]
    # compute polite per-request delay from exchange.rateLimit (ms -> s)
    rate_limit_ms = getattr(exchange, "rateLimit", None)
    per_request_delay = (rate_limit_ms / 1000.0) if (isinstance(rate_limit_ms, (int, float)) and rate_limit_ms > 0) else 0.25

    if verbose and missing:
        print(f"Fetching {len(missing)} symbols individually using per-request delay {per_request_delay:.3f}s ...")

    try:
        for i, s in enumerate(missing, start=1):
            try:
                tickers[s] = exchange.fetch_ticker(s)
            except KeyboardInterrupt:
                # allow user to cancel cleanly
                if verbose:
                    print("Interrupted by user during fetch_ticker loop.")
                raise
            except Exception:
                # ignore fetch errors for individual symbols but optionally warn
                if verbose:
                    print(f"warning: failed to fetch ticker for {s}")
                continue
            # polite per-request sleep to respect exchange rate limits and avoid ccxt.throttle long sleeps
            time.sleep(per_request_delay)
    except KeyboardInterrupt:
        # propagate up but allow cleanup if necessary
        if verbose:
            print("Fetch loop cancelled by user (KeyboardInterrupt).")
        raise

    return tickers


def top_24h_volume_kraken(limit: int = DEFAULT_LIMIT, only_usd: bool = True,
                          verbose: bool = False) -> pd.DataFrame:
    """
    Return DataFrame for top markets on Kraken ranked by estimated 24h quote-volume (USD).
    """
    ex = ccxt.kraken()
    ex.load_markets()  # important

    symbols = get_candidate_symbols(ex, only_usd=only_usd)
    if verbose:
        print(f"Found {len(symbols)} candidate symbols (only_usd={only_usd}).")

    tickers = fetch_tickers_safe(ex, symbols=symbols, verbose=verbose)
    rows = []
    for sym, t in tickers.items():
        last = t.get("last") or t.get("close") or None
        base_vol = t.get("baseVolume") or 0.0
        quote_vol_est = estimate_quote_volume_from_ticker(t)
        rows.append({
            "symbol": sym,
            "last": _safe_float(last, None),
            "base_volume_24h": _safe_float(base_vol, 0.0),
            "quote_volume_24h_est": _safe_float(quote_vol_est, 0.0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(by="quote_volume_24h_est", ascending=False).reset_index(drop=True)
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Get top Kraken markets by 24h volume (quote notional).")
    p.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT, help="Number of top markets to return")
    p.add_argument("--out", "-o", type=str, help="Optional CSV output path")
    p.add_argument("--no-usd", action="store_true", help="Include non-USD markets as well")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose progress output")
    args = p.parse_args(argv)

    try:
        df = top_24h_volume_kraken(limit=args.limit, only_usd=not args.no_usd, verbose=args.verbose)
    except Exception as e:
        print("Error fetching top volumes from Kraken:", e)
        return 1

    if df.empty:
        print("No data returned.")
        return 0

    # Pretty print top rows
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".2f", showindex=True))

    if args.out:
        try:
            df.to_csv(args.out, index=False)
            if args.verbose:
                print(f"Wrote top {len(df)} rows to {args.out}")
        except Exception as e:
            print(f"Failed to write CSV to {args.out}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())