# scripts/fetch_midcap400_snapshot.py
"""One-time: scrape the current S&P MidCap 400 tickers from Wikipedia into a
snapshot file (same format as the nasdaq100/russell2000 snapshots). Survivorship
note: this is a CURRENT snapshot, not point-in-time — see the midcap400 research
spec for the bias-calibration that bounds the resulting bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from ggTrader.data.core.index_constituents import normalize_yf_ticker  # noqa: E402

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
OUT = Path("data/universe/midcap400_tickers_snapshot_2026-06-24.txt")


def main() -> None:
    # Wikipedia blocks bare urllib; send a browser-like User-Agent.
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read()
    tables = pd.read_html(html)
    # The constituents table is the one with a "Symbol" column.
    const = next(t for t in tables if "Symbol" in t.columns)
    raw = [str(s).strip() for s in const["Symbol"].tolist() if str(s).strip()]
    tickers = sorted({normalize_yf_ticker(t) for t in raw})
    if not (380 <= len(tickers) <= 420):
        raise SystemExit(f"Expected ~400 tickers, got {len(tickers)} — check the page structure")
    OUT.write_text("\n".join(tickers) + "\n")
    print(f"wrote {len(tickers)} tickers to {OUT}")


if __name__ == "__main__":
    main()
