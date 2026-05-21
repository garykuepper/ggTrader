"""Fear & Greed Index fetcher (alternative.me).

Free public API, no auth required. Returns daily values 0-100 with classification:
  0-25:   Extreme Fear
  26-46:  Fear
  47-54:  Neutral
  55-75:  Greed
  76-100: Extreme Greed

Used as a market sentiment indicator in PnL reports and ``ggt signals``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

API_URL = "https://api.alternative.me/fng/"


def _classification_emoji(value: int) -> str:
    if value <= 25:
        return "😱"  # Extreme Fear
    if value <= 46:
        return "😟"  # Fear
    if value <= 54:
        return "😐"  # Neutral
    if value <= 75:
        return "😊"  # Greed
    return "🤑"  # Extreme Greed


def fetch_fear_greed(limit: int = 1, timeout: float = 10.0) -> Optional[dict]:
    """Fetch the most recent N days of Fear & Greed Index values.

    Returns ``None`` on network failure (caller should degrade gracefully).
    On success returns:

        {
            "value":          47,
            "classification": "Neutral",
            "emoji":          "😐",
            "date":           date(2026, 5, 6),
            "timestamp":      1778112000,
            "history": [ {value, classification, date, timestamp}, ... ]   # if limit > 1
        }
    """
    try:
        r = requests.get(API_URL, params={"limit": limit}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  [F&G] WARNING: fetch failed — {e!r}")
        return None

    items = payload.get("data") or []
    if not items:
        return None

    def _row(entry: dict) -> dict:
        ts = int(entry["timestamp"])
        v = int(entry["value"])
        return {
            "value": v,
            "classification": entry["value_classification"],
            "emoji": _classification_emoji(v),
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).date(),
            "timestamp": ts,
        }

    latest = _row(items[0])
    if limit > 1:
        latest["history"] = [_row(e) for e in items]
    return latest


def fetch_fear_greed_history(days: int = 365) -> list[dict]:
    """Fetch the last ``days`` of daily Fear & Greed values for backfill.

    Returns a list of ``{value, classification, emoji, date, timestamp}`` rows
    in reverse chronological order. Empty list on failure.
    """
    result = fetch_fear_greed(limit=days)
    if result is None:
        return []
    return result.get("history", [result])


def format_inline(latest: Optional[dict]) -> str:
    """One-line summary for inclusion in plain-text reports / signals header.

    Example: ``F&G: 47 Neutral 😐``
    """
    if not latest:
        return "F&G: n/a"
    return f"F&G: {latest['value']} {latest['classification']} {latest['emoji']}"


def format_inline_with_delta(latest: Optional[dict], history_days: int = 7) -> str:
    """Inline summary with a delta-vs-N-days-ago suffix when ``history`` is present.

    Example: ``F&G: 47 Neutral 😐 (+5 vs 7d)``
    """
    if not latest:
        return "F&G: n/a"
    base = format_inline(latest)
    hist = latest.get("history") or []
    if len(hist) >= history_days + 1:
        prior = hist[history_days]
        delta = latest["value"] - prior["value"]
        sign = "+" if delta >= 0 else ""
        return f"{base} ({sign}{delta} vs {history_days}d)"
    return base
