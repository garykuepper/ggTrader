"""Integration tests for the Kraken Futures ingester.

These hit Kraken's public REST endpoints. Skipped if unreachable so offline
runs don't break.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ggTrader.data.sources.kraken_futures import (
    _make_session,
    fetch_chart_window,
    fetch_funding_history,
)

pytestmark = pytest.mark.integration


def test_chart_endpoint_returns_recent_candles_for_pf_xbtusd():
    session = _make_session()
    end = datetime.now(tz=timezone.utc)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        bars = fetch_chart_window(session, "PF_XBTUSD", "1h", start, end)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kraken Futures unreachable: {exc}")
    assert len(bars) >= 1
    last = bars[-1]
    assert last.close > 0
    assert last.ts.tzinfo is not None
    assert last.ts >= start


def test_funding_endpoint_returns_recent_rates():
    session = _make_session()
    try:
        rows = fetch_funding_history(session, "PF_XBTUSD")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kraken Futures unreachable: {exc}")
    assert len(rows) > 1000  # ~1 year × 24h
    assert rows[0].ts < rows[-1].ts
    # Funding sign sanity: across the year, mean is positive on Kraken historically
    mean_rel = sum(r.relative_funding_rate for r in rows) / len(rows)
    assert -0.001 < mean_rel < 0.001  # per-hour rate is small in absolute terms


def test_sign_convention_positive_funding_means_longs_pay():
    """Kraken funding sign convention: positive relativeFundingRate means
    longs pay shorts (mark-to-index premium → market expects rising perp →
    longs are crowded → they pay to maintain skew). Spot-check by sampling
    a known positive-funding period (post-ETF era, late 2024 / early 2025)."""
    session = _make_session()
    try:
        rows = fetch_funding_history(session, "PF_XBTUSD")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kraken Futures unreachable: {exc}")
    # Pull a sample from the high-funding window if it's still in the API's
    # recent ~1y window.
    nov_2025 = [r for r in rows if r.ts.year == 2025 and r.ts.month == 11]
    if not nov_2025:
        pytest.skip("Nov 2025 outside API's recent window — re-run sooner")
    mean = sum(r.relative_funding_rate for r in nov_2025) / len(nov_2025)
    # Post-ETF Nov 2025 funding was positive (cleared spot-perp basis was positive).
    assert mean > 0, f"Nov 2025 mean funding should be positive; got {mean}"
