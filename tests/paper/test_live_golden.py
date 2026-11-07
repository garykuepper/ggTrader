"""P7 acceptance: the rebuilt signal runner reproduces P0's live golden on the frozen tape.

The golden (`research/programme/vectorbt-engine/legacy-references/live_golden_legacy.json`) holds,
for the last 120 sessions, what the pre-rebuild `generate_signals` returned on the 365-day window:
`buys`, `sells`, `last_exit`, `as_of` and `universe_size`. Here the real, rebuilt
`signal_runner.generate_signals` runs on the same frames (the live loader is replaced by a slice
of the hashed tape; everything else is production code, including the engine self-check and the
production sanity bounds) and must return the same lists, session by session.

Skipped when the frozen tape (git-ignored, ~33 MB) is absent, as on CI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "research/programme/vectorbt-engine/legacy-references/live_golden_legacy.json"
TAPE = REPO / "data/snapshots/ohlcv_20261106.parquet"
EVAL_SPAN = ("2021-01-31", "2026-04-30")
BAR_WINDOW = ("2019-03-01", "2026-10-30")

pytestmark = [
    # Not in the fast core (38 s) but required in the FULL tier before any deploy of paper/ or the
    # signal code: it is the only check that the rebuilt live signal equals the pre-rebuild one.
    pytest.mark.slow,
    pytest.mark.skipif(
        not (TAPE.exists() and GOLDEN.exists()), reason="frozen tape or live golden is absent"
    ),
]


@pytest.fixture(scope="module")
def tape_frame() -> pd.DataFrame:
    """The whole PIT-union tape as the live loader's `(symbol, field)` frame."""
    from ggTrader.data.bars import load_bars

    bars = load_bars(
        universe="sp500",
        universe_span=EVAL_SPAN,
        extra_symbols=("SPY", "TLT", "GLD", "PDBC", "BIL"),
        start=BAR_WINDOW[0],
        end=BAR_WINDOW[1],
    )
    return bars.legacy_frame()


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return json.loads(GOLDEN.read_text())


def test_golden_has_120_sessions(golden):
    assert len(golden) == 120
    assert {"buys", "sells", "last_exit", "as_of", "session", "universe_size"} <= set(golden[0])


def test_rebuilt_signal_runner_reproduces_the_live_golden(tape_frame, golden):
    """120 of 120 sessions: buys, sells, last_exit, as_of and universe_size all equal."""
    from ggTrader.paper import signal_runner

    symbols_of = tape_frame.columns.get_level_values(0)

    def fake_fetch(symbols, start, end):
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        rows = (tape_frame.index >= lo) & (tape_frame.index <= hi)
        cols = symbols_of.isin(symbols)
        return tape_frame.loc[rows, cols]

    mismatches = []
    total_buys = total_sells = 0
    with patch.object(signal_runner, "fetch_stock_ohlcv", side_effect=fake_fetch):
        for rec in golden:
            # 12:45 PT, expressed in UTC; the exact hour only fixes today's UTC/ET dates.
            now = datetime.fromisoformat(rec["session"]).replace(
                hour=20, minute=0, tzinfo=timezone.utc
            )
            got = signal_runner.generate_signals(now=now)
            diff = {
                key: (got[key], rec[key])
                for key in ("buys", "sells", "last_exit", "as_of", "universe_size")
                if got[key] != rec[key]
            }
            assert got["bars"]["trimmed_partial_bar"] is False
            total_buys += len(got["buys"])
            total_sells += len(got["sells"])
            if diff:
                mismatches.append((rec["session"], diff))

    assert not mismatches, f"{len(mismatches)} of {len(golden)} sessions differ: {mismatches[:2]}"
    # The totals P4 recorded: a pass that compared two empty lists would not match these.
    assert (total_buys, total_sells) == (1028, 5519)
