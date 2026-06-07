# Per-Venue Availability Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track which coins are tradeable on each venue (Kraken, Binance.US) as a persisted, version-controlled snapshot, and intersect the mover-ranking universe against the execution venue's snapshot before the top-N cut.

**Architecture:** Two layers. Layer 1 — a `fetch_venue_listings()` helper + a `update_venue_listings.py` CLI write a per-venue JSON snapshot of active USD spot pairs to `data/universe/{venue}_listings.json`. Layer 2 — the existing `update_universe_ccxt.py` ranker loads that snapshot and intersects its live-volume candidates against it. OHLCV stays venue-agnostic; venue enters only via the listings snapshot and the live volume figure.

**Tech Stack:** Python 3.11, `ccxt` (exchange listings/tickers), `pytest` + `monkeypatch` (mocked — no live API in tests). Test runner: `.venv/bin/python -m pytest`. Source layout: packages under `src/`, importable as `ggTrader.*`.

**Reference spec:** `docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md`

---

## File Structure

- **Create** `src/ggTrader/data/core/venue_listings.py` — all Layer-1 logic + the pure intersection helper. One responsibility: per-venue availability. Functions: `fetch_venue_listings`, `write_venue_listings`, `load_venue_listing_symbols`, `filter_to_listed`.
- **Create** `scripts/update_venue_listings.py` — thin CLI wrapper over `write_venue_listings`.
- **Modify** `scripts/update_universe_ccxt.py` — import the helpers, add a `listings_dir` param, insert the intersection step before the volume sort/floor/cap.
- **Create** `tests/test_venue_listings.py` — all unit tests (filtering, normalization/dedup, load, atomic write, ranker intersection).
- **Modify** `.gitignore` — explicitly keep `data/universe/*_listings.json` tracked.
- **Create** (generated, committed) `data/universe/kraken_listings.json`, `data/universe/binanceus_listings.json`.

Existing constants reused (do not redefine): `SYMBOL_MAPPING`, `STABLE_BASES` in `src/ggTrader/data/core/constants.py`.

---

### Task 1: `fetch_venue_listings` — snapshot active USD spot pairs

**Files:**
- Create: `src/ggTrader/data/core/venue_listings.py`
- Test: `tests/test_venue_listings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_venue_listings.py`:

```python
import json

import pytest

from ggTrader.data.core import venue_listings
from ggTrader.data.core.venue_listings import fetch_venue_listings


# base/quote/active/spot mirror the ccxt market structure
FAKE_MARKETS = {
    "BTC/USD": {"base": "BTC", "quote": "USD", "active": True, "spot": True},
    "XXBT/USD": {"base": "XXBT", "quote": "USD", "active": True, "spot": True},   # maps to BTC -> dedup
    "ETH/USD": {"base": "ETH", "quote": "USD", "active": True, "spot": True},
    "SOL/USDT": {"base": "SOL", "quote": "USDT", "active": True, "spot": True},    # non-USD quote -> drop
    "OLD/USD": {"base": "OLD", "quote": "USD", "active": False, "spot": True},     # inactive -> drop
    "BTC/USD:USD": {"base": "BTC", "quote": "USD", "active": True, "spot": False}, # perp -> drop
    "USDT/USD": {"base": "USDT", "quote": "USD", "active": True, "spot": True},    # stable base -> drop
}


class _FakeExchange:
    def load_markets(self):
        return FAKE_MARKETS


@pytest.fixture
def fake_kraken(monkeypatch):
    monkeypatch.setitem(venue_listings.SUPPORTED_VENUES, "kraken", _FakeExchange)


def test_fetch_keeps_only_active_usd_spot_nonstable(fake_kraken):
    listings = fetch_venue_listings("kraken")
    symbols = [e["symbol"] for e in listings]
    # BTC (deduped from BTC/USD + XXBT/USD) and ETH only; sorted
    assert symbols == ["BTC", "ETH"]


def test_fetch_normalizes_and_keeps_first_ccxt_symbol(fake_kraken):
    listings = fetch_venue_listings("kraken")
    btc = next(e for e in listings if e["symbol"] == "BTC")
    assert btc["ccxt_symbol"] == "BTC/USD"
    assert btc["base"] == "BTC"
    assert btc["quote"] == "USD"


def test_fetch_unsupported_venue_raises():
    with pytest.raises(ValueError):
        fetch_venue_listings("coinbase")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.data.core.venue_listings'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ggTrader/data/core/venue_listings.py`:

```python
"""Per-venue tradeable-coin availability snapshots (Layer 1 of the universe pipeline).

OHLCV price/volume is effectively venue-agnostic, but *which* coins are listed
differs by venue. This module captures a current snapshot of the active USD spot
pairs on a venue so the mover ranker can intersect its candidates against a
tracked, version-controlled availability set. See
docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md.
"""

from __future__ import annotations

import ccxt

from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING

# Exchanges we support generating availability snapshots for.
SUPPORTED_VENUES = {"kraken": ccxt.kraken, "binanceus": ccxt.binanceus}

DEFAULT_LISTINGS_DIR = "data/universe"


def fetch_venue_listings(venue: str) -> list[dict]:
    """Fetch the active USD spot listings for a venue.

    Args:
        venue: Exchange name; one of ``SUPPORTED_VENUES`` (case-insensitive).

    Returns:
        Sorted, de-duplicated list of ``{symbol, ccxt_symbol, base, quote}`` dicts,
        where ``symbol`` is the normalized base (e.g. ``"BTC"``). One entry per
        normalized base; the first ccxt market id encountered wins.

    Raises:
        ValueError: If ``venue`` is not supported.
    """
    venue = venue.lower()
    exchange_cls = SUPPORTED_VENUES.get(venue)
    if exchange_cls is None:
        raise ValueError(
            f"Unsupported venue: {venue!r} (expected one of {sorted(SUPPORTED_VENUES)})"
        )
    exchange = exchange_cls()
    markets = exchange.load_markets()

    listings: list[dict] = []
    for ccxt_symbol, market in markets.items():
        if not (market.get("active") and market.get("spot")):
            continue
        if market.get("quote") != "USD":
            continue
        base = market.get("base") or ""
        if "." in base or ":" in base:
            continue
        std_base = SYMBOL_MAPPING.get(base, base)
        if std_base in STABLE_BASES:
            continue
        listings.append(
            {
                "symbol": std_base,
                "ccxt_symbol": ccxt_symbol,
                "base": std_base,
                "quote": "USD",
            }
        )

    listings.sort(key=lambda e: e["symbol"])

    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in listings:
        if entry["symbol"] in seen:
            continue
        seen.add(entry["symbol"])
        deduped.append(entry)
    return deduped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/data/core/venue_listings.py tests/test_venue_listings.py
git commit -m "feat(universe): fetch_venue_listings snapshot of active USD spot pairs"
```

---

### Task 2: `load_venue_listing_symbols` + `filter_to_listed`

**Files:**
- Modify: `src/ggTrader/data/core/venue_listings.py`
- Test: `tests/test_venue_listings.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_venue_listings.py`:

```python
from ggTrader.data.core.venue_listings import (
    filter_to_listed,
    load_venue_listing_symbols,
)


def test_filter_to_listed_drops_unlisted():
    candidates = [{"symbol": "BTC"}, {"symbol": "FOO"}, {"symbol": "ETH"}]
    kept = filter_to_listed(candidates, {"BTC", "ETH"})
    assert [c["symbol"] for c in kept] == ["BTC", "ETH"]


def test_load_symbols_returns_set(tmp_path):
    (tmp_path / "kraken_listings.json").write_text(
        json.dumps({"listings": [{"symbol": "BTC"}, {"symbol": "ETH"}]})
    )
    symbols = load_venue_listing_symbols("kraken", listings_dir=str(tmp_path))
    assert symbols == {"BTC", "ETH"}


def test_load_symbols_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_venue_listing_symbols("kraken", listings_dir=str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -k "filter_to_listed or load_symbols" -v`
Expected: FAIL with `ImportError: cannot import name 'filter_to_listed'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ggTrader/data/core/venue_listings.py`:

```python
import json
from pathlib import Path


def filter_to_listed(
    candidates: list[dict], listed_symbols: set[str], key: str = "symbol"
) -> list[dict]:
    """Keep only candidates whose ``key`` value is in ``listed_symbols``.

    Pure function; preserves input order.
    """
    return [c for c in candidates if c[key] in listed_symbols]


def load_venue_listing_symbols(
    venue: str, listings_dir: str = DEFAULT_LISTINGS_DIR
) -> set[str]:
    """Load the set of available ``symbol`` values from a venue's snapshot.

    Raises:
        FileNotFoundError: If the snapshot does not exist (with a hint to run the
            ``update_venue_listings`` command). The ranker must fail loud rather
            than silently skip the availability filter.
    """
    venue = venue.lower()
    path = Path(listings_dir) / f"{venue}_listings.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No availability snapshot at {path}. Run: "
            f"python scripts/update_venue_listings.py --venue {venue}"
        )
    with open(path) as f:
        payload = json.load(f)
    return {entry["symbol"] for entry in payload.get("listings", [])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/data/core/venue_listings.py tests/test_venue_listings.py
git commit -m "feat(universe): load_venue_listing_symbols + filter_to_listed helpers"
```

---

### Task 3: `write_venue_listings` — atomic snapshot writer with empty guard

**Files:**
- Modify: `src/ggTrader/data/core/venue_listings.py`
- Test: `tests/test_venue_listings.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_venue_listings.py`:

```python
from ggTrader.data.core.venue_listings import write_venue_listings


def test_write_creates_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        venue_listings,
        "fetch_venue_listings",
        lambda v: [{"symbol": "BTC", "ccxt_symbol": "BTC/USD", "base": "BTC", "quote": "USD"}],
    )
    out = write_venue_listings("binanceus", listings_dir=str(tmp_path))
    payload = json.loads(out.read_text())
    assert payload["venue"] == "binanceus"
    assert payload["count"] == 1
    assert payload["listings"][0]["symbol"] == "BTC"
    assert "updated_at" in payload


def test_write_empty_preserves_existing(monkeypatch, tmp_path):
    existing = tmp_path / "kraken_listings.json"
    existing.write_text('{"sentinel": true}')
    monkeypatch.setattr(venue_listings, "fetch_venue_listings", lambda v: [])
    with pytest.raises(RuntimeError):
        write_venue_listings("kraken", listings_dir=str(tmp_path))
    # existing good snapshot must be untouched
    assert existing.read_text() == '{"sentinel": true}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -k write -v`
Expected: FAIL with `ImportError: cannot import name 'write_venue_listings'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ggTrader/data/core/venue_listings.py` (note: `import os` and `from datetime import ...` go with the other imports at the top of the file):

```python
import os
from datetime import datetime, timezone


def write_venue_listings(venue: str, listings_dir: str = DEFAULT_LISTINGS_DIR) -> Path:
    """Fetch and atomically write a venue's availability snapshot.

    Writes to a temp file then ``os.replace`` so a partial/failed write never
    clobbers an existing good snapshot. Refuses to write an empty listing set
    (treated as a fetch failure) to avoid wiping a valid snapshot.

    Returns:
        Path to the written ``{venue}_listings.json``.

    Raises:
        RuntimeError: If the fetched listing set is empty.
    """
    venue = venue.lower()
    listings = fetch_venue_listings(venue)
    if not listings:
        raise RuntimeError(
            f"Refusing to write empty listings for {venue!r}; "
            "aborting to preserve any existing snapshot."
        )

    out_dir = Path(listings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{venue}_listings.json"

    payload = {
        "venue": venue,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(listings),
        "listings": listings,
    }

    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/data/core/venue_listings.py tests/test_venue_listings.py
git commit -m "feat(universe): write_venue_listings atomic snapshot writer with empty guard"
```

---

### Task 4: `update_venue_listings.py` CLI (Layer 1 command)

**Files:**
- Create: `scripts/update_venue_listings.py`

- [ ] **Step 1: Write the script**

Create `scripts/update_venue_listings.py`:

```python
"""CLI: write per-venue availability snapshots (Layer 1 of the universe pipeline).

    python scripts/update_venue_listings.py --venue all
    python scripts/update_venue_listings.py --venue binanceus

Writes data/universe/{venue}_listings.json. See
docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md.
"""

from __future__ import annotations

import argparse
import os

from ggTrader.data.core.venue_listings import (
    DEFAULT_LISTINGS_DIR,
    SUPPORTED_VENUES,
    write_venue_listings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-venue availability snapshots.")
    parser.add_argument(
        "--venue",
        type=str,
        default=(os.getenv("EXCHANGE") or "kraken").lower(),
        choices=[*sorted(SUPPORTED_VENUES), "all"],
        help="Venue to snapshot, or 'all' (default: $EXCHANGE env, else 'kraken').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_LISTINGS_DIR,
        help=f"Output directory (default: {DEFAULT_LISTINGS_DIR}).",
    )
    args = parser.parse_args()

    venues = sorted(SUPPORTED_VENUES) if args.venue == "all" else [args.venue]
    for venue in venues:
        print(f"Fetching live listings for {venue}...")
        out_path = write_venue_listings(venue, listings_dir=args.out_dir)
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses (no live call)**

Run: `.venv/bin/python scripts/update_venue_listings.py --help`
Expected: usage text listing `--venue {binanceus,kraken,all}` and `--out-dir`.

- [ ] **Step 3: Commit**

```bash
git add scripts/update_venue_listings.py
git commit -m "feat(universe): update_venue_listings CLI (Layer 1 snapshot command)"
```

---

### Task 5: Wire the intersection into `update_universe_ccxt.py` (Layer 2)

**Files:**
- Modify: `scripts/update_universe_ccxt.py` (imports near top; `generate_ccxt_universe` signature ~line 73; insert filter just before `candidates.sort(...)` ~line 141)
- Test: `tests/test_venue_listings.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_venue_listings.py`:

```python
import importlib.util
from pathlib import Path as _Path


def _load_ranker():
    path = _Path(__file__).resolve().parent.parent / "scripts" / "update_universe_ccxt.py"
    spec = importlib.util.spec_from_file_location("update_universe_ccxt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRankExchange:
    id = "kraken"

    def load_markets(self):
        return {}

    def fetch_tickers(self):
        return {
            "BTC/USD": {"quoteVolume": 1000, "last": 1, "baseVolume": 1000},
            "FOO/USD": {"quoteVolume": 900, "last": 1, "baseVolume": 900},
            "ETH/USD": {"quoteVolume": 800, "last": 1, "baseVolume": 800},
        }


def test_ranker_drops_unlisted_before_topn(monkeypatch, tmp_path):
    ranker = _load_ranker()
    monkeypatch.setattr(ranker.ccxt, "kraken", lambda: _FakeRankExchange())

    # snapshot lists BTC and ETH but NOT FOO
    (tmp_path / "kraken_listings.json").write_text(
        json.dumps({"listings": [{"symbol": "BTC"}, {"symbol": "ETH"}]})
    )

    out_path = tmp_path / "out.json"
    ranker.generate_ccxt_universe(
        limit=10,
        output_path=str(out_path),
        window="24h",
        venue="kraken",
        min_volume=0.0,
        listings_dir=str(tmp_path),
    )

    results = json.loads(out_path.read_text())
    symbols = {r["symbol"] for r in results}
    assert "FOO" not in symbols
    assert {"BTC", "ETH"} <= symbols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -k ranker -v`
Expected: FAIL — `TypeError: generate_ccxt_universe() got an unexpected keyword argument 'listings_dir'`

- [ ] **Step 3: Add the import**

In `scripts/update_universe_ccxt.py`, below the existing line `from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING`, add:

```python
from ggTrader.data.core.venue_listings import (
    DEFAULT_LISTINGS_DIR,
    filter_to_listed,
    load_venue_listing_symbols,
)
```

- [ ] **Step 4: Add the `listings_dir` parameter**

In `scripts/update_universe_ccxt.py`, change the `generate_ccxt_universe` signature from:

```python
def generate_ccxt_universe(
    limit: int = 50,
    output_path: str = "data/top_50_ccxt_volume.json",
    window: str = "24h",
    venue: str | None = None,
    min_volume: float = 0.0,
):
```

to:

```python
def generate_ccxt_universe(
    limit: int = 50,
    output_path: str = "data/top_50_ccxt_volume.json",
    window: str = "24h",
    venue: str | None = None,
    min_volume: float = 0.0,
    listings_dir: str = DEFAULT_LISTINGS_DIR,
):
```

- [ ] **Step 5: Insert the intersection step**

In `scripts/update_universe_ccxt.py`, find the end of the candidate-building loop, immediately before:

```python
    # Sort by 24h volume for the initial filter
    candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
```

Insert this block directly above that comment (same indentation, inside `generate_ccxt_universe`):

```python
    # Layer-1 availability intersection: keep only coins present in the venue's
    # committed listings snapshot, BEFORE the volume floor / top-N cut. Fails loud
    # if the snapshot is missing (no silent fall-through to an unfiltered universe).
    listed_symbols = load_venue_listing_symbols(venue, listings_dir=listings_dir)
    before = len(candidates)
    candidates = filter_to_listed(candidates, listed_symbols)
    print(
        f"Availability filter ({venue}): {len(candidates)}/{before} USD candidates "
        f"are in the listings snapshot."
    )

```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py -v`
Expected: PASS (9 passed)

- [ ] **Step 7: Run the full suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_venue_listings.py tests/test_signals.py -v`
Expected: PASS (no import or signature breakage in the ranker module)

- [ ] **Step 8: Commit**

```bash
git add scripts/update_universe_ccxt.py tests/test_venue_listings.py
git commit -m "feat(universe): intersect mover ranking with venue listings snapshot"
```

---

### Task 6: Generate, track, and commit the initial snapshots

**Files:**
- Modify: `.gitignore`
- Create (generated): `data/universe/kraken_listings.json`, `data/universe/binanceus_listings.json`

> This step requires live network access (public ccxt `load_markets`, no API key).
> It satisfies the new hard dependency introduced in Task 5 — after this, the
> ranker has snapshots to intersect against.

- [ ] **Step 1: Keep the snapshots tracked in git**

In `.gitignore`, directly below the line `!data/top_50_ccxt_volume.json`, add:

```gitignore

# Per-venue availability snapshots (Layer 1 registry) — keep tracked
!data/universe/
!data/universe/*_listings.json
```

- [ ] **Step 2: Generate both snapshots (live)**

Run: `.venv/bin/python scripts/update_venue_listings.py --venue all`
Expected: prints `Wrote data/universe/kraken_listings.json` and `Wrote data/universe/binanceus_listings.json`.

- [ ] **Step 3: Sanity-check the output**

Run: `.venv/bin/python -c "import json; d=json.load(open('data/universe/binanceus_listings.json')); print(d['venue'], d['count'], [e['symbol'] for e in d['listings'][:5]])"`
Expected: `binanceus <N> [...]` with a plausible coin count (Binance.US is thin — typically a few dozen USD spot pairs) and recognizable symbols (BTC, ETH, etc.).

- [ ] **Step 4: Confirm git will track them**

Run: `git check-ignore data/universe/binanceus_listings.json; echo "exit=$?"`
Expected: `exit=1` (NOT ignored). If it prints the path with `exit=0`, the gitignore allow-rule from Step 1 is missing or misordered — fix before committing.

- [ ] **Step 5: Commit**

```bash
git add .gitignore data/universe/kraken_listings.json data/universe/binanceus_listings.json
git commit -m "feat(universe): commit initial Kraken + Binance.US listings snapshots"
```

---

## Self-Review

**Spec coverage:**
- Layer 1 snapshot per venue → Tasks 1, 3, 4, 6.
- Layer 2 intersection before top-N, volume ranking unchanged → Task 5.
- JSON format `{venue, updated_at, count, listings:[{symbol,ccxt_symbol,base,quote}]}` → Task 3.
- Deterministic ordering (git-friendly diffs) → Task 1 (`listings.sort`).
- Missing snapshot fails loud → Task 2 (`load_venue_listing_symbols` raises) + Task 5 (no fallback).
- Atomic write, never overwrite with partial/empty → Task 3 (`os.replace` + empty guard).
- Live ticker volume retained → Task 5 leaves the existing `fetch_tickers` path untouched.
- Tests all mock the API → Tasks 1-5 use `monkeypatch`/fakes; only Task 6 hits live.
- Out-of-scope items (history, price-movement ranking, DB table, consumer migration) → not implemented. ✓

**Type/name consistency:** `fetch_venue_listings`, `write_venue_listings`, `load_venue_listing_symbols`, `filter_to_listed`, `SUPPORTED_VENUES`, `DEFAULT_LISTINGS_DIR` used identically across module, CLI, ranker, and tests. `generate_ccxt_universe(..., listings_dir=...)` matches the Task 5 call site in the test. Snapshot key `symbol` is the normalized base in both producer (Task 1) and consumer (`filter_to_listed` default key, Task 2). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✓
