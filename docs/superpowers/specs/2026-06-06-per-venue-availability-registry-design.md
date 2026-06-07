# Per-Venue Availability Registry + Venue-Filtered Mover Ranking

**Date:** 2026-06-06
**Status:** Approved design, pending implementation plan

## Problem & Motivation

OHLCV price/volume data is effectively venue-agnostic — the same coin prices
nearly identically on Kraken and Binance.US, so a single shared `ohlcv` series
is fine for backtesting price action regardless of execution venue. The thing
that genuinely differs by venue is **which coins are actually listed and
tradeable**. That distinction only bites at one point: when measuring/ranking
the highest movers to pick a trading universe, you can only select coins listed
on your execution venue.

Today the codebase handles venue selection *implicitly*: `update_universe_ccxt.py`
queries the chosen venue live (`fetch_tickers` / `load_markets`), so its ranked
output is inherently venue-specific, and the research pipeline already switches
venue via the `EXCHANGE` env var (passing `--venue` through, caching per-venue in
`universe_cache`). What's missing is what the user explicitly asked for:

1. A **persisted, inspectable per-venue availability record** — a tracked list of
   "what coins are available on Kraken vs Binance.US" — rather than an implicit
   set computed live and buried inside ranked output.
2. An **explicit intersection** of the mover ranking against that record before
   the top-N cut.

## Decisions (from brainstorming)

- **Tracking scope:** current snapshot only. No delisting history / point-in-time
  record. The snapshot is refreshed when the Layer-1 command
  (`update_venue_listings`) is run — not on every ranker run (see Deliberate
  freshness behavior). Schema kept simple; history is explicitly out of scope.
- **Layer split:** two layers — a pure per-venue availability snapshot (Layer 1),
  and a ranking step (Layer 2) that reads it and intersects before top-N.
- **Ranking metric:** unchanged. Keep the existing 30-day volume ranking; only
  insert the venue-availability intersection. No price-movement or composite
  scoring.
- **Volume source:** live exchange ticker (current behavior). Venue enters via
  both availability AND volume at ranking time. Ranking still requires a live API
  call; it is not moved to the shared `ohlcv` table.
- **Approach:** A — a shared `fetch_venue_listings()` helper used by both a
  standalone snapshot command and the ranker (DRY + decoupled).

## Architecture

```text
load_markets(venue) ──► data/universe/{venue}_listings.json      (Layer 1, git-committed)
                                   │
live fetch_tickers(venue) ─ rank by 30d $vol ─ ∩ listings ─ floor ─ top-N ─► selected universe (Layer 2)
```

### Components

1. **`fetch_venue_listings(venue) -> list[dict]`** — new shared helper.
   - Location: `src/ggTrader/data/core/venue_listings.py` (importable by both the
     snapshot CLI and the ranker; keeps `update_universe_ccxt.py` from being the
     import source for production code).
   - Calls `ccxt.{venue}().load_markets()`.
   - Keeps markets where `active and spot and quote == 'USD'`.
   - Skips stable bases (reuse `STABLE_BASES`) and weird/index/derivative symbols
     (base containing `.` or `:`), matching the existing ranker's filters.
   - Normalizes base via existing `SYMBOL_MAPPING`.
   - Returns `[{symbol, ccxt_symbol, base, quote}]`, where `symbol` is the
     normalized base (e.g. `"BTC"`), matching the key the ranker already uses
     internally (`standard_base`). `base` carries the raw pre-normalization ccxt
     base (e.g. `"XXBT"`) for audit/debug; consumers match on `symbol`, not `base`.
   - Venue guard: only `kraken`, `binanceus` supported; else `ValueError`
     (reuse the existing `{"kraken": ccxt.kraken, "binanceus": ccxt.binanceus}`
     mapping pattern).

2. **`scripts/update_venue_listings.py`** — thin Layer-1 CLI.
   - Args: `--venue {kraken|binanceus|all}` (default from `EXCHANGE` env, else
     `kraken`); `--out-dir data/universe` (default).
   - For each venue, calls `fetch_venue_listings`, writes
     `data/universe/{venue}_listings.json` sorted deterministically.
   - This is the tracked availability registry; committed to git.

3. **`scripts/update_universe_ccxt.py`** (Layer 2) — gains an intersection step.
   - Before the volume floor / top-N cut, load
     `data/universe/{venue}_listings.json` and intersect the candidate set
     against its `symbol` set.
   - Keep the existing live `fetch_tickers` volume ranking otherwise unchanged.
   - Output (`top_…volume.json` / per-run `top_ccxt_volume.json`) unchanged in
     shape; consumers untouched.

### Data format

`data/universe/{venue}_listings.json`:

```json
{
  "venue": "binanceus",
  "updated_at": "2026-06-06T15:30:00Z",
  "count": 11,
  "listings": [
    {"symbol": "BTC", "ccxt_symbol": "BTC/USD", "base": "XXBT", "quote": "USD"}
  ]
}
```

`listings` sorted by `symbol` so git diffs are meaningful and runs are
deterministic.

## Deliberate freshness behavior

The ranker filters live candidates against the **committed** snapshot, not a
fresh `load_markets()` call. A coin newly listed on the venue but not yet in the
snapshot is excluded until `update_venue_listings` is re-run. This is
intentional: it makes "what can I trade" a reviewed, version-controlled decision
rather than whatever the API returned that second. Refresh is a single command.

## Error handling

- **Missing listings file** in the ranker → fail loud with a message instructing
  the user to run `update_venue_listings --venue X` first. No silent
  fall-through to an unfiltered universe.
- **Unsupported venue** → `ValueError` (reuse existing guard).
- **ccxt / network failure in Layer 1** → exit non-zero and leave the existing
  committed snapshot untouched. Never overwrite a good file with a partial or
  empty set.

## Testing

All tests mock `load_markets()` / `fetch_tickers()` — no live API.

- **Unit — `fetch_venue_listings` filtering:** mocked `load_markets()` payload
  containing active/inactive, USD/non-USD, spot/derivative, and stable-base
  markets; assert only active USD spot non-stable pairs survive and bases are
  normalized via `SYMBOL_MAPPING`.
- **Unit — ranker intersection:** a candidate present in live tickers but absent
  from the snapshot is dropped before the top-N cut; a candidate present in both
  survives.
- **Unit — missing snapshot:** ranker raises a clear error when the listings file
  is absent (no silent skip).
- **Unit — Layer 1 atomicity:** a simulated ccxt failure does not overwrite an
  existing committed snapshot.

## Out of scope (YAGNI)

- Delisting / point-in-time history (snapshot only).
- Price-movement or composite mover ranking (volume ranking unchanged).
- Moving volume ranking to the shared `ohlcv` table (stays on live ticker).
- DB-backed listings table (JSON files chosen).
- Migrating consumers — they already resolve venue via `EXCHANGE` and read the
  ranker's existing output shape.
