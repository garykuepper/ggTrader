# S6: Data and point-in-time universe layer, in vectorbt 1.2.0 terms

Date 2026-11-06. Read-only on the repo and DB (sessions forced `readonly`, no Yahoo/Tiingo calls from the loader, nothing written).
Evidence: `.../vbt-spec/S6-evidence/` (scripts `p1`-`p7`, outputs `*.out`, prototypes `tsdata.py`, `pit.py`).
Stack used: `venv120` = Python 3.12, pandas 3.1.1, numpy 2.5.4, vectorbt 1.2.0, ggTrader loader modules imported unchanged
(a psycopg2/dotenv symlink shim in `S6-evidence/shim/` and `PYTHONPATH=src`). That the loader and index modules import and run on pandas 3 is itself a result.

## 0. Verdicts in six lines

1. **Universe mask: replace the per-bar loop with `pit.py` (51 lines).** 0 mismatches against `eligible_at` on 457 dates x 4 configs (1,828 date-config checks); builds in 0.05 s vs about 370 s (ESTIMATE, extrapolated from the 457-date loop at 0.19 s/date). 1.2 MB.
2. **Read path: a 69-line `vbt.Data` subclass works** on 646 symbols x 1,933 bars: 32 s DB read, 1.07 GB peak RSS, `get('Close')` 0.45 s, `vbt.RSI/MA/ATR` over the result in 0.04-0.11 s each. But OSS `vbt.Data` is only a dict-of-DataFrames holder (no `.run`, `.close`, `.features`, `pull`, `fetch_symbol`; those are PRO). The value is the `download`/`update`/`loc` hooks, not the indicators.
3. **Fetch layer: KEEP for live, do not adopt `YFData` or `AlpacaData`.** `YFData` fails on the exact things the loader was built for (section 3).
4. **Cache layer / TimescaleDB store: KEEP.** Research should stop fetching: read-only DB (or a frozen parquet snapshot, 1.2 s) and an explicit refresh command. That removes the 10+ minute delisted-ticker stalls and makes the negative cache unnecessary on the lab path.
5. **Corporate actions: no vbt glue needed or possible.** Lab bars are provider-adjusted closes (total-return style); `split_check`/`dividend_check` are broker-account reporting only and are untouched by the vbt move.
6. **Data gaps to flag, not fix:** 17 PIT members with no bars at all (1.29% of member-days), 72 symbols whose history ends early, member-day price coverage 95.2% (2019) rising to 99.8% (2025), stale constituent file (last snapshot 2026-06-02), a 1-of-646-symbols final bar in the tape.

## 1. What the current data path guarantees (so nothing is lost)

| Stage | File:line | Guarantees | Does NOT guarantee |
|---|---|---|---|
| Yahoo batch fetch | `data/live/yfinance_loader.py:50-80` | One `yf.download(..., auto_adjust=True, group_by="ticker")` call for all symbols; columns lowercased `(symbol, field)`; index forced tz-aware UTC; all-NaN rows dropped; exceptions return empty frame. `end` pushed +1 day because Yahoo's end is exclusive. | Adjustment basis is fixed at fetch time (see section 5). Delisted symbols silently absent. |
| Tiingo fallback | `data/live/tiingo_loader.py:35-50` | Same output shape; covers some delisted names; 50 req/h free tier = 72 s between requests. | Needs `TIINGO_API_KEY`; ~12 min wasted per run on dead tickers (docstring `negative_cache.py:3-5`). |
| DB read | `cached_yfinance_loader.py:164-205` `_db_fetch` | Exact stock tickers, `venue='yfinance'`; naive-UTC column localised to UTC; pivot to `(symbol, field)`. | No dedupe: pivot would raise on a duplicate `(timestamp, symbol)` (PK prevents it). |
| DB write | `:207-290` `_cache_to_db` | **Strips tz before insert** (the whole-tape corruption fix, lines 235-245); drops rows with NaN close; upsert on PK; `trades=0`. | Not atomic across symbols. |
| Freshness | `:303-396` `fetch_ohlcv` | Per-symbol: full fetch if uncached or first bar > start+7d and not a recorded inception; incremental fetch if last bar older than 1.5 intervals; DB is source of truth afterward. | Partial intraday bar can be cached as final (the reason `refresh_benchmark_tape` overwrites a 30-day tail, `paper/signal_runner.py:30-57`). |
| Inception table | `:101-160` | Stops a 2021 IPO being re-fetched forever. Writes a table (`CREATE IF NOT EXISTS`) on read. | Silent on failure (logs error, treats as unknown). |
| Negative cache | `lab/negative_cache.py` | Opt-in (`use_negative_cache`); 14-day TTL skip list in `ohlcv_no_data`; **live must not use it** (`data.py:78-82`). | Also does DDL on first call. |
| `fetch_stock_ohlcv` | `lab/data.py:63-176` | DB first, then Yahoo for missing, then Tiingo, records new no-data symbols, clips to `[start,end]`, optional `min_coverage`. **Shared by live** (`signal_runner.py:10`, `overlay.py:11`). | Fetches and writes DB as a side effect of a "load". |
| `collapse_daily_duplicates` | `lab/data.py:179-231` | Drops the minority timestamp convention (SPY stray rows). | **Now a no-op on the real tape**: 2019-2026 index has 0 duplicate dates, hour-of-day `{0}` only, Mon-Fri only (`p6_tape_quality.out`). Heuristic, silent. |
| Membership | `index_constituents.py:61-72` | Latest snapshot row dated `<=` asof (same-day effective); tickers normalised `BRK.B -> BRK-B`; raises if asof predates 1996. | File last row **2026-06-02**; later dates reuse it (stale). |
| `eligible_at` | `lab/data.py:288-313` | Rules, in order: PIT member; symbol present in `past`; `len(close.dropna()) >= min_history_bars` (default 400, live/sweeps 60); optional `sorted(eligible)[:max_stocks]` (alphabetical cap). **No top-N-by-notional rule and no sector cap here.** Top-N lives in strategies; the sector cap (`max_sector_count`, static `data/universe/sp500_sectors.json`, not PIT) is applied in `EnsembleSignal.select` (`ensemble.py:146-150`, `registry.py:57-69`). | `max_sector_count` is NOT applied on the sweep mask path (`pit_entry_mask` uses `eligible_at` only). |
| `pit_entry_mask` | `lab/sweep.py:101-125` | Daily `{universe_fn(d, ohlcv.loc[:d])}` per bar; ANDed into entries only, exits untouched (`sweep.py:136-160`). | O(T^2); ~65% of profiled WFO time (F). |
| `eligibility_mask` | `strategies/indicators.py:30-64` | The `to_targets` path: mask keyed on rebalance-date plans, forward-filled, False before first plan. Live passes a single plan at `last_bar`. | A second, coarser mask definition (rebalance granularity). |
| Price field | all | `close` = Yahoo `auto_adjust=True` close (split + dividend adjusted as of fetch time). Open/high/low adjusted the same way. | Raw prices are not stored. |

Live path facts that constrain the design: `signal_runner.generate_signals` fetches `lookback_days=365` for current members (line 60-77), uses `LabConfig(min_history_bars=60)`, and `fetch_stock_ohlcv` may hit Yahoo and write the DB at 12:45 PT. `eligible_at`/`pit_entry_mask` are used only by `lab/` (grep: blend, cli, data, sweep, wfo), never by `paper/`.

## 2. vectorbt 1.2.0 data model (read from the wheel, then prototyped)

Source: `venv120/.../vectorbt/data/base.py`, `custom.py`.

- **Names differ from PRO.** OSS API is `Data.download(symbols, ...)` -> `Data.download_symbol(symbol, **kw)` (classmethod, `base.py:557-612`), `update_symbol`/`update` (`:612-672`), `from_data` (`:477-555`), `get(column)` (`:701`), `concat` (`:675`), `loc/iloc/xs` indexing, `save/load` (pickle). There is **no** `fetch_symbol`, `pull`, `.close/.high` attributes, `features`, or `Data.run`. Verified: `hasattr` all False (`p2_data_vbt.out`). Indicators run on `data.get('Close')`, i.e. a plain dates x symbols DataFrame.
- **Required constructor arg** `download_kwargs` (missing it raises `TypeError`, hit in prototype).
- **Alignment.** `from_data` localises/converts tz if asked, sets `index.freq = inferred_freq` (None for holiday calendars, harmless), then `align_index(missing=...)` and `align_columns(missing=...)`. Defaults: index `nan` (union, with a warning), columns `raise` (the source of "Symbols have mismatching columns"). Different start dates therefore arrive as leading NaN, same as today's pivot.
- **Missing/delisted symbols:** `download` loops `download_symbol` per symbol and has no skip policy. A symbol that returns an empty frame breaks `from_data`. The prototype overrides `download` to do **one bulk query**, drops symbols with no rows into `missing_symbols`, pre-reindexes to the shared index (no warnings), and passes `missing_columns="nan"`.
- **Scale test (real tape, read-only):**

| Step | Result |
|---|---|
| SQL read via `_db_fetch`, 40 / 100 / 662 symbols, 2019-03-01 to 2026-11-05 | 10.9 s / 19.3 s / 51.6 s (first run; 32 s on the second, DB cache warm). Fixed cost about 10 s regardless of size. |
| `TimescaleData.download` 663 requested -> 646 symbols x 1,933 bars x 5 fields | 32.3 s, peak RSS 1.07 GB |
| `get('Close')` stock `Data.concat` (double Python loop, `.loc[:, s] =`) vs my `pd.concat` override | stock about 2.5 s for 646 symbols (0.39 s per 100, extrapolated) vs 0.45 s; values identical. **The override is optional**, not a requirement. |
| `vbt.RSI.run(close, 14)`, `vbt.MA.run(close,[20,50])`, `vbt.ATR.run(h,l,c,14)` | 0.11 s / 0.04 s / 0.09 s; MA20 max abs diff vs `rolling(20).mean()` = 0.0 |
| `Data.loc["2024"]` | works, wrapper index and all 646 symbols preserved |
| parquet snapshot (5 fields) read | 1.15 s, 48 MB on disk |

  Memory is far under the 6 GB cap; the 5-field frames are 1,933 x 646 x 8 B = 10 MB each.
- **tz.** The loader returns tz-aware UTC midnight. Keep that (it is what `lab/` already uses: `pd.Timestamp(start, tz="UTC")`, `rebalance_dates` calls `tz_convert(None)`). **Never pass `tz_convert` that shifts the day, and never write a `Data` index back to the DB**; the only DB writer stays `_cache_to_db`.
- **Gotcha: indicator outputs carry a parameter level.** `vbt.RSI.run(close, window=14).rsi` has columns `(rsi_window, symbol)`; `entries & mask` then fails with `ValueError: cannot join with no overlapping index names`. Fix: `mask.vbt.tile(n_params, keys=ent.columns.levels[0])`, verified aligned (`p5b_param_cols.out`: 3,199 of 3,605 entries survive the mask on 50 symbols x 2 windows). For one parameter, `.droplevel(0, axis=1)`.
- **Honest assessment of `vbt.Data` here:** the OSS class adds about 69 lines for roughly what a `dict[str, DataFrame]` gives, plus `update()`, `loc`, pickling and `stats/plot`. It does not run indicators. Because the lab's `Strategy` protocol and `EnsembleSignal` consume `(symbol, field)` MultiIndex frames, adopting `Data` also needs a thin `to_multiindex()` shim until strategies are rewritten around per-field frames. Recommend: ship `TimescaleData` (it satisfies "use vectorbt's own"), but treat it as an adapter, not an architecture.

## 3. `YFData` and `AlpacaData` as replacements

Run 2026-11-06 with venv120 (yfinance 1.2.0), network read only, nothing persisted (`p3_yfdata.out`, `p3b_yf_dst.out`).

| Behaviour | Result |
|---|---|
| Index | tz-aware UTC, **mixed 04:00 and 05:00** (EDT/EST local midnight): AAPL 2023-2024 gives hour 5: 171 bars, hour 4: 330 bars. Not a clean midnight; writing that to the naive column is exactly the corruption class in CLAUDE.md. |
| Start/end | `download_symbol` converts via `to_tzaware_datetime(start, tz=get_local_tz())` (`custom.py:278-281`, comment: yfinance uses mktime). On this host (America/Los_Angeles) a date-only start is interpreted in local time. `end` exclusive (`end=2024-01-05` returned Jan 2-4). |
| Mixed ETF + stock | `YFData.download(['SPY','AAPL'])` -> `ValueError: Symbols have mismatching columns` (SPY has `Capital Gains`). Works with `missing_columns="drop"`. |
| Delisted/renamed (SIVB, ANTM) | **Crashes the whole call**: `AttributeError: 'Index' object has no attribute 'tz'` (empty frame reaches `from_data`), including when mixed with good symbols. A skip wrapper is the same custom code we have. |
| Adjustment | `Ticker.history` default `auto_adjust=True`: AAPL 2024-01-02 = 183.4040. Our cached tape = 183.40404 (equal to 3e-8), so both are dividend-adjusted, consistent. (One symbol checked; the patchwork risk below is not disproved.) |
| Speed | serial, one HTTP call per symbol: 12 symbols full history 8.2 s, i.e. about 0.7 s/symbol, **about 7.5 min for 646 symbols (ESTIMATE)**, versus one batched `yf.download` today. |
| Extras | Brings `Dividends`, `Stock Splits`, `Capital Gains` columns, which the vbt OSS portfolio ignores. |

`AlpacaData` (`custom.py:768-945`): could not run it (no keys, `alpaca` package not in venv120), so its class behaviour is UNVERIFIED beyond reading. From source: one symbol per request; **`limit=500` default**, so a daily history request returns at most 500 bars (about 2 years) unless the caller raises it, a silent truncation for our 1,900-day windows; `adjustment="all"` (adjusted at request time, same patchwork issue). Via the Alpaca market-data MCP (free IEX feed, read only):

| Probe | Result |
|---|---|
| SIVB, TWTR (delisted), 2022-01-03..06 | **bars returned**, stamped `T05:00:00Z`; `get_asset SIVB` -> 404, so delisted history is available even though the asset is inactive |
| AAPL 2016-01, 2019-03, 2020-07-01..08, ANTM 2019-03 | **empty** |
| AAPL, ELV 2021-01-04..07 | returned; ANTM 2021 empty, ELV returned (renamed ticker resolves under the new name) |
| IEX volume | AAPL 2022-01-03 `v=1,219,127` against typical 100M+ SIP volume: **unusable for notional-volume ranking** |

Conclusion: IEX history does not reach our 2019 warm-up (starts somewhere between 2020-07 and 2021-01); SIP is paid (owner: no paid data). Alpaca stays a live-execution and last-resort-gap source, not a research tape.

**Keep/replace table**

| Layer | Verdict | Reason |
|---|---|---|
| Fetch (`YFinanceDataLoader` batch + Tiingo fallback) | **KEEP** (live frozen to ~12-29; `lab.data.fetch_stock_ohlcv` is imported by the deployed paper path) | `YFData` is serial, tz-ambiguous, crashes on delisted/mixed. No replacement is strictly better. |
| Cache logic in `CachedYFinanceLoader.fetch_ohlcv` + inception table | **KEEP for live; remove from the research read path** | Research then never writes or fetches. |
| Negative cache (`lab/negative_cache.py`) | **Obsolete once research is read-only**; delete after the freeze | It exists only to avoid refetching dead tickers. |
| TimescaleDB `ohlcv` store | **KEEP** | PK `(timestamp, symbol, interval, venue)`, naive UTC, current tape is clean (Mon-Fri 00:00, 0 duplicates). Add a frozen parquet snapshot per study for reproducibility and 1.2 s loads (not a replacement for the store). |
| `collapse_daily_duplicates` | **Replace by a hard assert** (unique index, midnight stamps, weekdays) | It is a silent heuristic for a bug that is now fixed; a violation should fail loudly. |

## 4. Point-in-time universe in vectorbt terms

### 4.1 Mask construction (verified)

`pit.py` replaces `pit_entry_mask`: membership as a step function of the constituent-change snapshots (`searchsorted` + one fancy index into an `n_snapshots x n_symbols` bool table), `close.notna().cumsum() >= min_history_bars`, optional alphabetical `max_stocks` cap (`mask.cumsum(axis=1) <= k` over sorted columns), optional per-sector cap via grouped `cumsum`. It differs from Opus's F-check script, which still looped `universe_members_asof` per day (1.0 s); this version has no per-day Python loop (0.05 s).

Verification (`p4_pit_verify.py`, real tape 1,933 x 646): 457 sampled dates = 160 random (seed 7) + 317 within +/-1 bar of all 114 constituent-change dates in range. Each date compared as a symbol set against `eligible_at(d, ohlcv.loc[:d], cfg)`:

| Config | Mean eligible | Symbol mismatches | `eligible_at` loop | vector build |
|---|---|---|---|---|
| min_history 400 (lab default) | 394.1 | **0** | 88.0 s | 0.05 s |
| min_history 60 (live) | 485.9 | **0** | 87.3 s | 0.05 s |
| 60 + `max_stocks=50` | 49.2 | **0** | 88.7 s | 0.08 s |
| 60 + `max_sector_count=3` (vs `eligible_at` + `apply_sector_constraints`) | 35.4 | **0** | 85.3 s | 0.12 s |

Whole-tape mask: 1.2 MB bool, +0 GB. Extrapolated original cost for all 1,933 bars: about 370 s (ESTIMATE; `p4_pit_verify.json` measured 0.173 s on early dates, 0.19 s/date averaged over the whole sample, and it grows with history length).
Limits of the check: it covers the daily-mask semantics only; the `eligibility_mask(plans)` rebalance-granularity variant is not reproduced (it should simply be fed rows of this daily mask at rebalance dates). The sector-map test inlines the verbatim logic of `apply_sector_constraints` because importing `lab.strategies` needs `defusedxml`, absent from venv120.

### 4.2 Combining with vbt signals (real tape, `p5_mask_in_vbt.out`)

RSI cross signals from `vbt.RSI` on 646 symbols, `Portfolio.from_signals(size=0.03 percent, slippage=0.0005)`, 0.3-0.4 s per simulation:

| Variant | Trades | Entry signals on non-members | Bars held while not a member | Stranded in dead tickers |
|---|---|---|---|---|
| A. no mask (legacy lookahead) | 17,991 | **5,388** | 52,393 | n/a |
| B. `entries & mask`, exits untouched (**current semantics**) | 14,579 | 0 | 8,596 | 8 |
| C. B + exit on index removal (`exits | (member.shift(1) & ~member)`) | 14,579 | 0 | 5,938 | n/a |
| D. C + terminal exit on each series' last valid bar | 14,574 | 0 | 1,525 | 2 |
| E. D + suppress same-bar entry at the terminal bar | 14,574 | 0 | **0** | **0** |

Definitions and consequences:
- **The mask gates entries only.** An open position whose symbol leaves the index keeps running to its normal exit (B). This is what `sweep.py` does today by design (`exits are left alone`, docstring at `sweep.py:138-141`). 143 removal events in this tape.
- **"Exit on removal" (C)** closes at the first non-member bar. It is a policy change, not parity: do not turn it on in the re-baseline arm that must be comparable with history; offer it as a flagged option. Mask-then-time-stop ordering matters: mask must be applied **before** any time-stop is derived from entries (this is already the `to_targets` path; the sweep path masks after, a known divergence in F).
- **Dead price tails are a separate issue vbt does not handle.** When the close turns NaN for good, vbt keeps the position open at the last price forever (DRE: value constant 10,021.84 to the end of the tape). 8 positions were stranded in B. A terminal exit at each series' last valid bar fixes 6; the other 2 (NBL, EQR) fail because an entry fired on that same last bar (default conflict handling ignores the pair). E handles this. The terminal-exit price is the last traded price, so it overstates recovery for failures (flag).
- **Weight strategies** (`from_orders`, `target_percent`): use the mask as `weights.where(mask, 0)` before the order matrix, so a removed symbol targets 0 at its first non-member bar. That is exit-on-removal by construction; document it as the weight-path policy and keep `rebalance_dates` unchanged. `_sweep_fold_weights` starts flat per window (F), so no mask carry-over is needed there.
- **Param-grid sweeps:** tile the mask with `mask.vbt.tile(n, keys=...)` (section 2). Mask memory is trivial; the dense stop arrays are the real memory driver (F: D measured 3.1 GB synthetic).

### 4.3 Where it lives

`ggTrader/lab/universe.py` (new): `membership_mask(index, columns)`, `pit_mask(close, *, min_history_bars, max_stocks=None, max_sector_count=None)`, `terminal_exits(close)`, `removal_exits(member)`. Pure functions, no I/O, no vbt import. `sweep.py` and `wfo.py` call `pit_mask(close, ...)` once per run instead of `pit_entry_mask(ohlcv, universe_fn)` per fold; the `universe_fn` plumbing in `wfo.py` is deleted. Live (`paper/`) is untouched: it already filters members in `generate_signals` and masks via a one-plan `eligibility_mask`.

## 5. Corporate actions

- **Lab:** no explicit events. Bars are Yahoo `auto_adjust=True` closes (and OHLC), i.e. a total-return series; dividends are implicitly reinvested at the ex-date. vbt OSS `Portfolio` has no dividend or cash-flow support (D), and none is needed for this representation. Cash-dividend timing and taxes belong to `TaxLotBook` (out of scope).
- **Patchwork risk (UNVERIFIED, flag):** adjustment factors are fixed at fetch time. Bars appended by a later incremental fetch sit on a newer basis than the bars already cached, so a split or dividend between fetches leaves a step at the join. The loader only does a full refetch for uncached or late-start symbols, not on a corporate action. I could not isolate real splits from real crashes with a price-only scan (53 candidate one-day jumps, most are distressed names: PARA, NKTR, SIVB after failure, AVB = EQR merger tape); AAPL spot-check matched Yahoo to 3e-8. A real check needs Yahoo's split/dividend calendar vs the cache; it is a data-layer test to add, not a vbt task.
- **Live:** `paper/split_check.py` (`find_split_applied_symbols`, `compute_split_corrections`) and `paper/dividend_check.py` (`compute_dividend_accruals`) compare the Alpaca corporate-action feed with `paper_snapshots` and correct reporting only (MNST 2026-08-11). They never touch bars. They stay as they are and sit outside the vbt migration.
- **Glue acceptable:** none for dividends/splits in the simulator. If a raw-price view is ever needed (tax lots), reconstruct it from the Alpaca corporate-actions feed in the tax-lot layer, not in the data adapter.

## 6. Gaps free data cannot close (flag, do not fix)

| Gap | Evidence | Design response |
|---|---|---|
| PIT members with **no bars at all** (renames/acquisitions): ABC, ADS, ANTM, BLL, CBS, COG, CTL, FBHS, FRC, GPS, HCP, HFC, MMC, PKI, RE, TMK, WLTW | 17 symbols, 12,555 of 973,923 member-days (1.29%), 2019-2026 (`p7_coverage.out`) | `Bars.missing` list returned and written into every run's JSON; the mask is False for them. A ticker-alias table (ANTM->ELV, ABC->COR) is a symbology fix and is a separate decision (Alpaca MCP resolved ELV for 2021, so aliases are findable). |
| Price coverage of PIT member-days | 95.2% (2019), 96.0% (2020), 96.4% (2021), 97.7% (2022), 98.8% (2023), 99.7% (2024), 99.8% (2025), 99.3% (2026). Early-window survivorship is the largest | Report `coverage_pct` per year beside every number (existing `coverage_stats` already computes it); never silently fill. |
| Early-ending series | 72 symbols with last bar >10 days before tape end (ABMD 2023-01, AGN 2020-05, ATVI 2023-10 ...). Memory's "72 delisted tickers" is consistent; I did not verify it is the same set | Terminal exit at last valid bar (variant E); flag last-price recovery as optimistic; keep in a `delisted` list in the run JSON. |
| Post-failure junk bars (SIVB, SBNY, FRC) | SIVB +54% on 2023-06-15 after failure; the PIT mask excludes them once they leave the index | Keep mask-before-signals; no data repair. |
| Interior gaps | 1,515 NaN cells between first and last valid bar | Leave as NaN; vbt skips orders on NaN price. Count in the run report. |
| Ragged tape end | At snapshot time the last bar (2026-11-05) held **1 of 646** closes; the previous bar 573/646 (73 of those are the dead names). Cause not diagnosed (SPY-only refresh vs full-universe bars) | Loader contract: report coverage of the last bar and drop/flag trailing rows below a coverage threshold before computing signals. Check with `daily-trader-check` whether the live `close.index[-1]` can ever be that SPY-only row. |
| Stale constituent file | Last snapshot 2026-06-02 (`index_constituents`); later dates reuse it | Add an age check to the mask builder (raise or warn when `index.max() - last_snapshot > N` days). |
| Volume | Stored volume is Yahoo consolidated (non-zero NaN-free); 1,746 zero-volume cells with a close. IEX volume is unusable | Top-N by notional (`close*volume`) stays computable from the tape; do not source volume from Alpaca free. |
| Paid data | Owner refuses | Nothing to design. |

## 7. Recommended data-layer design

```
src/ggTrader/data/
  bars.py          NEW  ~70 lines  TimescaleData(vbt.Data), ReadOnlyLoader, load_bars()
  snapshot.py      NEW  ~30 lines  freeze/read parquet snapshot + sha256 into run JSON
  live/            UNCHANGED until after the ~12-29 freeze (yfinance_loader, cached_yfinance_loader, tiingo_loader)
src/ggTrader/lab/
  universe.py      NEW  ~55 lines  membership_mask, pit_mask, terminal_exits, removal_exits (pure)
  data.py          SHRINKS         keep STOCK_BASE_CONFIG, rebalance_dates, equity_universe_between;
                                   fetch_stock_ohlcv stays for live until freeze, then splits into
                                   live-only fetch (kept in data/live) and research load_bars
  negative_cache.py DELETE after freeze
```

Public functions:
- `load_bars(symbols, start, end=None, *, snapshot=None) -> TimescaleData`: read-only DB or parquet snapshot; no network, no DDL, no writes; returns `missing_symbols`; asserts unique index, midnight stamps, weekdays (replaces `collapse_daily_duplicates`); `.get("Close")` etc. via the Data hooks. Ship `to_multiindex()` as a temporary shim for strategies.
- `pit_mask(close, *, min_history_bars=400, max_stocks=None, max_sector_count=None) -> DataFrame[bool]`.
- `terminal_exits(close)`, `removal_exits(member)`: optional policy toggles, off by default for parity.
- `refresh_bars(symbols, start, end)`: the only research-side function allowed to fetch; wraps the existing loader's fetch + `_cache_to_db`; explicit, logged, never called from `ggt lab`.

Sequencing guard (live is frozen to ~12-29): everything above except removals is additive. Do not touch `lab/data.fetch_stock_ohlcv` while `paper/signal_runner.py` and `paper/overlay.py` import it. Moving `pit_mask` into `wfo.py`/`sweep.py` is lab-only and needs only the oracle gate from F (mask identical on every oracle date).

Line estimates (all ESTIMATE; prototype counts are exact: tsdata.py 69, pit.py 51):
- Added: about 70 (bars) + 30 (snapshot) + 55 (universe) = about 155.
- Removed after the freeze: `pit_entry_mask` 25 + `eligible_at` 26 + `universe_fn` plumbing in `wfo.py`/`blend.py`/`cli.py` (about 20) + `collapse_daily_duplicates` 53 (to a 6-line assert) + research branches of `fetch_stock_ohlcv` (Tiingo, negative cache, gap-fetch, about 90) + `negative_cache.py` 73 = about 290.
- Net: about -135 lines, and the research path loses every network and DDL side effect. That net is small next to the speed and determinism gain; do not oversell it as a code-size win.

## 8. Risks and what I did not prove

- `ReadOnlyLoader` sets `set_session(readonly=True)` on the same SELECT code; I did not attempt a write to prove rejection. It reuses private `_db_fetch`; promote it to a public read function when this ships.
- `TimescaleData.update()` is not implemented or tested (OSS `Data.update` needs `update_symbol`; for a DB cache the right update is "re-read", so skip it).
- Mask equivalence is shown on the daily-mask semantic for 4 configs and 457 dates, not for `eligibility_mask(plans)` and not for the other universes (`nasdaq100`, `midcap400` use static snapshots, trivially a constant mask).
- DB read time is dominated by about 10 s fixed cost and varies with DB cache state (51.6 s vs 32 s for the same query); the snapshot, not the DB, should be what repeated WFO runs read.
- The ragged-last-bar finding is a point-in-time observation on 2026-11-06 and may be transient.
- Variants C/D/E change numbers by policy; they were tested only on RSI toy signals with equal 3% sizing, not on the ensemble. They show mechanism, not P&L impact.
