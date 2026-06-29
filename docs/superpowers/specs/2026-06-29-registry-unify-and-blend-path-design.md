# Design: Unified Strategy Registry + First-Class `--blend` Path

**Date:** 2026-06-29
**Status:** Approved (brainstorm) — proceeding to plan
**Motivation:** Architecture review (2026-06-28) found the strategy registry triplicated and hand-synced, and the portfolio-of-sleeves blend logic (`allocation.py`) built+validated but reachable only through throwaway research scripts. This makes "trying combinations of strategies" a bespoke-script exercise. This work makes blending a first-class, persisted `ggt lab` capability and collapses the registry to one source of truth.

## Goals

1. **Single-source strategy registry** — one map; everything else derived. Adding a strategy = one line.
2. **`ggt lab --blend` path** — blend N `strategy@universe` sleeves through the gated WFO using the validated rolling inverse-vol → target-vol overlay, persisted as a normal lab run.

## Non-Goals

- **NOT to reopen equity diversification.** The 3-way equity blend is a closed NO-GO (research plan §4-B: blended Sharpe 1.05 < 1.12 core). This is infrastructure; its first intended real use is the parked orthogonal **crypto-carry** sleeve.
- No new weighting schemes beyond the validated inverse-vol/target-vol (no fixed 50/50, 70/30 modes — YAGNI; the closed equity arc is where those were used).
- No decorator-based auto-registration (keeps the codebase's explicit-dict style).
- No changes to the strategy classes themselves (the Voter/Combiner decomposition is explicitly deferred per the architecture review).

## Part A — Single-Source Strategy Registry

### Current state (the problem)
The name→class mapping is duplicated across three hand-synced sites:
- `strategies/__init__.py:18-33` — `STRATEGY_REGISTRY` (all 13).
- `strategies/signals.py:604-646` — `_build_signal_registry()` + a separate `SIGNAL_STRATEGY_NAMES` tuple + `build_signal_strategy()`.
- `strategies/momentum.py:75-80` — `_REGISTRY` + `STRATEGY_NAMES` + `build_strategy()`.

No test asserts they agree. Adding a signal strategy = 4 edits across 3 files.

### Design
`STRATEGY_REGISTRY` in `strategies/__init__.py` stays the **one** source of truth (it already lists all 13). All names/builders derive from it, keyed off each class's existing `target_kind` attribute.

**New module `strategies/registry.py`** — the derivation helpers. It must NOT import the strategy classes at module top level (that would re-introduce the cycle); instead it imports `STRATEGY_REGISTRY` lazily *inside* each function:
```python
def _registry() -> dict[str, type]:
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    return STRATEGY_REGISTRY
def signal_strategy_names() -> tuple[str, ...]: ...   # target_kind == "signals"
def weight_strategy_names() -> tuple[str, ...]: ...   # target_kind == "weights"
def all_strategy_names() -> tuple[str, ...]: ...
def signal_registry() -> dict[str, type]: ...         # filtered view (for back-compat)
def build_strategy(name: str, cfg: LabConfig): ...    # reg[name](cfg), ValueError if unknown
```
`__init__.py` re-exports these helpers (so `from ggTrader.lab.strategies import build_strategy` works).

**Circular import:** `__init__.py` imports the strategy modules to build `STRATEGY_REGISTRY`, so those modules cannot read the registry at *their* import time. The shims below therefore resolve lazily.

**Backward-compat shims via module `__getattr__` (PEP 562)** — keeps all ~10 existing importers (cli.py, 2 scripts, ~6 test files) working with ZERO call-site edits, while making the old names derive from the single source:
- `signals.py`: delete the literal `_build_signal_registry`, `_get_registry`, `SIGNAL_STRATEGY_NAMES`, `build_signal_strategy`. Add a module `__getattr__(name)` returning, lazily: `SIGNAL_STRATEGY_NAMES` → `registry.signal_strategy_names()`; `build_signal_strategy` → `registry.build_strategy`; `_get_registry` → `registry.signal_registry` (so existing `_get_registry()` calls still return the signal-only dict).
- `momentum.py`: delete the literal `_REGISTRY`, `STRATEGY_NAMES`, `build_strategy`. Add a module `__getattr__(name)` returning `STRATEGY_NAMES` → `registry.weight_strategy_names()`; `build_strategy` → `registry.build_strategy`. (The unified `build_strategy` handles all kinds; momentum's old one only handled weights — superset behavior, existing call sites unaffected.)

**Instantiation is uniform:** every strategy constructor takes `cfg` first-positional with all else defaulted (verified across `EnsembleSignal`, `EnsembleICSignal`, `CrossSectionalMomentum`, `DualMomentum`, the 7 `signals.py` classes). `build_strategy` calls `cls(cfg)`.

### Sync test (new)
`tests/lab/test_registry.py`: for every `(name, cls)` in `STRATEGY_REGISTRY` assert `cls.name == name` and `cls.target_kind in {"signals", "weights"}`; assert `set(signal_strategy_names()) | set(weight_strategy_names()) == set(STRATEGY_REGISTRY)` and the two are disjoint; assert `build_strategy(name, cfg)` returns an object whose `.name == name` for every name.

## Part B — `--blend` Path

### CLI
New mode, mutually exclusive with `--sweep`/`--wfo`:
```
ggt lab --blend "ensemble@sp500,xs_momentum@nasdaq100" \
        --target-vol 0.068 --blend-window 60 --max-leverage 2.0
```
- `--blend` value: comma-separated `strategy@universe` sleeves. `strategy` must be in `all_strategy_names()`; `universe` in `UNIVERSE_CHOICES`.
- `--target-vol` (default 0.068), `--blend-window` (default 60), `--max-leverage` (default 2.0) — pass-through to `combine_sleeves`.
- Parser: add `--blend` to the existing mutually-exclusive `mode` group (`cli.py:55`).

### New module `src/ggTrader/lab/blend.py`
`run_blend(sleeves, cfg, eval_start, eval_end, market, base_config, *, target_vol, window, max_leverage) -> BlendResult` where `sleeves: list[tuple[str, str]]` (strategy_name, universe).

Logic (the concrete validated assembly, ported from `scripts/multi_sleeve_research.py:50-89`):
1. **One shared OHLCV load** across the union of all sleeves' universe members + `SPY`:
   `members[u] = equity_universe_between(eval_start, eval_end, universe=u)` for each distinct universe `u`; `all_symbols = sorted(⋃ members ∪ {"SPY"})`; `ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)` (with the same `warmup_days` prefix the CLI/script computes). `available = set(ohlcv.columns.get_level_values(0))`.
2. For each `(strategy, universe)` sleeve, with `label = f"{strategy}@{universe}"`:
   - `syms = [x for x in members[universe] if x in available]`.
   - `result = run_wfo(strategy, STRATEGY_REGISTRY[strategy], cfg, ohlcv[syms], ohlcv["SPY"]["close"].dropna(), eval_start, eval_end, market, base_config, grid=build_grid(STRATEGY_REGISTRY[strategy]))`.
   - `curves[label] = result.oos_equity`.
3. **Align on the intersection of all sleeve curve dates**, then daily returns:
   `common = reduce(intersection, [c.index for c in curves.values()])`; `returns_df = pd.DataFrame({label: curves[label].reindex(common).pct_change() for label in curves}).dropna()`.
4. `blended, diag = combine_sleeves(returns_df, target_vol=..., window=..., max_leverage=...)`; `blend_eq = (1+blended).cumprod() * START_CASH`.
5. `curve_stats(blend_eq)` + each `curve_stats(curves[label].reindex(common))` + SPY benchmark → Markdown table (blend vs each sleeve vs SPY), plus the gated-sleeve correlation matrix (`returns_df.corr()`) and realized leverage (`diag["scale"]` avg/max), as the script prints.
6. **Persist as a normal lab run:** `persist.init_schema()`; `run_id = persist.start_run(name=f"blend:{','.join(labels)}", market, freq, eval_start, eval_end, params={"sleeves": labels, "target_vol": ..., "window": ..., "max_leverage": ...})`; `persist.write_returns_equity` for `blend_eq` (name `"blend"`) AND each sleeve curve (name = label), each vs the SPY bench curve; `persist.write_summary(run_id, "blend", blend_stats, spy_stats, {**diag-summary})`; `persist.finish_run(run_id)`.

`BlendResult = NamedTuple(blended_equity, sleeve_equity: dict[str, Series], diag: DataFrame, table: str, run_id: str)`.

### CLI dispatch
In `cli.py` `main`, add the `--blend` branch BEFORE the current single-universe OHLCV load (blend loads per-sleeve data itself), so the single-universe load is guarded out (`if not args.blend:` around it, or move it into the non-blend branches). The branch parses sleeves, calls `run_blend`, prints the table, returns the `run_id`.

### Retire scripts
Delete `scripts/multi_sleeve_research.py` and `scripts/portfolio_blend.py` — superseded; their logic now lives in tested `blend.py`. (Confirm no other caller: only their own invocations reference them.)

## Leak Safety

Inherited, not re-derived: `combine_sleeves` is OOS-correct (weights/scale at each rebalance use only returns strictly before that date — already tested) and `run_wfo` produces honest OOS curves (already tested). `blend.py` only assembles their outputs; the blend test asserts assembly correctness, not leak-freedom of the primitives.

## Testing

- **Registry** (`tests/lab/test_registry.py`): sync test as specified in Part A.
- **blend** (`tests/lab/test_blend.py`): monkeypatch `run_wfo` to return synthetic `WfoResult` with known `oos_equity` per sleeve; assert (a) sleeve returns assembled/aligned correctly, (b) a 2-sleeve blend of two equal-vol streams ≈ 50/50 weight, (c) `persist` calls fire (mock/patch persist: one `start_run`, `write_returns_equity` per sleeve + blend, one `write_summary`, one `finish_run`), (d) `BlendResult` fields populated.
- **CLI** (`tests/lab/test_cli.py` extend): `--blend "a@sp500,b@nasdaq100"` parses to `[("a","sp500"),("b","nasdaq100")]`; `--blend` with `--wfo` is rejected (mutually exclusive); an unknown strategy or universe in a sleeve raises a clear error.

## Files Touched

| File | Change |
|---|---|
| `src/ggTrader/lab/strategies/registry.py` | new — lazy derivation helpers (`signal_strategy_names`, `weight_strategy_names`, `all_strategy_names`, `signal_registry`, `build_strategy`) |
| `src/ggTrader/lab/strategies/__init__.py` | `STRATEGY_REGISTRY` = sole source; re-export the registry helpers |
| `src/ggTrader/lab/strategies/signals.py` | delete `_build_signal_registry`/`_get_registry`/`SIGNAL_STRATEGY_NAMES`/`build_signal_strategy` literals; add module `__getattr__` lazy shims deriving from `registry` |
| `src/ggTrader/lab/strategies/momentum.py` | delete `_REGISTRY`/`STRATEGY_NAMES`/`build_strategy` literals; add module `__getattr__` lazy shims deriving from `registry` |
| `src/ggTrader/lab/blend.py` | new — `run_blend` + `BlendResult` |
| `src/ggTrader/lab/cli.py` | `--blend`/`--target-vol`/`--blend-window`/`--max-leverage` args; blend dispatch branch; guard single-universe load |
| `scripts/multi_sleeve_research.py`, `scripts/portfolio_blend.py` | delete |
| `tests/lab/test_registry.py` | new |
| `tests/lab/test_blend.py` | new |
| `tests/lab/test_cli.py` | extend with `--blend` parse/validation cases |
