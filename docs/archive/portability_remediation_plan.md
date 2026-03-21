## Portability remediation plan (portability-first, minimal behavior change)

This document proposes a staged set of changes to make `ggTrader` run reliably on
case-sensitive filesystems (Linux/macOS/CI) and align with your Python standards
(snake_case filenames/modules).

### Stage A — Confirm and inventory (no behavior changes)

- **Lock the canonical module names** to snake_case:
  - `ggTrader.indicators.signals`
  - `ggTrader.core.portfolio`, `ggTrader.core.position`, `ggTrader.core.trading`,
    `ggTrader.core.screener`
- **Confirm all import sites** that reference the “wrong” casing/module names. Known sites:
  - Indicators:
    - `src/ggTrader/core/fast_backtest.py` → `ggTrader.indicators.signals`
    - `src/ggTrader/indicators/__init__.py` → `ggTrader.indicators.signals`
    - `src/ggTrader/core/Trading.py` → `ggTrader.indicators.signals`
    - `src/ggTrader/core/execution_engine.py` → `ggTrader.indicators.Signals` (mixed)
    - tests: `tests/test_signals.py`, `tests/test_broadcasting.py`, `tests/test_trading.py`
  - Core:
    - `src/ggTrader/core/__init__.py` imports `.portfolio/.position/.trading/.screener`
    - `src/ggTrader/core/Trading.py`, `src/ggTrader/core/Portfolio.py` use `ggTrader.core.portfolio`
      etc
    - tests: `tests/test_*` import `ggTrader.core.portfolio/position/trading/screener`

### Stage B — Rename files to snake_case and update imports

#### B1) Rename files (source of truth becomes snake_case)

- **Indicators**
  - Rename `src/ggTrader/indicators/Signals.py` → `src/ggTrader/indicators/signals.py`

- **Core**
  - Rename `src/ggTrader/core/Trading.py` → `src/ggTrader/core/trading.py`
  - Rename `src/ggTrader/core/Portfolio.py` → `src/ggTrader/core/portfolio.py`
  - Rename `src/ggTrader/core/Position.py` → `src/ggTrader/core/position.py`
  - Rename `src/ggTrader/core/Screener.py` → `src/ggTrader/core/screener.py`

These renames alone fix the “Windows-only” import behavior and match your stated standards.

#### B2) Update imports (make them consistent everywhere)

- In **core package exports**: update `src/ggTrader/core/__init__.py` if needed so it imports from
  the renamed modules (same module names as today, but now the files actually exist).
- In **engine modules**:
  - `src/ggTrader/core/fast_backtest.py`: ensure it imports `SignalFactory` from
    `ggTrader.indicators.signals`
  - `src/ggTrader/core/execution_engine.py`: replace `from ggTrader.indicators.Signals import ...`
    with `from ggTrader.indicators.signals import ...`
  - `src/ggTrader/core/trading.py` (renamed): update any intra-core imports to snake_case modules
- In **indicators package export**: `src/ggTrader/indicators/__init__.py` should import from
  `ggTrader.indicators.signals` (lowercase file).
- In **tests**: update any references to `ggTrader.core.trading` etc only if they break after
  renames (they should remain the same imports, but now resolve cross-platform).
- In **notebooks**:
  - `notebooks/ohlcv_signal_processing.ipynb` uses `ggTrader.indicators.signals`, which will become
    correct once the file is renamed.

#### B3) Add a temporary compatibility shim (optional)

If you want a softer transition for any downstream imports (or old notebooks):

- Keep a small stub module at the old path that re-exports the new symbols.
  - Example: keep `Signals.py` with `from .signals import *` (but this violates your snake_case
    filename standard, so only do it temporarily, and exclude it from packaging/usage).

Given your portability-first priority and standards, the cleaner approach is to **do the renames**
and fix imports, without keeping legacy-case files.

### Stage C — Harden `get_latest_params()` path resolution

In `src/ggTrader/utils/results_manager.py`:

- Make `get_latest_params()` anchor to the same project root as `ResultsManager` by using
  `find_project_root() / "results"` rather than `Path("results").absolute()`.

This avoids “works depending on current working directory” behavior in shells/CI.

### Notes / non-portability follow-ups (not part of this plan)

- `INTERVAL` vs `FREQ` metadata mismatch can be tightened by defaulting `FREQ` to `INTERVAL` when
  unspecified, but that’s a behavioral/semantics decision and not required for portability.
- Docs drift (CLI flags, expected output files) can be updated after portability is solved.

