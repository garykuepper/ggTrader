## Portability verification checklist (post-fix)

Run these checks after applying the portability renames/import fixes.

### 1) Import smoke tests (repo root)

Run from the repository root (where `pyproject.toml` lives):

```bash
python -c "import sys; sys.path.append('src'); import ggTrader"
python -c "import sys; sys.path.append('src'); from ggTrader.core.orchestrator import run_backtest_orchestrator"
python -c "import sys; sys.path.append('src'); from ggTrader.core.fast_backtest import FastBacktest"
python -c "import sys; sys.path.append('src'); from ggTrader.indicators.signals import Signals, SignalFactory"
```

If you use editable installs, also verify:

```bash
pip install -e .
python -c "import ggTrader; from ggTrader.core.orchestrator import run_backtest_orchestrator"
```

### 2) Script entrypoints (imports + argparse)

These should all return help text without import errors:

```bash
python scripts/run_backtest.py --help
python scripts/run_sensitivity_analysis.py --help
python scripts/run_walk_forward_optimization.py --help
```

### 3) Tests (fast “import correctness” sweep)

At minimum, run tests that previously depended on lowercase imports:

```bash
pytest -q tests/test_signals.py
pytest -q tests/test_broadcasting.py
pytest -q tests/test_orchestrator.py
pytest -q tests/test_trading.py
pytest -q tests/test_portfolio_position.py
pytest -q tests/test_screener.py
```

### 4) Notebook sanity (if you use notebooks)

- Open `notebooks/single_backtest_runner.ipynb` and run the first cells through the orchestrator
  call. Confirm the imports resolve without any `sys.path` hacks other than the intended one.
- Open `notebooks/ohlcv_signal_processing.ipynb` and confirm `from ggTrader.indicators.signals ...`
  resolves.

### 5) CI / Linux check (recommended)

On a Linux runner (or WSL), rerun sections 1–3. This is where casing issues show up immediately.

