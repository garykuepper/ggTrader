"""L0/L3: engine settings are explicit, validated and applied at import."""

from __future__ import annotations

import warnings

import pytest

from ggTrader.engine import config
from ggTrader.engine.config import EngineSettings


def test_default_is_rust_with_252_day_year() -> None:
    s = EngineSettings.from_env({})
    assert s.engine == "rust"
    assert s.year_freq == "252 days"
    assert s.freq == "1D"


@pytest.mark.parametrize("value", ["rust", "numba"])
def test_env_selects_engine(value: str) -> None:
    assert EngineSettings.from_env({"GGT_VBT_ENGINE": value}).engine == value


@pytest.mark.parametrize("value", ["auto", "AUTO", "cuda", "rustt"])
def test_auto_and_unknown_engines_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="GGT_VBT_ENGINE"):
        EngineSettings.from_env({"GGT_VBT_ENGINE": value})


def test_blank_env_means_default() -> None:
    assert EngineSettings.from_env({"GGT_VBT_ENGINE": "  "}).engine == "rust"


def test_settings_are_frozen() -> None:
    with pytest.raises(Exception):
        EngineSettings().engine = "numba"  # type: ignore[misc]


def test_import_applied_the_settings() -> None:
    import vectorbt as vbt

    assert vbt.settings["engine"] in ("rust", "numba")  # never "auto"
    assert vbt.settings["returns"]["year_freq"] == "252 days"
    assert vbt.settings["array_wrapper"]["freq"] == "1D"


def test_returns_accessor_uses_the_252_day_year() -> None:
    import pandas as pd
    import vectorbt as vbt  # noqa: F401  (registers .vbt)

    rets = pd.Series([0.01, -0.02, 0.015], index=pd.bdate_range("2024-01-01", periods=3))
    assert rets.vbt.returns.ann_factor == 252


def test_configure_returns_the_applied_settings(use_engine) -> None:
    applied = config.configure(EngineSettings(engine="numba"))
    assert applied.engine == "numba"
    assert config.active_engine() == "numba"


def test_mismatch_warning_becomes_an_error() -> None:
    import vectorbt_rust
    from vectorbt import _engine

    original = vectorbt_rust.__version__
    with warnings.catch_warnings():
        config.configure()
        vectorbt_rust.__version__ = "0.0.0"
        _engine.clear_engine_cache()
        try:
            with pytest.raises(UserWarning, match="version mismatch"):
                _engine.is_rust_available()
        finally:
            vectorbt_rust.__version__ = original
            _engine.clear_engine_cache()
    assert _engine.is_rust_available()


def test_every_setting_is_documented_by_spec_reference() -> None:
    for name in ("GGT_VBT_ENGINE", "year_freq", "freq", "contract"):
        assert name in config.__doc__


def test_mismatch_warning_is_fatal_on_the_simulator_path_too() -> None:
    """Second, independent route: a simulation that asks for Rust on a mismatched wheel
    raises instead of silently running numba (the cost spec 5.5 exists to prevent)."""
    import numpy as np
    import pandas as pd
    import vectorbt_rust
    from vectorbt import _engine

    from ggTrader.engine import portfolio

    close = pd.DataFrame({"a": np.linspace(10.0, 12.0, 60)})
    entries = pd.DataFrame({"a": [i == 5 for i in range(60)]})
    cfg = {"START_CASH": 1000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    original = vectorbt_rust.__version__
    with warnings.catch_warnings():
        config.configure()
        config.set_engine("rust")
        vectorbt_rust.__version__ = "0.0.0"
        _engine.clear_engine_cache()
        try:
            with pytest.raises(UserWarning, match="version mismatch"):
                portfolio.run_signals(entries, ~entries, close, cfg)
        finally:
            vectorbt_rust.__version__ = original
            _engine.clear_engine_cache()


def test_importing_the_engine_installs_the_mismatch_filter() -> None:
    """A fresh interpreter, no configure() call by the test: the import alone must do it."""
    import subprocess
    import sys

    code = (
        "import warnings, ggTrader.engine\n"
        "hits = [f for f in warnings.filters if f[0] == 'error' and f[1] is not None "
        "and 'version mismatch' in f[1].pattern and f[2] is UserWarning]\n"
        "raise SystemExit(0 if hits else 3)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
