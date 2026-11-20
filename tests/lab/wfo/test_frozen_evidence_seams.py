"""The names and patch seams that frozen research evidence still uses stay in place.

Not golden-driven, so it stays in the fast core. Only the evidence files whose ``ggTrader``
imports all resolve today are covered (measured 2026-11-18); the files that already fail to
import stay broken whatever the refactor does and are not asserted:

* ``p10-evidence/dsr_one_combo.py``: ``run.run_wfo``, ``run.run_dsr`` replaced on the module,
  ``gates.run_dsr`` and ``gates.MIN_DSR_BARS``, and the fold keys ``gates_passed``, ``dsr_value``.
* ``p8-evidence/probes/golden_dump.py``: ``engine.config.set_engine`` and
  ``tests.engine.scenarios.run_all``.
* ``D-examples/d1_wfo_splitter.py`` and ``d5_full_history_wfo.py``: ``lab.wfo.generate_folds``.
"""

from __future__ import annotations

import contextlib
import inspect
import io
from typing import Any

import pandas as pd
import pytest

from ggTrader.engine import config as engine_config
from ggTrader.lab import wfo
from ggTrader.lab.wfo import gates, run
from tests.lab.wfo import helpers as h
from tests.lab.wfo.test_run_wfo_characterization import INSIDE_TRAIN, _toy_run


def test_frozen_evidence_seams_still_resolve() -> None:
    # p10-evidence/dsr_one_combo.py
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategies.ensemble import Ensemble
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo import cli_support
    from tests.oracles.legacy_dsr import dsr_check

    assert callable(STOCK_BASE_CONFIG.keys) and callable(Ensemble.live_grid)
    assert callable(LabConfig) and callable(dsr_check)
    assert callable(cli_support.load_run_bars) and callable(cli_support.sleeve_inputs)
    assert callable(gates.run_dsr) and isinstance(gates.MIN_DSR_BARS, int)
    assert callable(run.run_dsr) and callable(run.run_wfo)
    keywords = set(inspect.signature(run.run_wfo).parameters)
    assert {
        "eval_start",
        "eval_end",
        "market",
        "base_config",
        "grid",
        "mask",
        "verbose",
    } <= keywords
    # p8-evidence/probes/golden_dump.py
    from tests.engine import scenarios

    assert callable(engine_config.set_engine) and callable(scenarios.run_all)
    # D-examples/d1_wfo_splitter.py and d5_full_history_wfo.py
    folds = wfo.generate_folds(
        pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")
    )
    assert len(folds) > 1 and folds[0].train_start == pd.Timestamp("2021-01-31", tz="UTC")


def test_replacing_run_dsr_on_the_module_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dsr_one_combo.py`` assigns ``run.run_dsr``; ``run_wfo`` must call that replacement."""
    seen: list[tuple[int, float]] = []
    real = gates.run_dsr

    def spy(returns: pd.Series, n_trials: int, threshold: float) -> Any:
        seen.append((n_trials, threshold))
        return real(returns, n_trials, threshold)

    monkeypatch.setattr(run, "run_dsr", spy)
    record = _toy_run(INSIDE_TRAIN)
    assert len(seen) == len(record["fold_results"]) == 3
    # the evidence reads these two fold keys
    for row in record["fold_results"]:
        assert isinstance(row["gates_passed"], bool) and isinstance(row["dsr_value"], float)


def test_the_toy_helpers_the_pins_use_exist() -> None:
    assert callable(h.toy_bars) and h.ToyHold.kind == "signals"
    with contextlib.redirect_stdout(io.StringIO()):
        assert h.toy_bars(n_bars=3).close.shape == (3, 1)
