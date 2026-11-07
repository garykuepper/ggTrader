"""The worked example in docs/adding_a_strategy.md runs on the real API and passes the checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ggTrader.lab.strategies.contract import simulate_signal_out
from ggTrader.lab.strategies.leak import signal_leak_check
from ggTrader.lab.strategies.registry import contract_problems
from tests.lab.strategies.helpers import make_bars

EXAMPLE = Path(__file__).resolve().parents[3] / "docs" / "examples" / "sma_cross.py"
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def sma_cross():
    spec = importlib.util.spec_from_file_location("docs_example_sma_cross", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmaCross


def test_the_example_implements_the_contract(sma_cross) -> None:
    assert contract_problems(sma_cross) == []
    assert len(sma_cross.grid) == 4  # (10,50) (10,100) (20,50) (20,100); fast < slow always


def test_the_example_builds_one_column_block_per_combination(sma_cross) -> None:
    bars = make_bars(seed=5, n_bars=300, n_syms=4)
    out = sma_cross().build(bars)
    assert out.entries.shape == (300, 4 * 4) and out.exits.shape == out.entries.shape
    assert out.entries.columns.names == ["fast", "slow", "symbol"]
    assert out.entries.to_numpy().any()  # it trades


def test_the_example_has_no_lookahead(sma_cross) -> None:
    bars = make_bars(seed=5, n_bars=300, n_syms=4)
    dates = [bars.close.index[p] for p in (80, 150, 220)]
    assert signal_leak_check(sma_cross(), bars, sma_cross.grid, dates) == []


def test_the_example_simulates_on_the_engine(sma_cross) -> None:
    bars = make_bars(seed=5, n_bars=300, n_syms=4)
    out = sma_cross().build(bars)
    base = {"START_CASH": 100_000.0, "FEES": 0.0005, "SLIPPAGE": 0.0005, "FREQ": "1D"}
    result = simulate_signal_out(out, bars.close, base | {"SIGNAL_POSITION_SIZE": 0.1})
    assert np.isfinite(np.asarray(result.value.to_numpy())).all()
