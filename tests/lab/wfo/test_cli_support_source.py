"""Source selection reaches the bars loader."""

import pandas as pd
import pytest

from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import cli_support


@pytest.mark.parametrize(
    "snapshot,source", [(None, None), ("tape.parquet", None), (None, "snapshot")]
)
def test_load_run_bars_source_forwarded(monkeypatch, snapshot, source):
    monkeypatch.setattr(cli_support, "equity_universe_between", lambda *a, **k: ["AAA"])
    seen = {}
    monkeypatch.setattr(cli_support, "load_bars", lambda *a, **k: seen.update(k))
    cli_support.load_run_bars(
        ["sp500"],
        LabConfig(),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        snapshot,
        source,
    )
    assert seen["source"] == source
    assert seen["snapshot_path"] == snapshot
