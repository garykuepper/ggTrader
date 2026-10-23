"""Checks for the frozen century leverage inputs and mechanics."""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from ggTrader.lab.fred_data import parse_fred_csv
from scripts import static_leverage_aftertax as a23


def test_a23_frozen_loader_matches_live_columns_and_values(monkeypatch) -> None:
    """Frozen A23 files must follow the same code path as an identical live fetch."""
    source = Path("data/external/a23_inputs")

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs: object) -> pd.DataFrame:
            frame = pd.read_csv(source / f"{self.symbol}.csv", index_col="date", parse_dates=True)
            frame.index = frame.index.tz_localize("America/New_York")
            return frame

    monkeypatch.setattr(a23.yf, "Ticker", Ticker)
    monkeypatch.setattr(
        a23,
        "load_fred_series",
        lambda *_args: parse_fred_csv((source / "DTB3.csv").read_text(), "DTB3"),
    )
    live = a23.load_prices(0.0075)
    frozen = a23.load_prices(0.0075, frozen_inputs=source)
    for actual, expected in zip(frozen[:2], live[:2], strict=True):
        assert_frame_equal(actual, expected)
    assert frozen[2] == live[2]
