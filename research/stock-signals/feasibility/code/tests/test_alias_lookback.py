"""Check consecutive-session coverage for candidate ticker links."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "alias_lookback.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("alias_lookback", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_requires_every_session_and_does_not_count_post_change():
    sessions = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"])
    rows = [
        {"date": "2021-01-06", "missing": ["OLD"]},
        {"date": "2021-01-07", "missing": ["OLD"]},
    ]
    bars = pd.DataFrame({"symbol": ["NEW", "NEW"], "timestamp": sessions[[0, 2]]})
    result = module.count_lookback(rows, bars, sessions, {"OLD": ("NEW", "2021-01-07")}, 3)
    assert result["OLD"] == {"eligible": 1, "same_day": 1, "complete_lookback": 0}
    bars = pd.concat([bars, pd.DataFrame({"symbol": ["NEW"], "timestamp": [sessions[1]]})])
    assert (
        module.count_lookback(rows, bars, sessions, {"OLD": ("NEW", "2021-01-07")}, 3)["OLD"][
            "complete_lookback"
        ]
        == 1
    )


def test_rejects_decision_dates_with_insufficient_calendar_history():
    sessions = pd.to_datetime(["2021-01-04", "2021-01-05"])
    bars = pd.DataFrame({"symbol": ["NEW"], "timestamp": [sessions[0]]})
    with pytest.raises(ValueError, match="calendar"):
        module.count_lookback(
            [{"date": "2021-01-04", "missing": ["OLD"]}],
            bars,
            sessions,
            {"OLD": ("NEW", "2022-01-01")},
            2,
        )
