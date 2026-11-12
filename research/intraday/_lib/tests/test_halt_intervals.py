"""Synthetic checks of conservative halt-to-session mapping."""

import halt_intervals
import pandas as pd
import pytest

from ggTrader.engine.portfolio import run_signals

pytestmark = pytest.mark.study


@pytest.fixture
def sessions():
    return pd.DataFrame(
        {
            "open": [
                "2023-03-10T09:30:00-05:00",
                "2023-03-13T09:30:00-04:00",
                "2023-03-14T09:30:00-04:00",
            ],
            "close": [
                "2023-03-10T16:00:00-05:00",
                "2023-03-13T13:00:00-04:00",
                "2023-03-14T16:00:00-04:00",
            ],
        },
        index=pd.to_datetime(["2023-03-10", "2023-03-13", "2023-03-14"], utc=True),
    )


def test_multiday_halt_carries_through_weekend_and_resumes_at_open(sessions):
    events = [
        {"symbol": "A", "start": "2023-03-10T08:35:00-05:00", "end": "2023-03-14T09:30:00-04:00"}
    ]
    mask = halt_intervals.daily_halt_mask(sessions, ["A", "B"], events)
    assert mask.index.equals(sessions.index)
    assert mask["A"].tolist() == [True, True, False]
    assert mask["B"].tolist() == [False, False, False]


def test_session_close_is_inclusive_and_unknown_resumption_stays_flagged(sessions):
    events = [{"symbol": "A", "start": "2023-03-13T13:00:00-04:00", "end": None}]
    mask = halt_intervals.daily_halt_mask(sessions, ["A"], events)
    assert mask["A"].tolist() == [False, True, True]


def test_halt_fully_outside_regular_sessions_does_not_flag(sessions):
    events = [
        {"symbol": "A", "start": "2023-03-13T13:01:00-04:00", "end": "2023-03-14T09:29:00-04:00"}
    ]
    assert not halt_intervals.daily_halt_mask(sessions, ["A"], events).to_numpy().any()


def test_multiple_events_accumulate_instead_of_overwriting(sessions):
    events = [
        {"symbol": "A", "start": "2023-03-10T10:00:00-05:00", "end": "2023-03-10T11:00:00-05:00"},
        {"symbol": "A", "start": "2023-03-14T10:00:00-04:00", "end": "2023-03-14T11:00:00-04:00"},
    ]
    assert halt_intervals.daily_halt_mask(sessions, ["A"], events)["A"].tolist() == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "fault", ["naive_open", "reversed", "overlap", "duplicate_index", "wrong_label", "non_midnight"]
)
def test_bad_session_calendar_is_rejected(sessions, fault):
    if fault == "naive_open":
        sessions.iloc[0, sessions.columns.get_loc("open")] = "2023-03-10 09:30"
    elif fault == "reversed":
        sessions.iloc[0, sessions.columns.get_loc("close")] = "2023-03-10T08:00:00-05:00"
    elif fault == "overlap":
        sessions.iloc[1, sessions.columns.get_loc("open")] = "2023-03-10T15:00:00-05:00"
    elif fault == "duplicate_index":
        sessions.index = pd.DatetimeIndex([sessions.index[0], sessions.index[0], sessions.index[2]])
    elif fault == "wrong_label":
        sessions.index = sessions.index + pd.Timedelta(days=1)
    else:
        sessions.index = sessions.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        halt_intervals.daily_halt_mask(sessions, ["A"], [])


@pytest.mark.parametrize(
    "events",
    [
        [{"symbol": "B", "start": "2023-03-10T08:00:00Z", "end": None}],
        [{"symbol": "A", "start": "2023-03-10 08:00", "end": None}],
        [{"symbol": "A", "start": "2023-03-10T08:00:00Z", "end": "2023-03-10T07:00:00Z"}],
        [{"symbol": "A", "start": "2023-03-10T08:00:00Z"}],
    ],
)
def test_bad_halt_event_is_rejected(sessions, events):
    with pytest.raises(ValueError):
        halt_intervals.daily_halt_mask(sessions, ["A"], events)


def test_generated_mask_refuses_hypothetical_held_position(sessions):
    events = [{"symbol": "A", "start": "2023-03-13T10:00:00-04:00", "end": None}]
    mask = halt_intervals.daily_halt_mask(sessions, ["A"], events)
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=sessions.index)
    entries = pd.DataFrame({"A": [True, False, False]}, index=sessions.index)
    exits = pd.DataFrame({"A": [False, True, False]}, index=sessions.index)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="halt during an open position"):
        run_signals(entries, exits, close, config, halt_mask=mask)
