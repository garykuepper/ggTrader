"""Check the membership join on invented snapshots and flags."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "seam_membership.py"  # root-ok: sibling
spec = importlib.util.spec_from_file_location("seam_membership", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_uses_latest_prior_snapshot_and_keeps_unknown_and_nonmembers():
    history = pd.DataFrame(
        {"tickers": [{"A"}, {"B"}]}, index=pd.to_datetime(["2024-01-03", "2024-01-05"], utc=True)
    )
    flags = [
        {"symbol": "A", "date": "2024-01-02"},
        {"symbol": "A", "date": "2024-01-04"},
        {"symbol": "A", "date": "2024-01-05"},
        {"symbol": "B", "date": "2024-01-06"},
    ]
    rows = module.assess(flags, history, "2024-01-02", "2024-01-05")
    assert [r["membership"] for r in rows] == ["unknown", "member", "no_exact_match"]
    assert rows[1]["membership_snapshot"] == "2024-01-03"
    assert rows[2]["membership_snapshot"] == "2024-01-05"
    assert rows[1]["snapshot_age_days"] == 1


def test_normalizes_flag_tickers_and_rejects_normalized_duplicates():
    history = pd.DataFrame({"tickers": [{"BRK.B"}]}, index=pd.to_datetime(["2024-01-03"], utc=True))
    row = {"symbol": "brk.b", "date": "2024-01-04"}
    assert module.assess([row], history, "2024-01-01", "2024-01-10")[0]["membership"] == "member"
    with pytest.raises(ValueError, match="duplicate"):
        module.assess(
            [row, {"symbol": "BRK-B", "date": "2024-01-04"}], history, "2024-01-01", "2024-01-10"
        )


def test_rejects_unordered_membership_snapshots():
    history = pd.DataFrame(
        {"tickers": [{"A"}, {"A"}]}, index=pd.to_datetime(["2024-01-05", "2024-01-03"], utc=True)
    )
    with pytest.raises(ValueError, match="membership"):
        module.assess([], history, "2024-01-01", "2024-01-10")


def test_rename_is_not_misrepresented_as_nonmembership():
    history = pd.DataFrame({"tickers": [{"ABC"}]}, index=pd.to_datetime(["2021-01-01"], utc=True))
    result = module.assess(
        [{"symbol": "COR", "date": "2021-03-30"}], history, "2021-01-01", "2021-12-31"
    )
    assert result[0]["membership"] == "no_exact_match"


def test_rejects_empty_membership_and_reverse_dates():
    history = pd.DataFrame({"tickers": [set()]}, index=pd.to_datetime(["2024-01-03"], utc=True))
    with pytest.raises(ValueError, match="empty"):
        module.assess([{"symbol": "A", "date": "2024-01-04"}], history, "2024-01-01", "2024-01-10")
    with pytest.raises(ValueError, match="end precedes"):
        module.assess([], history, "2024-01-10", "2024-01-01")
