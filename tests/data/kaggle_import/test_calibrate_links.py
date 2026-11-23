"""Synthetic tests of the calibration decisions."""

import pandas as pd

from scripts.kaggle_calibrate_links import choose, score, split_truth, truth_from_rows


def test_truth_excludes_reused_ticker():
    rows = pd.DataFrame(
        {
            "date": [20200101, 20210101, 20220101],
            "permno": [1, 2, 1],
            "stock_ticker": ["A", "A", "B"],
        }
    )
    truth, reverse, tickers = truth_from_rows(rows)
    assert truth == {"B": 1}
    assert reverse[1] == ["A", "B"]
    assert tickers["A"] == [1, 2]


def test_split():
    symbols = [str(i) for i in range(20)]
    a, b = split_truth(symbols)
    assert (a, b) == split_truth(symbols[::-1])
    assert len(a) == 16 and len(b) == 4 and not set(a) & set(b)


def test_score():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "permno": [1, 3, None],
            "status": ["linked", "linked", "ambiguous"],
        }
    )
    assert score(frame, {"A": 1, "B": 2, "C": 3}, ["A", "B", "C"]) == {
        "correct": 1,
        "false": 1,
        "unlinked": 0,
        "ambiguous": 1,
        "duplicate": 0,
    }


def test_choose_calibration_only():
    scores = [
        dict(false=1, correct=10, agreement_bar=0.999, margin=0.01),
        dict(false=0, correct=2, agreement_bar=0.99, margin=0.001),
        dict(false=0, correct=3, agreement_bar=0.995, margin=0.001),
        dict(false=0, correct=3, agreement_bar=0.999, margin=0.001),
        dict(false=0, correct=3, agreement_bar=0.999, margin=0.01),
    ]
    assert choose(scores) == scores[-1]
    assert choose(scores[:1]) == scores[0]
