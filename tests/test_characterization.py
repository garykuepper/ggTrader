"""Tests for canonical characterization records."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from tests.characterization import RTOL, check_golden, compare, encode


def test_encode_preserves_dict_order():
    assert encode({"a": 1, "b": 2}) != encode({"b": 2, "a": 1})


def test_encode_tags_numpy_and_pandas_types():
    assert encode(np.float64(1.5)) != encode(1.5)
    assert encode(np.bool_(True)) != encode(True)
    assert encode(pd.Timestamp("2020-01-01T12:00:00+05:30"))["v"].endswith("+05:30")
    assert encode(pd.NaT) == {"$type": "pandas.NaT"}


def test_encode_tags_non_finite_floats():
    assert [encode(x) for x in (float("nan"), float("inf"), -float("inf"))] == [
        {"$float": "NaN"},
        {"$float": "Infinity"},
        {"$float": "-Infinity"},
    ]
    assert encode(np.float64("nan"))["v"] == {"$float": "NaN"}


def test_encode_rejects_unknown_types():
    for value in (object(), OrderedDict()):
        with pytest.raises(TypeError, match=type(value).__name__):
            encode(value)


def test_compare_is_exact_on_structure_and_tolerant_on_floats():
    one_ulp = np.nextafter(1.0, 2.0).item()
    assert not compare(1.0, one_ulp, rtol=RTOL, exact=False)
    assert compare(1.0, one_ulp, rtol=RTOL, exact=True)
    for exact in (False, True):
        assert compare(1, 1.0, rtol=RTOL, exact=exact)


def test_compare_reports_a_path():
    assert "a[1]" in compare({"a": [1, 2]}, {"a": [1, 3]}, rtol=RTOL, exact=False)


def test_check_golden_regenerate_then_pass(tmp_path, monkeypatch):
    path = tmp_path / "golden.json"
    with pytest.raises(pytest.fail.Exception, match="GGT_REGEN_CHARACTERIZATION=1"):
        check_golden(path, {"a": [1, 2]})
    monkeypatch.setenv("GGT_REGEN_CHARACTERIZATION", "1")
    with pytest.raises(pytest.fail.Exception, match="regenerated"):
        check_golden(path, {"a": [1, 2]})
    monkeypatch.delenv("GGT_REGEN_CHARACTERIZATION")
    check_golden(path, {"a": [1, 2]})
    with pytest.raises(pytest.fail.Exception, match=r"\$dict"):
        check_golden(path, {"a": [1, 3]})


def test_frame_and_series_round_trip_dtypes():
    frame = pd.DataFrame({"f": [1.5], "i": [1]})
    assert encode(frame)["$frame"]["dtypes"] == ["float64", "int64"]
    assert encode(frame["i"])["$series"]["dtype"] == "int64"
    assert encode(np.array([[1, 2]]))["$array"] == {"dtype": "int64", "values": [[1, 2]]}
