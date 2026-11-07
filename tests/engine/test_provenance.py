"""L0/L1: provenance stamps every result and compare() refuses to mix epochs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import config, provenance
from ggTrader.utils import results

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prov() -> dict:
    return provenance.provenance()


def test_schema_fields_present(prov: dict) -> None:
    expected = {
        "schema",
        "python",
        "pandas",
        "numpy",
        "numba",
        "llvmlite",
        "scipy",
        "vectorbt",
        "vectorbt_rust",
        "engine_requested",
        "engine_effective",
        "rust_calls",
        "lock_sha256",
        "git_sha",
        "git_dirty",
        "contract_version",
        "baseline_epoch",
        "definitions",
        "created_utc",
    }
    assert expected <= set(prov)
    assert prov["schema"] == 1


def test_versions_come_from_the_installed_packages(prov: dict) -> None:
    import platform

    import numba
    import vectorbt as vbt

    assert prov["vectorbt"] == vbt.__version__
    assert prov["pandas"] == pd.__version__
    assert prov["numpy"] == np.__version__
    assert prov["numba"] == numba.__version__
    assert prov["python"] == platform.python_version()
    assert prov["vectorbt_rust"] == prov["vectorbt"]


def test_engine_effective_comes_from_the_probe_not_the_setting(monkeypatch) -> None:
    from ggTrader.engine import selfcheck

    config.set_engine("rust")
    monkeypatch.setattr(
        selfcheck,
        "_run_probe",
        lambda engine: {"engine": "numba", "rust_calls": 0, "final_value": 1.0},
    )
    with pytest.raises(selfcheck.EngineError):
        provenance.provenance()


@pytest.mark.parametrize("name", ["rust", "numba"])
def test_requested_and_effective_engine(name: str) -> None:
    config.set_engine(name)
    p = provenance.provenance()
    assert p["engine_requested"] == p["engine_effective"] == name
    assert (p["rust_calls"] > 0) is (name == "rust")


def test_lock_hash_is_the_sha256_of_uv_lock(prov: dict) -> None:
    lock = REPO / "uv.lock"
    assert prov["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()


def test_definitions_record_the_e1_conventions(prov: dict) -> None:
    d = prov["definitions"]
    assert d["year_freq"] == "252 days"
    assert d["bb_ddof"] == 1
    assert d["rsi"] == "wilder window=2n-1 ewm"
    assert d["macd"] == "ema macd_ewm=signal_ewm=True"
    assert d["atr"] == "sma ewm=False"
    assert d["dsr"] == "var_sharpe=252/(T-1) bias=False"
    assert prov["baseline_epoch"] == "E1"
    assert prov["contract_version"] == config.CONTRACT_VERSION


def test_definitions_are_stamped_per_statistic_with_the_legacy_set_for_e1(prov: dict) -> None:
    d = prov["definitions"]
    for key in ("sharpe", "volatility", "max_drawdown", "sortino", "cagr", "window", "dsr"):
        assert isinstance(d[key], str) and d[key], key
    assert d["sortino"].startswith("vectorbt") and d["cagr"].startswith("vectorbt")
    assert "second bar" in d["window"] and d["year_freq"] == "252 days"


def test_provenance_is_json_serialisable(prov: dict) -> None:
    assert json.loads(json.dumps(prov)) == prov


def test_extra_fields_are_merged_but_cannot_override_the_stack(prov: dict) -> None:
    p = provenance.provenance(tape_sha256="abc")
    assert p["tape_sha256"] == "abc"
    with pytest.raises(ValueError, match="override"):
        provenance.provenance(extra={"vectorbt": "9.9.9"})


# --- compare -----------------------------------------------------------------------------


def _mutated(prov: dict, **changes) -> dict:
    out = copy.deepcopy(prov)
    out.update(changes)
    return out


def test_same_stack(prov: dict) -> None:
    c = provenance.compare(prov, copy.deepcopy(prov))
    assert (c.flag, c.comparable) == ("SAME_STACK", True)


def test_engine_diff_is_informational(prov: dict) -> None:
    other = "numba" if prov["engine_effective"] == "rust" else "rust"
    c = provenance.compare(prov, _mutated(prov, engine_effective=other))
    assert (c.flag, c.comparable) == ("ENGINE_DIFF", True)


def test_patch_diff_for_other_libraries(prov: dict) -> None:
    c = provenance.compare(prov, _mutated(prov, numpy="9.9.9", created_utc="x"))
    assert (c.flag, c.comparable) == ("PATCH_DIFF", True)
    assert "numpy" in c.differences


def test_vectorbt_version_diff_is_conditionally_comparable(prov: dict) -> None:
    c = provenance.compare(prov, _mutated(prov, vectorbt="1.2.1", vectorbt_rust="1.2.1"))
    assert c.flag == "VBT_VERSION_DIFF" and c.comparable
    assert "golden" in c.note


def test_pandas_major_diff_is_not_comparable(prov: dict) -> None:
    c = provenance.compare(prov, _mutated(prov, pandas="2.3.3"))
    assert (c.flag, c.comparable) == ("PANDAS_MAJOR_DIFF", False)


def test_contract_diff_is_not_comparable(prov: dict) -> None:
    c = provenance.compare(prov, _mutated(prov, contract_version=prov["contract_version"] + 1))
    assert (c.flag, c.comparable) == ("CONTRACT_DIFF", False)


def test_a_different_tape_is_not_comparable(prov: dict) -> None:
    a = _mutated(prov, tape_sha256="aaa")
    c = provenance.compare(a, _mutated(prov, tape_sha256="bbb"))
    assert (c.flag, c.comparable) == ("TAPE_DIFF", False)
    assert c.differences["tape_sha256"] == ("aaa", "bbb")
    assert provenance.compare(a, _mutated(prov, tape_sha256="aaa")).comparable
    # an unrecorded tape is unknown, not equal
    assert not provenance.compare(a, _mutated(prov, tape_sha256=None)).comparable


def test_a_silent_definition_edit_is_not_comparable(prov: dict) -> None:
    edited = copy.deepcopy(prov)
    edited["definitions"]["atr"] = "a new atr"  # a silent edit without a contract bump
    assert not provenance.compare(prov, edited).comparable


@pytest.mark.parametrize(
    ("changes", "flag", "comparable"),
    [
        ({}, "SAME_STACK", True),
        ({"engine_effective": "numba"}, "ENGINE_DIFF", True),
        ({"numpy": "9.9.9"}, "PATCH_DIFF", True),
        ({"vectorbt": "1.2.1"}, "VBT_VERSION_DIFF", True),
        ({"pandas": "2.3.3"}, "PANDAS_MAJOR_DIFF", False),
        ({"tape_sha256": "x"}, "TAPE_DIFF", False),
        ({"contract_version": 99}, "CONTRACT_DIFF", False),
        ({"definitions": {"year_freq": "365 days"}}, "CONTRACT_DIFF", False),
    ],
    ids=["same", "engine", "patch", "vbt", "pandas_major", "tape", "contract", "definitions"],
)
def test_comparability_matrix_second_independent_check(prov: dict, changes, flag, comparable):
    """Every flag against an explicit table, so no non-comparable class can slip into the
    comparable set (the single-test protection the review flagged)."""
    base = _mutated(prov, engine_effective="rust", engine_requested="rust")
    other = _mutated(base, **changes)
    got = provenance.compare(base, other)
    assert (got.flag, got.comparable) == (flag, comparable)
    assert provenance.FLAGS.index(flag) >= 0
    non_comparable = {"PANDAS_MAJOR_DIFF", "TAPE_DIFF", "CONTRACT_DIFF", "EPOCH_DIFF"}
    assert provenance._COMPARABLE.isdisjoint(non_comparable)


def test_cross_epoch_comparison_is_refused_without_the_flag(prov: dict) -> None:
    other = _mutated(prov, baseline_epoch="E2")
    with pytest.raises(provenance.EpochMismatch, match="E1.*E2"):
        provenance.compare(prov, other)
    c = provenance.compare(prov, other, allow_cross_epoch=True)
    assert (c.flag, c.comparable) == ("EPOCH_DIFF", False)


def test_worst_difference_wins(prov: dict) -> None:
    other = _mutated(prov, numpy="9.9.9", engine_effective="x", pandas="2.3.3")
    assert provenance.compare(prov, other).flag == "PANDAS_MAJOR_DIFF"


def test_legacy_results_without_provenance_are_never_compared(prov: dict) -> None:
    with pytest.raises(provenance.EpochMismatch, match="legacy"):
        provenance.compare(None, prov)
    assert provenance.compare(None, prov, allow_cross_epoch=True).flag == "EPOCH_DIFF"


# --- result JSON --------------------------------------------------------------------------


def test_write_result_json_stamps_provenance(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    results.write_result_json(
        out,
        {"sharpe": 0.59, "nan": float("nan"), "ts": pd.Timestamp("2026-01-02"), "n": np.int64(3)},
    )
    data = json.loads(out.read_text())
    assert data["sharpe"] == 0.59 and data["n"] == 3
    assert data["nan"] is None and data["ts"].startswith("2026-01-02")
    assert data["provenance"]["vectorbt"] == provenance.provenance()["vectorbt"]
    assert data["provenance"]["definitions"]["year_freq"] == "252 days"


def test_write_result_json_refuses_a_payload_that_already_has_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        results.write_result_json(tmp_path / "x.json", {"provenance": {}})


def test_write_result_json_is_atomic_and_creates_parents(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "r.json"
    results.write_result_json(out, {"x": 1})
    assert out.exists() and not list(out.parent.glob("*.tmp"))


def test_read_result_json_rejects_a_file_without_provenance(tmp_path: Path) -> None:
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({"sharpe": 1.0}))
    with pytest.raises(ValueError, match="provenance"):
        results.read_result_json(legacy)
    new = tmp_path / "new.json"
    results.write_result_json(new, {"sharpe": 1.0})
    assert results.read_result_json(new)["sharpe"] == 1.0


def test_read_result_json_rejects_a_missing_year_freq(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"provenance": {"definitions": {}}}))
    with pytest.raises(ValueError, match="year_freq"):
        results.read_result_json(bad)


def test_compare_ignores_the_strategy_source_hash(prov: dict) -> None:
    other = {**prov, "strategy_sha256": "a" * 64, "strategy_files": {"x.py": "b" * 64}}
    assert provenance.compare(prov, other).flag == "SAME_STACK"
