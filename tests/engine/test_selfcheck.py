"""L3: the dispatch probe proves the configured engine really ran."""

from __future__ import annotations

import subprocess
import sys

import pytest

from ggTrader.engine import config, selfcheck


def test_rust_is_dispatched_when_configured() -> None:
    """Fails if the Rust simulator is not what actually ran (silent numba fallback)."""
    config.set_engine("rust")
    result = selfcheck.selfcheck()
    assert result["engine"] == "rust"
    assert result["rust_calls"] >= 1
    assert result["vectorbt"] == result["vectorbt_rust"]


def test_numba_never_calls_rust() -> None:
    config.set_engine("numba")
    result = selfcheck.selfcheck()
    assert result["engine"] == "numba"
    assert result["rust_calls"] == 0


def test_both_engines_simulate_the_same_value() -> None:
    config.set_engine("rust")
    rust = selfcheck.selfcheck()["final_value"]
    config.set_engine("numba")
    numba = selfcheck.selfcheck()["final_value"]
    assert rust == numba


def test_probe_fails_when_the_kernel_runs_on_another_engine(monkeypatch) -> None:
    """If Rust dispatch silently stops, the probe must raise, not report numba."""
    config.set_engine("rust")
    real = selfcheck._run_probe

    def fake_probe(engine: str | None) -> dict:
        return real("numba")

    monkeypatch.setattr(selfcheck, "_run_probe", fake_probe)
    with pytest.raises(selfcheck.EngineError, match="ran on numba"):
        selfcheck.selfcheck()


def test_auto_is_rejected() -> None:
    import vectorbt as vbt

    vbt.settings["engine"] = "auto"
    with pytest.raises(selfcheck.EngineError, match="auto"):
        selfcheck.selfcheck()


def test_unknown_required_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        selfcheck.selfcheck("auto")


def test_missing_rust_wheel_fails_loudly(monkeypatch) -> None:
    from vectorbt import _engine

    config.set_engine("rust")
    monkeypatch.setattr(_engine, "is_rust_available", lambda: False)
    with pytest.raises(selfcheck.EngineError, match="vectorbt-rust"):
        selfcheck.selfcheck()


def test_pin_switches_the_process_engine() -> None:
    config.set_engine("rust")
    selfcheck.selfcheck("numba", pin=True)
    assert config.active_engine() == "numba"


def test_major_minor_pair_check_accepts_the_installed_pair() -> None:
    selfcheck.check_version_pair()


def _installed_parts() -> tuple[str, int]:
    """(major, minor) of the installed vectorbt: the tests below derive their cases from it."""
    major, minor, *_ = selfcheck.installed_versions()["vectorbt"].split(".")
    return major, int(minor)


def test_major_minor_pair_check_rejects_a_mismatch(monkeypatch) -> None:
    """The next minor of whatever is installed, never a literal that goes stale (P8, R2)."""
    real = selfcheck.installed_versions
    major, minor = _installed_parts()
    monkeypatch.setattr(
        selfcheck,
        "installed_versions",
        lambda: {**real(), "vectorbt_rust": f"{major}.{minor + 1}.0"},
    )
    with pytest.raises(selfcheck.EngineError, match="major.minor"):
        selfcheck.check_version_pair()


def test_a_different_major_is_rejected_too(monkeypatch) -> None:
    real = selfcheck.installed_versions
    major, minor = _installed_parts()
    monkeypatch.setattr(
        selfcheck,
        "installed_versions",
        lambda: {**real(), "vectorbt_rust": f"{int(major) + 1}.{minor}.0"},
    )
    with pytest.raises(selfcheck.EngineError, match="major.minor"):
        selfcheck.check_version_pair()


def test_patch_difference_is_allowed(monkeypatch) -> None:
    real = selfcheck.installed_versions
    major, minor = _installed_parts()
    monkeypatch.setattr(
        selfcheck, "installed_versions", lambda: {**real(), "vectorbt_rust": f"{major}.{minor}.9"}
    )
    selfcheck.check_version_pair()


@pytest.mark.parametrize("pair", [("1.1.2", "1.2.0"), ("1.2.0", "1.1.9"), ("2.0.0", "1.2.0")])
def test_pair_check_rejects_a_fixed_set_of_mismatched_pairs(monkeypatch, pair) -> None:
    """Literal pairs: independent of the installed version, so a bump cannot change the verdict."""
    monkeypatch.setattr(
        selfcheck,
        "installed_versions",
        lambda: {"vectorbt": pair[0], "vectorbt_rust": pair[1]},
    )
    with pytest.raises(selfcheck.EngineError, match="major.minor"):
        selfcheck.check_version_pair()


@pytest.mark.parametrize("pair", [("1.1.2", "1.1.0"), ("1.2.0", "1.2.7")])
def test_pair_check_allows_a_fixed_set_of_patch_differences(monkeypatch, pair) -> None:
    monkeypatch.setattr(
        selfcheck,
        "installed_versions",
        lambda: {"vectorbt": pair[0], "vectorbt_rust": pair[1]},
    )
    selfcheck.check_version_pair()


def test_module_entry_point_exits_zero_on_success() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "ggTrader.engine.selfcheck"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "rust" in proc.stdout


def test_module_entry_point_exits_nonzero_on_failure() -> None:
    import os

    env = {**os.environ, "GGT_VBT_ENGINE": "auto"}
    proc = subprocess.run(
        [sys.executable, "-m", "ggTrader.engine.selfcheck"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "GGT_VBT_ENGINE" in proc.stderr


def test_main_returns_one_on_failure(monkeypatch) -> None:
    def boom(*a, **k):
        raise selfcheck.EngineError("nope")

    monkeypatch.setattr(selfcheck, "selfcheck", boom)
    assert selfcheck.main([]) == 1


@pytest.mark.parametrize(("configured", "asked"), [("rust", "numba"), ("numba", "rust")])
def test_probe_reports_the_engine_that_ran_not_the_one_configured(
    configured: str, asked: str
) -> None:
    """Second, independent test: drive the probe directly against the opposite engine."""
    config.set_engine(configured)
    result = selfcheck._run_probe(asked)
    assert result["engine"] == asked
    assert (result["rust_calls"] > 0) is (asked == "rust")


def test_probe_counts_calls_through_a_wrapper_it_restores(monkeypatch) -> None:
    import vectorbt_rust.portfolio as rp

    config.set_engine("rust")
    before = rp.simulate_from_signals_rs
    selfcheck._run_probe(None)
    assert rp.simulate_from_signals_rs is before  # the counter never leaks into later runs
