"""Tests for the vectorbt engine smoke check used by the Dockerfile and CI."""

from __future__ import annotations

import pytest

from scripts.check_vbt_engine import check_engine, main

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _restore_engine_setting():
    """check_engine pins the process-wide engine; put it back for the rest of the suite."""
    import vectorbt as vbt

    before = vbt.settings["engine"]
    yield
    vbt.settings["engine"] = before


def test_rust_engine_dispatches_to_rust() -> None:
    result = check_engine("rust")
    assert result["engine"] == "rust"
    assert result["rust_calls"] >= 1
    assert result["vectorbt"] == result["vectorbt_rust"]


def test_numba_engine_never_calls_rust() -> None:
    result = check_engine("numba")
    assert result["engine"] == "numba"
    assert result["rust_calls"] == 0


def test_engines_agree_on_the_simulated_value() -> None:
    assert check_engine("rust")["final_value"] == check_engine("numba")["final_value"]


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        check_engine("auto")


def test_main_fails_when_rust_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from vectorbt import _engine

    monkeypatch.setattr(_engine, "is_rust_available", lambda: False)
    assert main(["--engine", "rust"]) == 1
