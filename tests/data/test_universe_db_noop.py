"""``is_noop_reload``: identical bytes are a no-op only under the same normalizer version."""

from __future__ import annotations

from ggTrader.data import universe_db
from ggTrader.data.universe_db import NORMALIZER_VERSION, is_noop_reload

RAW = "a" * 64


def _load(raw: str = RAW, version: int = NORMALIZER_VERSION) -> dict:
    return {"raw_sha256": raw, "normalizer_version": version}


def test_same_bytes_and_same_normalizer_is_a_no_op() -> None:
    assert is_noop_reload(_load(), RAW)


def test_different_bytes_are_not_a_no_op() -> None:
    assert not is_noop_reload(_load(), "b" * 64)


def test_same_bytes_under_a_different_normalizer_version_is_not_a_no_op(monkeypatch) -> None:
    assert is_noop_reload(_load(), RAW)
    monkeypatch.setattr(universe_db, "NORMALIZER_VERSION", NORMALIZER_VERSION + 1)
    assert not is_noop_reload(_load(version=NORMALIZER_VERSION), RAW)  # stored one is now old
    assert is_noop_reload(_load(version=NORMALIZER_VERSION + 1), RAW)
