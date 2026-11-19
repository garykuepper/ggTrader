"""Tests for the shared cached database engine."""

from unittest.mock import MagicMock

import pytest


def test_get_engine_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create the engine once and restore any existing cache afterward."""
    from ggTrader.utils import db

    engine = MagicMock()
    create = MagicMock(return_value=engine)
    monkeypatch.setattr(db, "_ENGINE", None)
    monkeypatch.setattr(db, "get_db_connection_string", lambda: "sqlite://")
    monkeypatch.setattr(db, "create_engine", create)
    assert db.get_engine() is engine
    assert db.get_engine() is engine
    create.assert_called_once_with("sqlite://")


def test_old_names_are_the_new_objects() -> None:
    """Keep the historical imports bound to the shared function."""
    from ggTrader.lab import persist as lab_persist
    from ggTrader.paper import persist as paper_persist
    from ggTrader.utils import db

    assert lab_persist.get_engine is db.get_engine
    assert paper_persist._get_engine is db.get_engine


def test_attribute_patch_on_lab_persist_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema initialization respects a patched lab-level engine function."""
    from ggTrader.lab import persist

    engine = MagicMock()
    monkeypatch.setattr(persist, "get_engine", lambda: engine)
    persist.init_schema()
    assert engine.begin.return_value.__enter__.return_value.execute.call_count == 5
