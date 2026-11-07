"""L0: every version shim must still be needed, or it is deleted (spec 6, S2 section 3)."""

from __future__ import annotations

import pytest

from ggTrader.engine import _compat


def test_every_registered_shim_is_still_needed() -> None:
    """Fails when upstream fixes the problem a shim works around: delete the shim."""
    for name, entry in _compat.SHIMS.items():
        assert entry.still_needed(), f"shim {name!r} is no longer needed; delete it"


def test_every_shim_documents_its_origin() -> None:
    for name, entry in _compat.SHIMS.items():
        assert entry.since and entry.reason and entry.url.startswith("http"), name


def test_registration_convention_works_and_rejects_duplicates() -> None:
    @_compat.shim(
        "tmp", since="1.2.1", reason="x", url="https://example.invalid", still_needed=lambda: True
    )
    def tmp() -> int:
        return 1

    try:
        assert _compat.SHIMS["tmp"].func() == 1
        with pytest.raises(ValueError, match="already registered"):
            _compat.shim("tmp", since="1", reason="x", url="https://x", still_needed=lambda: True)(
                tmp
            )
    finally:
        _compat.SHIMS.pop("tmp", None)


def test_a_shim_that_became_unnecessary_is_detected() -> None:
    @_compat.shim(
        "stale",
        since="1.2.1",
        reason="x",
        url="https://example.invalid",
        still_needed=lambda: False,
    )
    def stale() -> None:
        return None

    try:
        with pytest.raises(AssertionError, match="no longer needed"):
            test_every_registered_shim_is_still_needed()
    finally:
        _compat.SHIMS.pop("stale", None)
