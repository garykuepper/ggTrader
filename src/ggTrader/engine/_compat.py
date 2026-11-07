"""Version shims for upstream vectorbt changes: empty at the start of epoch E1.

Convention (spec 6, S2 section 3): version-conditional code lives only in this module, one
function per upstream change, written as::

    @shim("short_name", since="1.2.1", reason="what upstream changed", url="release URL",
          still_needed=lambda: <True while the workaround is required>)
    def short_name(...): ...

Gate by feature (``inspect.signature`` of the renamed argument), never by ``if version >``.
``tests/engine/test_compat.py`` runs every registered ``still_needed`` probe and fails the
moment one returns False, so a shim is deleted when upstream fixes the problem instead of
accumulating. There are no shims today.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Shim:
    """One registered workaround and the probe that says whether it is still needed."""

    name: str
    since: str
    reason: str
    url: str
    still_needed: Callable[[], bool]
    func: Callable[..., Any]


SHIMS: dict[str, Shim] = {}


def shim(
    name: str, *, since: str, reason: str, url: str, still_needed: Callable[[], bool]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a workaround; see the module docstring for the convention."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in SHIMS:
            raise ValueError(f"shim {name!r} is already registered")
        SHIMS[name] = Shim(name, since, reason, url, still_needed, func)
        return func

    return decorate
