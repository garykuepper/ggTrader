"""Validate and normalize EXIT_TOURNAMENT config values."""

from __future__ import annotations

from typing import Any


def parse_exit_tournament(
    raw_exits: Any,
    exit_registry: dict[str, Any],
    default: str = "atr_trailing",
) -> list[str]:
    """Return a validated ordered list of exit names; warn and skip unknown keys."""
    if raw_exits is None:
        return list(exit_registry.keys())
    if not isinstance(raw_exits, (list, tuple)):
        raw_exits = [raw_exits]
    out: list[str] = []
    for name in raw_exits:
        if name not in exit_registry:
            print(f"  WARNING: EXIT_TOURNAMENT contains unknown exit '{name}' — skipping.")
        else:
            out.append(str(name))
    if not out:
        print(f"  WARNING: No valid exits in EXIT_TOURNAMENT; defaulting to {default}.")
        return [default]
    return out
