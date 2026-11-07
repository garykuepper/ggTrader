"""Failure-message helpers for the upstream tripwires (P8b).

A tripwire that fails during an upgrade must let the agent choose ACCEPT, RE-BASELINE or
REJECT without reading vectorbt source. ``explain`` builds that message: it states which side
is the reference (an independent formula or a recorded upstream behaviour), which side is the
installed vectorbt, the size of the gap, the probable upstream change and the triage item in
``.claude/skills/vbt-upgrade/SKILL.md``.
"""

from __future__ import annotations

from importlib import metadata


def installed() -> str:
    """The installed pair, for a failure message."""
    pair = []
    for name in ("vectorbt", "vectorbt-rust"):
        try:
            pair.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pair.append(f"{name} (not installed)")
    return " / ".join(pair)


def relative_gap(installed_value: float, reference_value: float) -> str:
    """``|a-b|/|b|`` as text, tolerant of zero and non-finite inputs."""
    try:
        if reference_value == 0:
            return f"absolute {abs(installed_value - reference_value):.3g}"
        return f"relative {abs(installed_value - reference_value) / abs(reference_value):.3g}"
    except TypeError:
        return "n/a"


def explain(
    quantity: str,
    *,
    installed_value: object,
    reference_value: object,
    reference: str,
    cause: str,
    triage: str,
) -> str:
    """The standard tripwire message (see the module docstring)."""
    gap = ""
    if isinstance(installed_value, (int, float)) and isinstance(reference_value, (int, float)):
        gap = f" (gap {relative_gap(float(installed_value), float(reference_value))})"
    return (
        f"\n  quantity:   {quantity}"
        f"\n  installed:  {installed_value!r}   <- {installed()}"
        f"\n  reference:  {reference_value!r}   <- {reference}{gap}"
        f"\n  probably:   {cause}"
        f"\n  next:       {triage}"
    )
