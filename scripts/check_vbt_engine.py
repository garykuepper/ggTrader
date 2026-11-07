"""Image/CI smoke check: prove vectorbt dispatches its simulator to the Rust engine.

Thin wrapper over ``ggTrader.engine.selfcheck`` (the single implementation, spec 5.5):
`engine="auto"` silently falls back to numba when the vectorbt-rust wheel is missing or its
minor version does not match vectorbt (a 20-30 s JIT penalty per fresh container, no error).
This script fails loudly instead: it pins the engine, turns the mismatch warning into an
error, counts calls into the Rust simulator, and runs one tiny `from_signals`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ggTrader.engine import selfcheck

ENGINES = ("rust", "numba")


def check_engine(required: str = "rust") -> dict[str, Any]:
    """Pin the engine, run one tiny from_signals and return what actually ran.

    Args:
        required: "rust" or "numba"; the engine the kernel must run on.

    Returns:
        Versions, the engine that ran, the number of Rust simulator calls, the final value.

    Raises:
        ValueError: if ``required`` is not "rust" or "numba" (``auto`` is rejected).
        RuntimeError: if the Rust wheel is unavailable or the kernel ran on another engine.
    """
    if required not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {required!r}")
    return selfcheck.selfcheck(required, pin=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 0 only when the pinned engine really ran."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, default="rust", help="required engine")
    args = parser.parse_args(argv)
    try:
        result = check_engine(args.engine)
    except Exception as exc:  # noqa: BLE001 - any failure must fail the build
        print(f"ENGINE CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
