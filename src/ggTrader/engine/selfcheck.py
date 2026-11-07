"""Engine self-check: prove the configured vectorbt engine is the one that really runs.

``engine="auto"`` silently falls back to numba when the vectorbt-rust wheel is missing or its
major.minor differs from vectorbt's (spec 5.5). The probe wraps the Rust simulator, runs a
two-column trailing-stop ``from_signals`` and raises unless the number of Rust calls matches
the configured engine. This is the only module that touches vectorbt's private ``_engine``
(spec 5.2 rule 6). Run it with ``python -m ggTrader.engine.selfcheck``; it exits non-zero on
any failure, so the Dockerfile and CI can gate on it.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt import _engine

from ggTrader.engine import config

PROBE_BARS = 50


class EngineError(RuntimeError):
    """The vectorbt engine is not configured or not running as required."""


def installed_versions() -> dict[str, str]:
    """Versions of the vectorbt pair from package metadata, never from hand-written strings."""
    out = {}
    for key, dist in (("vectorbt", "vectorbt"), ("vectorbt_rust", "vectorbt-rust")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = ""
    return out


def check_version_pair() -> None:
    """vectorbt and vectorbt-rust must share major.minor, or vectorbt falls back to numba."""
    versions = installed_versions()
    if not versions["vectorbt_rust"]:
        raise EngineError("vectorbt-rust is not installed (install vectorbt[rust])")
    minors = {k: ".".join(v.split(".")[:2]) for k, v in versions.items()}
    if minors["vectorbt"] != minors["vectorbt_rust"]:
        raise EngineError(
            f"vectorbt {versions['vectorbt']} and vectorbt-rust {versions['vectorbt_rust']} "
            "differ in major.minor; vectorbt would silently fall back to numba"
        )


def rust_available() -> bool:
    """Whether vectorbt will accept the Rust wheel; a mismatch warning is fatal here."""
    _engine.clear_engine_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return bool(_engine.is_rust_available())


def _run_probe(engine: str | None) -> dict[str, Any]:
    """Run a tiny trailing-stop from_signals and count calls into the Rust simulator."""
    import vectorbt_rust.portfolio as rust_portfolio

    calls = {"n": 0}
    original = rust_portfolio.simulate_from_signals_rs

    @functools.wraps(original)
    def counted(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)

    rust_portfolio.simulate_from_signals_rs = counted
    try:
        close = pd.DataFrame(np.linspace(10.0, 12.0, PROBE_BARS)[:, None] * [1.0, 1.1])
        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        entries.iloc[3] = True
        extra = {} if engine is None else {"engine": engine}
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            ~entries,
            size=0.5,
            size_type="percent",
            sl_stop=0.05,
            sl_trail=True,
            cash_sharing=True,
            group_by=True,
            freq=config.BAR_FREQ,
            **extra,
        )
        final_value = float(portfolio.value().iloc[-1])
    finally:
        rust_portfolio.simulate_from_signals_rs = original
    return {
        "engine": "rust" if calls["n"] > 0 else "numba",
        "rust_calls": calls["n"],
        "final_value": final_value,
    }


def selfcheck(required: str | None = None, *, pin: bool = False) -> dict[str, Any]:
    """Prove the engine runs as configured; raise ``EngineError`` otherwise.

    Args:
        required: engine the kernel must run on; ``None`` means the configured engine.
        pin: also make ``required`` the process-wide engine (used by the smoke script).

    Returns:
        Versions, the engine that ran, the Rust call count and the probe's final value.
    """
    if required is not None and required not in config.ENGINES:
        raise ValueError(f"engine must be one of {config.ENGINES}, got {required!r}")
    if pin and required is not None:
        config.set_engine(required)
    configured = config.active_engine()
    if configured not in config.ENGINES:
        raise EngineError(
            f"vectorbt engine is {configured!r}; it must be one of {config.ENGINES} (never auto)"
        )
    expected = required or configured
    check_version_pair()
    if expected == "rust" and not rust_available():
        raise EngineError("engine=rust required but vectorbt-rust is missing or mismatched")
    probe = _run_probe(None if required is None or required == configured else expected)
    if probe["engine"] != expected:
        raise EngineError(f"engine pinned to {expected} but the kernel ran on {probe['engine']}")
    versions = installed_versions()
    return {**versions, **probe, "year_freq": str(vbt.settings["returns"]["year_freq"])}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 0 only when the configured engine really ran."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=config.ENGINES, default=None, help="required engine")
    args = parser.parse_args(argv)
    try:
        result = selfcheck(args.engine)
    except Exception as exc:  # noqa: BLE001 - any failure must fail the build
        print(f"ENGINE CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
