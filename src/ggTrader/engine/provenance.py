"""Provenance: which stack produced a number, and whether two numbers can be compared.

Every result carries a provenance block (spec 5.8, schema 1) built from package metadata and
the dispatch probe, never from hand-written strings. ``compare`` classifies the difference
between two blocks and refuses to compare numbers across baseline epochs: an upgrade that
changes a number starts a new epoch instead of silently drifting (spec 6 item 7).

Epochs: E1 is the first baseline on vectorbt's own definitions (owner decision 2026-11-06:
vectorbt's definition wins where it differs from the old ggTrader one; the indicator presets
still reproduce the legacy signals, OD1(a)). A later definition change starts E2. Results from
the legacy engine carry no block at all.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ggTrader.engine import config, selfcheck, stats

SCHEMA = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_FILE = REPO_ROOT / "uv.lock"

#: Definitions that do not depend on the statistic set chosen for a run.
BASE_DEFINITIONS: dict[str, Any] = {
    "rsi": "wilder window=2n-1 ewm",
    "macd": "ema macd_ewm=signal_ewm=True",
    "bb_ddof": 1,
    "atr": "sma ewm=False",
    "sharpe": "vectorbt sharpe_ratio ddof=1 year_freq=252 days",
    "volatility": "vectorbt annualized_volatility ddof=1 year_freq=252 days",
    "max_drawdown": "vectorbt max_drawdown of the compounded window path",
    "window": "equity rebased at the first bar, returns from the second bar (curve_stats window)",
    "dsr": "var_sharpe=252/(T-1) bias=False",
    "year_freq": config.YEAR_FREQ,
}


def definitions() -> dict[str, Any]:
    """The per-statistic definitions every result is computed with (vectorbt's)."""
    return {**BASE_DEFINITIONS, **stats.DEFINITION_LABELS}


#: Library versions recorded from package metadata (key -> distribution name).
_LIBRARIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "numba": "numba",
    "llvmlite": "llvmlite",
    "scipy": "scipy",
}
#: Mildest first; the worst flag among a comparison's differences wins.
FLAGS = (
    "SAME_STACK",
    "ENGINE_DIFF",
    "PATCH_DIFF",
    "VBT_VERSION_DIFF",
    "PANDAS_MAJOR_DIFF",
    "TAPE_DIFF",
    "MEMBERSHIP_DIFF",
    "CONTRACT_DIFF",
    "EPOCH_DIFF",
)
_COMPARABLE = {"SAME_STACK", "ENGINE_DIFF", "PATCH_DIFF", "VBT_VERSION_DIFF"}
#: Membership keys (spec 3.5): ``membership`` is the detail block (informational, ignored);
#: ``membership_sha256`` is the identity of the S&P 500 list read, handled by its own flag.
MEMBERSHIP_KEYS = ("membership", "membership_sha256")
#: Informational keys: when and from which code the result was made, not which stack ran it.
_IGNORED_KEYS = {
    "membership",
    "membership_sha256",
    "created_utc",
    "git_sha",
    "git_dirty",
    "lock_sha256",
    "rust_calls",
    "strategy_sha256",
    "strategy_files",
    "strategy_sources",
}


class EpochMismatch(ValueError):
    """Two results from different baseline epochs (or a legacy result) were compared."""


@dataclass(frozen=True)
class Comparison:
    """How two provenance blocks differ and whether their numbers can be set side by side."""

    flag: str
    comparable: bool
    differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    note: str = ""


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return ""


def _lock_sha256() -> str | None:
    try:
        return hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def provenance(
    *,
    tape_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
    membership: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block for a result produced now.

    Runs the engine dispatch probe, so ``engine_effective`` is what ran, not what was asked
    for; an engine that is not running as configured raises ``EngineError`` instead of
    producing a stamped number.

    Args:
        tape_sha256: hash of the frozen data snapshot behind the result, if any.
        extra: additional fields; they may not override any stack field.
        membership: the ``MembershipSource.provenance(asof)`` fields (``membership_sha256`` and
            the ``membership`` block). Added only when given: a record without them says nothing
            about which S&P 500 list it used (``compare`` treats that as unknown).
    """
    probe = selfcheck.selfcheck()
    block: dict[str, Any] = {
        "schema": SCHEMA,
        "python": platform.python_version(),
        **{key: _version(dist) for key, dist in _LIBRARIES.items()},
        "vectorbt": probe["vectorbt"],
        "vectorbt_rust": probe["vectorbt_rust"],
        "engine_requested": config.active_engine(),
        "engine_effective": probe["engine"],
        "rust_calls": probe["rust_calls"],
        "lock_sha256": _lock_sha256(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "contract_version": config.CONTRACT_VERSION,
        "baseline_epoch": config.BASELINE_EPOCH,
        "definitions": definitions(),
        "tape_sha256": tape_sha256,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for key, value in (extra or {}).items():
        if key in block:
            raise ValueError(f"extra may not override the stack field {key!r}")
        block[key] = value
    if membership is not None:
        unknown = set(membership) - set(MEMBERSHIP_KEYS)
        if unknown or "membership_sha256" not in membership:
            raise ValueError(f"membership must hold {MEMBERSHIP_KEYS}, got {sorted(membership)}")
        block.update(membership)
    return block


def _major(version: str) -> str:
    return version.split(".")[0]


def compare(
    a: dict[str, Any] | None, b: dict[str, Any] | None, *, allow_cross_epoch: bool = False
) -> Comparison:
    """Classify the difference between two provenance blocks (worst difference wins).

    Args:
        a: provenance of one result; ``None`` for a legacy result (vectorbt 0.28.5).
        b: provenance of the other result.
        allow_cross_epoch: permit the call across baseline epochs; the result is then
            ``EPOCH_DIFF`` and ``comparable`` is False (the delta is the re-baseline).

    Raises:
        EpochMismatch: different baseline epochs or a legacy result, without the flag.
    """
    if a is None or b is None:
        if not allow_cross_epoch:
            raise EpochMismatch("a legacy result (no provenance) cannot be compared with E1+")
        return Comparison("EPOCH_DIFF", False, note="legacy result versus a provenance-stamped one")
    if a["baseline_epoch"] != b["baseline_epoch"]:
        if not allow_cross_epoch:
            raise EpochMismatch(
                f"baseline epochs differ ({a['baseline_epoch']} vs {b['baseline_epoch']}); "
                "pass allow_cross_epoch=True to see the classification"
            )
        return Comparison(
            "EPOCH_DIFF",
            False,
            {"baseline_epoch": (a["baseline_epoch"], b["baseline_epoch"])},
            "number deltas across epochs are the re-baseline, not an effect",
        )
    diffs = {
        key: (a.get(key), b.get(key))
        for key in sorted(set(a) | set(b))
        if key not in _IGNORED_KEYS and a.get(key) != b.get(key)
    }
    flags, note = ["SAME_STACK"], ""
    engine_only = "engine_effective" in diffs and set(diffs) <= {
        "engine_effective",
        "engine_requested",
    }
    if engine_only:
        flags.append("ENGINE_DIFF")
    elif diffs:
        flags.append("PATCH_DIFF")
    if "vectorbt" in diffs or "vectorbt_rust" in diffs:
        flags.append("VBT_VERSION_DIFF")
        note = "comparable only if the golden suite passed on both versions"
    if "pandas" in diffs and _major(a["pandas"]) != _major(b["pandas"]):
        flags.append("PANDAS_MAJOR_DIFF")
    if "tape_sha256" in diffs:
        flags.append("TAPE_DIFF")
        note = "different data snapshots: a number gap may be the tape, not the engine"
    sha_a, sha_b = a.get("membership_sha256"), b.get("membership_sha256")
    if sha_a and sha_b and sha_a != sha_b:
        diffs["membership_sha256"] = (sha_a, sha_b)
        flags.append("MEMBERSHIP_DIFF")
        note = "different S&P 500 membership lists: a number gap may be the list, not the engine"
    elif bool(sha_a) != bool(sha_b):
        note = note or "membership list unknown on one side (no membership_sha256 recorded)"
    if "contract_version" in diffs or "definitions" in diffs:
        flags.append("CONTRACT_DIFF")
        if "definitions" in diffs:
            note = "statistic definitions differ: the numbers measure different things"
    flag = max(flags, key=FLAGS.index)
    return Comparison(flag, flag in _COMPARABLE, diffs, note)
