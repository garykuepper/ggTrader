"""Check that every code hash pinned in the research evidence still resolves to real code.

Frozen research reports record ``(repo path, sha256)`` pairs for the code that produced them,
plus a few hashes of ``{path: sha256}`` maps (the observer and feasibility ``code_sha256``). When
study code moves out of ``src/`` and ``scripts/`` those pins must keep resolving, or a reader can
no longer prove which code made a number. Each pin resolves one of four ways:

* ``MATCH``          the file at the recorded path has the recorded sha256;
* ``MOVED-MATCH``    ``research/CODEMOVED.csv`` says it moved, and the file at ``new_path`` matches;
* ``HISTORY-MATCH``  some earlier commit holds the recorded bytes at the recorded path;
* ``FAIL``           none of the above.

Walking history needs every commit, so a shallow clone is refused (exit 2). Exit 1 means a pin
failed or ``--expect`` did not hold. The script only reads git and files; it writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEMOVED = "research/CODEMOVED.csv"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_PATH_RE = re.compile(r"^(src|scripts|tests)/[\w./-]+\.(py|sh)$")
STATUSES = ("MATCH", "MOVED-MATCH", "HISTORY-MATCH", "FAIL")

OBSERVER_DIR = "research/intraday/spy-momentum-prospective-observer"
FEASIBILITY_DIR = "research/intraday/spy-momentum-feasibility"
#: The recorder's CODE_FILES as of release 280d4a0 (old paths: they are read from git history).
OBSERVER_FILES = (
    "src/ggTrader/lab/intraday_observer.py",  # history-read
    "scripts/intraday_observe.py",  # history-read
    "scripts/intraday_observer_job.py",  # history-read
    "src/ggTrader/lab/intraday_probe.py",  # history-read
)
#: The rehearsal ran before the job wrapper existed, so its map has three files.
REHEARSAL_FILES = (
    "src/ggTrader/lab/intraday_observer.py",  # history-read
    "scripts/intraday_observe.py",  # history-read
    "src/ggTrader/lab/intraday_probe.py",  # history-read
)
#: intraday_momentum.CODE_FILES as of commit 6e89281c.
FEASIBILITY_FILES = (
    "src/ggTrader/lab/intraday_momentum.py",  # history-read
    "src/ggTrader/lab/intraday_momentum_data.py",  # history-read
    "src/ggTrader/lab/intraday_probe.py",  # history-read
    "scripts/intraday_momentum_research.py",  # history-read
)


@dataclass(frozen=True)
class MapSpec:
    """A recorded hash of a ``{path: sha256}`` code map, and how to find the code behind it."""

    json_glob: str
    pointer: tuple[str, ...]
    files: tuple[str, ...]
    commit_pointer: tuple[str, ...] = ()
    commit_hint: str = ""


def canonical_maps() -> tuple[MapSpec, ...]:
    """The recorded code-map hashes that ``collect_pins`` cannot see as path/sha pairs."""
    return (
        MapSpec(
            f"{OBSERVER_DIR}/_intraday_observer_activation_verification.json",
            ("code_sha256",),
            OBSERVER_FILES,
            commit_pointer=("release_commit",),
        ),
        MapSpec(
            f"{OBSERVER_DIR}/_intraday_observer_rehearsal.json",
            ("identity", "bindings", "code"),
            REHEARSAL_FILES,
        ),
        MapSpec(
            f"{OBSERVER_DIR}/_intraday_observer_readiness_preflight.json",
            ("final_code_sha256",),
            OBSERVER_FILES,
            commit_pointer=("verification", "code_commit"),
        ),
        MapSpec(
            f"{FEASIBILITY_DIR}/_intraday_momentum_validation*.json",
            ("identities", "code_sha256"),
            FEASIBILITY_FILES,
            commit_hint="6e89281c",
        ),
    )


@dataclass(frozen=True)
class Pin:
    """One recorded ``(repo path, sha256)`` pair and where it was found."""

    source: str
    where: str
    path: str
    sha256: str


@dataclass
class Result:
    """The outcome of resolving one pin or map."""

    label: str
    status: str
    detail: str = ""


@dataclass
class Report:
    """All results of a run."""

    pins: list[Result] = field(default_factory=list)
    maps: list[Result] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Pin outcomes by status; a failed map counts once under FAIL."""
        out = dict.fromkeys(STATUSES, 0)
        for result in self.pins:
            out[result.status] += 1
        out["FAIL"] += sum(1 for m in self.maps if m.status == "FAIL")
        return out

    def summary(self) -> str:
        """One line: pin count, the four statuses and the maps that resolved."""
        c = self.counts()
        resolved = sum(1 for m in self.maps if m.status != "FAIL")
        return (
            f"pins={len(self.pins)} MATCH={c['MATCH']} MOVED-MATCH={c['MOVED-MATCH']} "
            f"HISTORY-MATCH={c['HISTORY-MATCH']} FAIL={c['FAIL']} maps={resolved}/{len(self.maps)}"
        )


# --- git ------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)


def is_shallow(repo: Path) -> bool:
    """True when the checkout lacks history, so history pins could not be resolved."""
    out = _git(repo, "rev-parse", "--is-shallow-repository").stdout.decode().strip()
    return out == "true"


def blob_at(repo: Path, commit: str, path: str) -> bytes | None:
    """The bytes of ``path`` at ``commit``, or ``None`` when the path is absent there."""
    done = _git(repo, "show", f"{commit}:{path}")
    return done.stdout if done.returncode == 0 else None


def history_commits(repo: Path, *paths: str) -> list[str]:
    """Commits reachable from HEAD that touched any of ``paths`` (deletions included)."""
    out = _git(repo, "log", "--full-history", "--format=%H", "--", *paths).stdout.decode()
    return out.split()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: object) -> str:
    """sha256 of canonical JSON, the recipe of ``intraday_probe.canonical_hash``."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha(body)


# --- collecting pins ------------------------------------------------------------------------


def _is_code_path(value: object) -> bool:
    return isinstance(value, str) and CODE_PATH_RE.match(value) is not None


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.match(value) is not None


def _walk(node: object, source: str, where: str, out: list[Pin]) -> None:
    if isinstance(node, dict):
        if _is_code_path(node.get("path")) and _is_sha(node.get("sha256")):
            out.append(Pin(source, where, node["path"], node["sha256"]))
        for key, value in node.items():
            if _is_code_path(key) and _is_sha(value):
                out.append(Pin(source, f"{where}/{key}", key, value))
            else:
                _walk(value, source, f"{where}/{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, source, f"{where}[{index}]", out)


def collect_pins(repo: Path) -> list[Pin]:
    """Every ``(code path, sha256)`` pair recorded in a JSON file under ``research/``."""
    pins: list[Pin] = []
    for path in sorted((repo / "research").rglob("*.json")):
        try:
            tree = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        _walk(tree, path.relative_to(repo).as_posix(), "", pins)
    return pins


def read_codemoved(repo: Path) -> dict[str, str]:
    """``old_path -> new_path`` for every moved row of ``research/CODEMOVED.csv``."""
    path = repo / CODEMOVED
    if not path.is_file():
        raise FileNotFoundError(f"{CODEMOVED} is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {r["old_path"]: r["new_path"] for r in rows if r["new_path"] and r["kind"] != "deleted"}


# --- resolving ------------------------------------------------------------------------------


def resolve_pin(repo: Path, pin: Pin, moved: dict[str, str]) -> Result:
    """Resolve one pin: MATCH, MOVED-MATCH, HISTORY-MATCH or FAIL."""
    label = f"{pin.path}  [{pin.source}]"
    here = repo / pin.path
    if here.is_file() and _sha(here.read_bytes()) == pin.sha256:
        return Result(label, "MATCH")
    new_path = moved.get(pin.path)
    if new_path and (repo / new_path).is_file():
        if _sha((repo / new_path).read_bytes()) == pin.sha256:
            return Result(label, "MOVED-MATCH", new_path)
    for commit in history_commits(repo, pin.path):
        data = blob_at(repo, commit, pin.path)
        if data is not None and _sha(data) == pin.sha256:
            return Result(label, "HISTORY-MATCH", commit[:8])
    return Result(label, "FAIL", "no file, move or commit holds these bytes")


def _dig(tree: object, pointer: tuple[str, ...]) -> object:
    for key in pointer:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def _map_hash_at(repo: Path, commit: str, files: tuple[str, ...]) -> str | None:
    shas: dict[str, str] = {}
    for name in files:
        data = blob_at(repo, commit, name)
        if data is None:
            return None
        shas[name] = _sha(data)
    return canonical_hash(shas)


def resolve_map(repo: Path, json_path: Path, spec: MapSpec) -> Result:
    """Find a commit at which the listed files hash to the map hash the JSON recorded."""
    rel = json_path.relative_to(repo).as_posix()
    tree = json.loads(json_path.read_text(encoding="utf-8"))
    recorded = _dig(tree, spec.pointer)
    label = f"{rel}  [{'/'.join(spec.pointer)}]"
    if not _is_sha(recorded):
        return Result(label, "FAIL", "no recorded hash at the pointer")
    hints = [spec.commit_hint] if spec.commit_hint else []
    named = _dig(tree, spec.commit_pointer) if spec.commit_pointer else None
    if isinstance(named, str):
        hints.insert(0, named)
    for commit in [*hints, *history_commits(repo, *spec.files)]:
        if _map_hash_at(repo, commit, spec.files) == recorded:
            return Result(label, "HISTORY-MATCH", commit[:8])
    return Result(label, "FAIL", "no commit makes these files hash to the recorded value")


def verify(repo: Path) -> Report:
    """Resolve every pin and every canonical map under ``repo``."""
    moved = read_codemoved(repo)
    report = Report()
    for pin in collect_pins(repo):
        report.pins.append(resolve_pin(repo, pin, moved))
    for spec in canonical_maps():
        for json_path in sorted(repo.glob(spec.json_glob)):
            report.maps.append(resolve_map(repo, json_path, spec))
    return report


# --- command line ---------------------------------------------------------------------------


def parse_expect(text: str) -> dict[str, int]:
    """Parse ``MATCH=15,FAIL=0`` into ``{"MATCH": 15, "FAIL": 0}``."""
    out: dict[str, int] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, sep, value = item.partition("=")
        if not sep or key not in STATUSES or not value.strip().isdigit():
            raise ValueError(f"bad --expect item {item!r}; use {'/'.join(STATUSES)}=<count>")
        out[key] = int(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser (building it does no work)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository root (default: here)")
    parser.add_argument("--summary", action="store_true", help="print the one-line summary only")
    parser.add_argument(
        "--expect",
        default="",
        help="exact counts that must hold, e.g. MATCH=15,MOVED-MATCH=0,HISTORY-MATCH=7,FAIL=0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve the pins; exit 0 when they hold, 1 on a failure or mismatch, 2 when shallow."""
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    if is_shallow(repo):
        print("shallow clone: verify_pins needs the full history (git fetch --unshallow)")
        return 2
    try:
        expected = parse_expect(args.expect)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = verify(repo)
    if not args.summary:
        for result in [*report.pins, *report.maps]:
            print(f"{result.status:<14} {result.label}  {result.detail}".rstrip())
    print(report.summary())
    counts = report.counts()
    wrong = {k: (v, counts[k]) for k, v in expected.items() if counts[k] != v}
    for key, (want, got) in wrong.items():
        print(f"expected {key}={want}, got {got}", file=sys.stderr)
    if wrong:
        return 1
    return 1 if counts["FAIL"] and "FAIL" not in expected else 0


if __name__ == "__main__":
    sys.exit(main())
