#!/usr/bin/env python3
"""Restore drill for schema ``universe``: dump, restore into a scratch database, verify, drop.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, section 8 and Appendix B11:
the recovery procedure is ``CREATE EXTENSION btree_gist`` on the target, then ``pg_restore`` of the
``pg_dump --schema=universe --format=custom`` file. This script proves it works on the real data
without touching the real database beyond reading it:

1. take a dump with ``ggt db membership export`` (or use ``--dump FILE``);
2. create a bare scratch database ``ggt_restore_drill_<id>`` on the same server;
3. ``CREATE EXTENSION btree_gist`` there, then ``pg_restore`` the dump into it;
4. compare the exact row count of every ``universe`` table with the source, compare every load's
   ``content_sha256``, run ``verify_vintage`` (the load against its own archived bytes) for every
   load id in the restored copy, and check that the append-only guard travelled;
5. drop the scratch database (``--keep`` leaves it for inspection).

No connection string, user or password is ever printed or written; ``pg_restore`` gets its
password through the environment. Exit 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse

import psycopg2

from ggTrader.data.membership import redact
from ggTrader.data.membership_export import EXPORT_DIR, export_universe
from ggTrader.data.universe_db import connect, loader_url, verify_vintage

#: Seconds ``pg_restore`` may take (the schema is a few MB).
RESTORE_TIMEOUT_SECONDS = 600
SCRATCH_PREFIX = "ggt_restore_drill_"


def scratch_url(url: str, name: str) -> str:
    """The same server and login as ``url`` but database ``name``."""
    return urlunparse(urlparse(url)._replace(path=f"/{name}"))


def restore_command(url: str, dump: Path) -> tuple[list[str], dict[str, str]]:
    """``(argv, extra_environment)`` for ``pg_restore`` into the database ``url`` names."""
    parts = urlparse(url)
    dbname = unquote(parts.path.lstrip("/"))
    argv = ["pg_restore", "--no-owner", f"--dbname={dbname}"]
    if parts.hostname:
        argv.append(f"--host={parts.hostname}")
    if parts.port:
        argv.append(f"--port={parts.port}")
    if parts.username:
        argv.append(f"--username={unquote(parts.username)}")
    argv.append(str(dump))
    env = {"PGPASSWORD": unquote(parts.password)} if parts.password else {}
    return argv, env


def table_counts(conn: psycopg2.extensions.connection) -> dict[str, int]:
    """Exact row count of every base table in schema ``universe``."""
    out: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'universe' AND c.relkind = 'r' ORDER BY 1"
        )
        names = [r[0] for r in cur.fetchall()]
        for name in names:
            cur.execute(f'SELECT count(*) FROM universe."{name}"')
            out[name] = int(cur.fetchone()[0])
    conn.rollback()
    return out


def load_hashes(conn: psycopg2.extensions.connection) -> dict[int, tuple[str, str]]:
    """``{load_id: (raw_sha256, content_sha256)}`` for every load."""
    with conn.cursor() as cur:
        cur.execute("SELECT load_id, raw_sha256, content_sha256 FROM universe.loads ORDER BY 1")
        rows = {int(r[0]): (r[1], r[2]) for r in cur.fetchall()}
    conn.rollback()
    return rows


def compare(source: dict, restored: dict) -> list[str]:
    """Human-readable differences between two ``{key: value}`` maps (empty = equal)."""
    problems = []
    for key in sorted(set(source) | set(restored)):
        if source.get(key) != restored.get(key):
            problems.append(f"{key}: source {source.get(key)!r} vs restored {restored.get(key)!r}")
    return problems


def guard_travelled(conn: psycopg2.extensions.connection) -> bool:
    """True if a DELETE on ``universe.loads`` is refused by the restored append-only trigger."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM universe.loads "
                "WHERE load_id = (SELECT min(load_id) FROM universe.loads)"
            )
    except psycopg2.Error as exc:
        conn.rollback()
        return "append-only" in str(exc)
    conn.rollback()
    return False


def drill(dump: Path | None, out_dir: Path, keep: bool) -> dict:
    """Run the drill and return the report (``report['ok']`` is the overall verdict)."""
    source_url = loader_url()
    report: dict = {"steps": [], "ok": False}

    def step(name: str, ok: bool, detail: str) -> bool:
        report["steps"].append({"step": name, "ok": ok, "detail": detail})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    if dump is None:
        result = export_universe(source_url, out_dir=out_dir)
        dump = result.path
        step("dump", True, f"{dump.name} {result.size_bytes} bytes sha256 {result.sha256}")
    else:
        step("dump", dump.is_file(), f"using {dump.name}")
    src = connect(source_url, read_only=True)
    try:
        source_counts, source_hashes = table_counts(src), load_hashes(src)
    finally:
        src.close()
    report["source_counts"] = source_counts

    name = f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(source_url)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}"')
    step("scratch database", True, f"created {name}")
    target = scratch_url(source_url, name)
    try:
        bare = psycopg2.connect(target)
        bare.autocommit = True
        with bare.cursor() as cur:
            cur.execute("CREATE EXTENSION btree_gist")
        bare.close()
        step("extension", True, "CREATE EXTENSION btree_gist")
        argv, extra = restore_command(target, dump)
        done = subprocess.run(
            argv,
            env={**os.environ, **extra},
            capture_output=True,
            text=True,
            check=False,
            timeout=RESTORE_TIMEOUT_SECONDS,
        )
        errors = [ln for ln in (done.stderr or "").splitlines() if ln.strip()]
        shown = redact(" | ".join(errors[:3]), source_url) if errors else "no messages"
        if not step("pg_restore", done.returncode == 0, f"exit {done.returncode}; {shown}"):
            return report
        restored = connect(target)
        try:
            restored_counts = table_counts(restored)
            problems = compare(source_counts, restored_counts)
            step(
                "row counts",
                not problems,
                f"{sum(restored_counts.values())} rows in {len(restored_counts)} tables, "
                + ("all equal" if not problems else "; ".join(problems)),
            )
            problems = compare(source_hashes, load_hashes(restored))
            step(
                "load hashes",
                not problems,
                f"{len(source_hashes)} loads: raw and content sha256 equal"
                if not problems
                else "; ".join(problems),
            )
            for load_id in sorted(source_hashes):
                verdict = verify_vintage(restored, load_id, None)
                bad = [c.name for c in verdict.checks if not c.ok]
                step(
                    f"verify --load-id {load_id}",
                    verdict.ok,
                    "OK" if verdict.ok else f"failed {bad}",
                )
            restored.rollback()
            step("append-only guard", guard_travelled(restored), "DELETE on universe.loads refused")
        finally:
            restored.close()
    finally:
        if keep:
            step("drop", True, f"kept {name} (drop it by hand)")
        else:
            with admin.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            step("drop", True, f"dropped {name}")
        admin.close()
    report["ok"] = all(s["ok"] for s in report["steps"])
    return report


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dump", type=Path, default=None, help="Restore this dump instead of taking a new one"
    )
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR, help="Where a new dump goes")
    parser.add_argument("--keep", action="store_true", help="Do not drop the scratch database")
    parser.add_argument("--json", type=Path, default=None, help="Also save the report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the drill; exit 0 only if every step passed."""
    args = build_parser().parse_args(argv)
    print("restore drill of schema universe")
    try:
        report = drill(args.dump, args.out_dir, args.keep)
    except Exception as exc:  # noqa: BLE001 - never echo a connection string
        first = (str(exc).strip().splitlines() or [""])[0]
        print(f"drill failed: {type(exc).__name__}: {redact(first, None)}", file=sys.stderr)
        return 1
    print("restore drill:", "PASSED" if report["ok"] else "FAILED")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
