"""AST import graph of ggt.py, src/, scripts/ and tests/ (P5c dependency-closure check).

Usage:
  import_graph.py graph [--json OUT]                 write the file-level import graph
  import_graph.py importers FILE_OR_MODULE ...       who imports these (direct)
  import_graph.py closure --deleted LIST.txt         files that import a deleted file (must be empty
                                                     once every deleted file's importers are gone)
Every import counts, at any depth (function-local included). String references to modules
(importlib.import_module("x"), "ggTrader.x" literals passed to patch/monkeypatch) are reported too.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCAN = ("src", "scripts", "tests")
EXTRA = ("ggt.py",)


def py_files() -> list[Path]:
    files = [ROOT / e for e in EXTRA if (ROOT / e).exists()]
    for d in SCAN:
        files += sorted(p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    return files


def module_name(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    if rel.parts[0] == "src":
        parts = list(rel.with_suffix("").parts[1:])
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    return None


def build() -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    files = py_files()
    mod2file = {}
    for f in files:
        m = module_name(f)
        if m:
            mod2file[m] = str(f.relative_to(ROOT))
    # script/test modules importable by basename (sys.path hacks, tests importing scripts)
    base2file: dict[str, str] = {}
    for f in files:
        rel = str(f.relative_to(ROOT))
        if rel.startswith(("scripts/", "tests/")) or rel == "ggt.py":
            base2file.setdefault(f.stem, rel)
    edges: dict[str, set[str]] = {}
    strrefs: dict[str, set[str]] = {}
    for f in files:
        rel = str(f.relative_to(ROOT))
        edges[rel] = set()
        strrefs[rel] = set()
        try:
            tree = ast.parse(f.read_text(), filename=rel)
        except SyntaxError:
            continue
        pkg = module_name(f)
        is_init = f.name == "__init__.py"
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    cur = (pkg or "").split(".")
                    if not is_init:
                        cur = cur[:-1]
                    cur = cur[: len(cur) - (node.level - 1)] if node.level > 1 else cur
                    base = ".".join(cur + ([base] if base else []))
                targets = [base] + [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if s.startswith("ggTrader.") and " " not in s:
                    strrefs[rel].add(s)
            for t in targets:
                if t in mod2file:
                    edges[rel].add(mod2file[t])
                    # importing a.b.c also executes a/__init__ and a.b/__init__
                    parts = t.split(".")
                    for i in range(1, len(parts)):
                        p = ".".join(parts[:i])
                        if p in mod2file:
                            edges[rel].add(mod2file[p])
                elif t in base2file and not t.startswith("ggTrader"):
                    edges[rel].add(base2file[t])
        for s in list(strrefs[rel]):
            # resolve to the longest existing module prefix
            parts = s.split(".")
            for i in range(len(parts), 0, -1):
                m = ".".join(parts[:i])
                if m in mod2file:
                    edges[rel].add(mod2file[m])
                    break
    edges = {k: {v for v in vs if v != k} for k, vs in edges.items()}
    return mod2file, edges, strrefs


def importers(edges: dict[str, set[str]], target: str) -> list[str]:
    return sorted(k for k, vs in edges.items() if target in vs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graph")
    g.add_argument("--json")
    i = sub.add_parser("importers")
    i.add_argument("names", nargs="+")
    c = sub.add_parser("closure")
    c.add_argument("--deleted", required=True, help="text file, one repo-relative path per line")
    a = ap.parse_args()
    mod2file, edges, _ = build()
    if a.cmd == "graph":
        out = {k: sorted(v) for k, v in sorted(edges.items())}
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"{len(out)} files, {sum(len(v) for v in out.values())} edges")
    elif a.cmd == "importers":
        for n in a.names:
            t = mod2file.get(n, n)
            print(f"{t}: {importers(edges, t)}")
    else:
        deleted = {ln.strip() for ln in Path(a.deleted).read_text().splitlines() if ln.strip()}
        bad = {}
        for d in sorted(deleted):
            left = [x for x in importers(edges, d) if x not in deleted]
            if left:
                bad[d] = left
        for d, l in bad.items():
            print(f"{d} <- {l}")
        print(f"{len(bad)} deleted files still imported by kept files")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
