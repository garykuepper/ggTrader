"""Compare frozen evidence folders without volatile run metadata."""

import argparse
import json
from pathlib import Path
from typing import Any

# src/ggTrader/engine/provenance.py:141-166
VOLATILE = {("provenance", "created_utc"), ("provenance", "git_sha"), ("provenance", "git_dirty")}
# scripts/p5_wfo_validation.py:322
VOLATILE |= {("wall_seconds",), ("diagnostics", "peak_rss_mb"), ("diagnostics", "timings_s")}
# scripts/p7_live_golden.py:127 (elapsed seconds of the run)
VOLATILE.add(("seconds",))
# scripts/p5_wfo_validation.py:322
P5_VOLATILE = ("peak_rss_mb", "new")


def compare(before: Path, after: Path) -> list[str]:
    """Return differences between two evidence trees."""
    differences: list[str] = []
    files = []
    for folder in (before, after):
        entries = (
            {p.relative_to(folder) for p in folder.rglob("*") if p.is_file()}
            if folder.is_dir()
            else set()
        )
        if not entries:
            differences.append(f"{folder}: missing or empty folder")
        files.append(entries)

    def walk(a: Any, b: Any, path: tuple, name: Path) -> None:
        if path in VOLATILE or (name.name.startswith("_p5_compare_") and path == P5_VOLATILE):
            return
        label = "".join(
            f"[{x}]" if isinstance(x, int) else ("." if i else "") + x for i, x in enumerate(path)
        )
        if type(a) is not type(b):
            differences.append(f"{name}: {label}")
        elif isinstance(a, dict):
            for key in sorted(a.keys() | b.keys()):
                child = (*path, key)
                if child in VOLATILE or (
                    name.name.startswith("_p5_compare_") and child == P5_VOLATILE
                ):
                    continue
                if key not in a or key not in b:
                    differences.append(f"{name}: " + ".".join(map(str, child)))
                else:
                    walk(a[key], b[key], child, name)
        elif isinstance(a, list):
            for i in range(max(len(a), len(b))):
                if i >= len(a) or i >= len(b):
                    differences.append(f"{name}: {label}[{i}]")
                else:
                    walk(a[i], b[i], (*path, i), name)
        elif a != b:
            differences.append(f"{name}: {label}")

    def failed(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("ok") is False or any(failed(v) for v in value.values())
        return isinstance(value, list) and any(failed(v) for v in value)

    for name in sorted(files[0] | files[1]):
        if name not in files[0] or name not in files[1]:
            differences.append(f"{name}: missing file")
            continue
        if name.suffix == ".json":
            try:
                a, b = (json.loads((folder / name).read_text()) for folder in (before, after))
            except (ValueError, OSError) as exc:
                differences.append(f"{name}: {exc}")
                continue
            if failed(a) or failed(b):
                differences.append(f"{name}: ok is false")
            walk(a, b, (), name)
        elif (before / name).read_bytes() != (after / name).read_bytes():
            differences.append(f"{name}: bytes differ")
    return differences


def main() -> int:
    """Parse arguments and print evidence differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    differences = compare(args.before, args.after)
    print("\n".join(differences), end="\n" if differences else "")
    return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
