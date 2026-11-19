"""Compare pytest JUnit outcomes against a baseline."""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _normalize(reason: str) -> str:
    """Strip line numbers and the checkout's absolute path, which differ between worktrees."""
    reason = re.sub(r"(?<![\w.])/[^\s:]*?/(?=(?:tests|research|src)/)", "", reason.strip())
    # Only a leading "file.py:LINE" is a line number; other numbers (such as "12:30") are content.
    return re.sub(r"^(\S+\.py):\d+(?=:)", r"\1", reason)


def outcomes(path: Path) -> dict[str, tuple[str, str]]:
    """Read test outcomes keyed by pytest node ID."""
    result = {}
    for case in ET.parse(path).iter("testcase"):
        parts = case.get("classname", "").split(".")
        filename = (
            "/".join(
                parts[
                    : next(
                        (i + 1 for i, p in enumerate(parts) if p.startswith("test_")), len(parts)
                    )
                ]
            )
            + ".py"
        )
        classes = parts[filename.count("/") + 1 :]
        node = "::".join([filename, *classes, case.get("name", "")])
        status = next(
            (child for child in case if child.tag in {"failure", "error", "skipped"}), None
        )
        reason = "" if status is None else _normalize(status.text or status.get("message", ""))
        if node in result:
            raise ValueError(f"duplicate node: {node}")
        result[node] = ("passed" if status is None else status.tag, reason)
    return result


def load(spec: str) -> dict[str, tuple[str, str]]:
    """Read one or more comma-separated JUnit files and merge their outcomes.

    `run_tests.sh --full` runs two pytest processes (tests/ and the study folders), and each
    one overwrites the same --junitxml path, so a gate keeps one file per half.
    """
    merged: dict[str, tuple[str, str]] = {}
    for part in spec.split(","):
        for node, outcome in outcomes(Path(part)).items():
            if node in merged:
                raise ValueError(f"duplicate node across files: {node}")
            merged[node] = outcome
    return merged


def main() -> int:
    """Parse arguments and report outcome breaches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="JUnit XML file, or several separated by commas")
    parser.add_argument("new", help="JUnit XML file, or several separated by commas")
    parser.add_argument("--must-pass", nargs="+", default=[])
    parser.add_argument("--must-pass-file", type=Path)
    args = parser.parse_args()
    try:
        baseline, new = load(args.baseline), load(args.new)
        if args.must_pass_file:
            args.must_pass.extend(
                line.strip()
                for line in args.must_pass_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
    except (OSError, ET.ParseError, ValueError) as exc:
        print(exc)
        return 1
    breaches = []
    if len(new) < len(baseline):
        breaches.append(f"new has fewer nodes: {len(new)} < {len(baseline)}")
    for node, (status, _) in baseline.items():
        if node not in new:
            breaches.append(f"{node}: missing from new")
        elif status == "passed" and new[node][0] != "passed":
            breaches.append(f"{node}: baseline pass no longer passes")
    for node, (status, reason) in new.items():
        old = baseline.get(node)
        # Anything but a pass must be exactly what the baseline had: a new skip, a new test that
        # fails, and an xfail that now fails (XPASS strict) are all breaches. A node that passed
        # in the baseline is already reported above.
        if status != "passed" and old != (status, reason) and not (old and old[0] == "passed"):
            breaches.append(f"{node}: new or changed {status}")
    for node in args.must_pass:
        if new.get(node, ("missing", ""))[0] != "passed":
            breaches.append(f"{node}: must pass")
    print("\n".join(breaches), end="\n" if breaches else "")
    return int(bool(breaches))


if __name__ == "__main__":
    raise SystemExit(main())
