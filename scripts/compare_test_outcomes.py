"""Compare pytest JUnit outcomes against a baseline."""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


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
        reason = (
            ""
            if status is None
            else re.sub(r":\d+(?=[:\s])", "", (status.text or status.get("message", "")).strip())
        )
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
    args = parser.parse_args()
    try:
        baseline, new = load(args.baseline), load(args.new)
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
        if status == "skipped" and baseline.get(node) != ("skipped", reason):
            breaches.append(f"{node}: new or changed skip")
    for node in args.must_pass:
        if new.get(node, ("missing", ""))[0] != "passed":
            breaches.append(f"{node}: must pass")
    print("\n".join(breaches), end="\n" if breaches else "")
    return int(bool(breaches))


if __name__ == "__main__":
    raise SystemExit(main())
