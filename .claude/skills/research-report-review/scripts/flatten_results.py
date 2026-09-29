"""Print every numeric leaf of a research results JSON, filtered by keyword.

Use it to check a report's or summary's quoted numbers against the raw
artifact: `flatten_results.py docs/research/_x_results.json sharpe max_dd`.
"""

import argparse
import json
from pathlib import Path


def leaves(node, path=""):
    """Yield (path, value) for every int/float leaf, depth-first."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("json_path", type=Path)
    parser.add_argument("keywords", nargs="*", help="case-insensitive path filters (any match)")
    args = parser.parse_args()
    data = json.loads(args.json_path.read_text())
    keys = [k.lower() for k in args.keywords]
    for path, value in leaves(data):
        if not keys or any(k in path.lower() for k in keys):
            print(f"{path}\t{round(value, 4)}")


if __name__ == "__main__":
    main()
