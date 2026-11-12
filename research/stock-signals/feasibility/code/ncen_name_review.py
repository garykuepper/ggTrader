"""Flag issuer-name differences in retrospective N-CEN ticker leads."""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from ggTrader.utils.manifest import sha256_file


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).lower())


def summarize(rows: list[dict]) -> dict:
    """Group exact-label leads by simple name agreement; do not infer identity."""
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        symbol, name, registrant = row["symbol"], row["name"], row["REGISTRANT_NAME"]
        if not all(isinstance(x, str) and x.strip() for x in (symbol, name, registrant)):
            raise ValueError("Blank or non-string issuer metadata")
        grouped[symbol]["screener"].add(name)
        grouped[symbol]["registrant"].add(registrant)
    matching, review = [], []
    for symbol, names in sorted(grouped.items()):
        screener = {_normalized(x.removesuffix(" Common Stock")) for x in names["screener"]}
        registrants = {_normalized(x) for x in names["registrant"]}
        if len(screener) == 1 and screener == registrants:
            matching.append(symbol)
        else:
            review.append(
                {"symbol": symbol, **{key: sorted(value) for key, value in names.items()}}
            )
    return {"matching_labels": matching, "review": review}


def main() -> None:
    """Read a pinned audit and write a new name-review record."""
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("input", "sha256", "out"):
        parser.add_argument(f"--{argument}", required=True)
    args = parser.parse_args()
    if sha256_file(Path(args.input)) != args.sha256:
        raise ValueError("Input SHA-256 mismatch")
    with open(args.input, encoding="utf-8") as handle:
        rows = json.load(handle)["matches"]
    result = {
        "purpose": "Retrospective name review; neither outcome proves identity.",
        "method": (
            "Remove exact ' Common Stock' suffix, NFKD-normalize, lowercase, "
            "retain ASCII letters and digits; no alias guessing."
        ),
        "source_sha256": args.sha256,
        "driver_sha256": sha256_file(Path(__file__)),
        **summarize(rows),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(f"{len(result['matching_labels'])} agreement; {len(result['review'])} review")


if __name__ == "__main__":
    main()
