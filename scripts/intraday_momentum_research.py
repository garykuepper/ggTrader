"""Approved r8 intraday SPY screen: print the frozen manifest, collect, validate or score.

Collection and scoring of real data require the owner's separate approval 2. With no command, or
with --help, this prints usage only: no credentials, network or writes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_COLLECTION = PROJECT_ROOT / "data/external/intraday_momentum_r8"
DEFAULT_VALIDATION = PROJECT_ROOT / "docs/research/_intraday_momentum_validation.json"
DEFAULT_RESULT = PROJECT_ROOT / "docs/research/_intraday_momentum_results.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser; parsing never imports the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Print the frozen request manifest (no network, no writes)")
    collect = commands.add_parser(
        "collect", help="Collect or resume the frozen manifest (needs approval 2 and credentials)"
    )
    collect.add_argument("--output-dir", type=Path, required=True, help="Collection directory")
    validate = commands.add_parser("validate", help="Signal-blind validation, written once")
    validate.add_argument("--input-dir", type=Path, default=DEFAULT_COLLECTION)
    validate.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    score = commands.add_parser("score", help="Single automatic score on a matching PASS")
    score.add_argument("--input-dir", type=Path, default=DEFAULT_COLLECTION)
    score.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    score.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one explicit command; errors are reported without bodies or secrets."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from ggTrader.lab import intraday_momentum as core

    try:
        if args.command == "plan":
            print(json.dumps(core.build_manifest(), indent=2, sort_keys=True))
            return 0
        if args.command == "collect":
            from ggTrader.lab.intraday_momentum_data import run_collection

            result = run_collection(args.output_dir)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "complete" else 1
        if args.command == "validate":
            result = core.run_validation(args.input_dir, args.output)
            print(result["label"])
            return 0 if result["status"] == "PASS" else 1
        result = core.run_score(args.input_dir, args.validation, args.output)
        print(result["verdict"])
        return 0
    except (FileExistsError, ValueError) as error:
        # Only the pipeline's own short reason codes are echoed; anything else is just its type.
        reason = str(error) if re.fullmatch(r"[a-z0-9_:.-]{1,80}", str(error)) else "error"
        print(f"refused: {type(error).__name__}:{reason}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
