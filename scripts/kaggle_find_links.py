"""Find return-matched identities using approved calibration settings."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ggTrader.data.kaggle_import.guards import safe_error as _safe_error
from ggTrader.data.kaggle_import.link_inputs import crsp_data, yahoo_data
from ggTrader.data.kaggle_import.link_manifest import file_sha256, settings_dict
from ggTrader.data.kaggle_import.links import APPROVED_LINK_SETTINGS, LinkSettings, find_links
from ggTrader.data.kaggle_import.spells import calendar_from_file
from ggTrader.data.universe_db import git_sha


def settings_from_dict(values: dict) -> LinkSettings:
    """Read approved settings, allowing a JSON file to override defaults."""
    defaults = asdict(APPROVED_LINK_SETTINGS)
    defaults.update(values)
    for name in ("window_start", "window_end"):
        if isinstance(defaults[name], str):
            defaults[name] = date.fromisoformat(defaults[name])
    return LinkSettings(**defaults)


def run(args: argparse.Namespace, settings: LinkSettings) -> pd.DataFrame:
    """Read input data and return links with Yahoo lifetimes."""
    y, yf, yl = yahoo_data(
        os.environ[args.yahoo_url_env],
        pd.Timestamp(settings.window_start),
        pd.Timestamp(settings.window_end),
    )
    c, cf, cl = crsp_data(args.crsp_file, y.index, args.sample_permnos, settings.min_sessions_short)
    result = find_links(y, c, yf, cf, yl, cl, settings, calendar_from_file(args.crsp_file))
    result["first_yf_date"] = result.symbol.map(yf)
    result["last_yf_date"] = result.symbol.map(yl)
    symbols = sorted(str(symbol) for symbol in y.columns)
    result.attrs["yahoo_fingerprint"] = {
        "n_symbols": len(symbols),
        "newest_timestamp": pd.Timestamp(yl.max()).isoformat() if len(yl) else None,
        "symbols_sha256": hashlib.sha256("\n".join(symbols).encode()).hexdigest(),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse inputs and write the link table without disclosing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crsp-file", type=Path, required=True)
    parser.add_argument("--mcgill-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-start")
    parser.add_argument("--window-end")
    parser.add_argument("--settings-json", type=Path)
    parser.add_argument("--yahoo-url-env", default="GGT_YAHOO_URL")
    parser.add_argument("--sample-permnos", type=int)
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.settings_json.read_text()) if args.settings_json else {}
        for name in ("window_start", "window_end"):
            if getattr(args, name):
                values[name] = getattr(args, name)
        settings = settings_from_dict(values)
        result = run(args, settings)
        result.to_csv(args.out, index=False)
        manifest = {
            "settings": settings_dict(settings),
            "csv_sha256": file_sha256(args.out),
            "crsp_file_sha256": file_sha256(args.crsp_file),
            "sample_permnos": args.sample_permnos,
            "yahoo_fingerprint": result.attrs["yahoo_fingerprint"],
            "git_sha": git_sha(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        args.out.with_name(args.out.name + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    except Exception as exc:
        print(f"Link finding failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
