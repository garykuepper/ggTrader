"""Find return-matched identities using approved calibration settings."""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ggTrader.data.kaggle_import.link_inputs import crsp_data, yahoo_data
from ggTrader.data.kaggle_import.links import LinkSettings, find_links
from ggTrader.data.kaggle_import.spells import calendar_from_file


def settings_from_dict(values: dict) -> LinkSettings:
    """Read approved settings, allowing a JSON file to override defaults."""
    defaults = dict(
        min_sessions=250,
        min_sessions_short=40,
        agreement_bar=0.999,
        margin=0.0005,
        window_start="2005-01-01",
        window_end="2024-12-31",
        hold_first_days=60,
    )
    defaults.update(values)
    for name in ("window_start", "window_end"):
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
        run(args, settings_from_dict(values)).to_csv(args.out, index=False)
    except Exception:
        print("Link finding failed; check inputs and database access.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
