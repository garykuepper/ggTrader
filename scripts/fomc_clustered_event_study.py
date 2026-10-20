"""Day-clustered FOMC drift event study on the treasury ETF universe.

Phase 0 of momentum re-test (Task 1):
Clusters returns by date (one observation per date) rather than symbol-days
to avoid multi-counting dates across TLT, IEF, and EDV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_reruns_wfo import (  # noqa: E402
    FOMC_EVAL_END,
    FOMC_EVAL_START,
    FOMC_ROUNDTRIP_COST,
    FOMC_UNIVERSE,
    FOMC_WARMUP_DAYS,
)

from ggTrader.lab.data import equity_universe_between, load_ohlcv  # noqa: E402
from ggTrader.lab.event_study import (  # noqa: E402
    event_study_test,
    in_position_mask,
    split_event_returns_by_date,
)
from ggTrader.lab.strategies.fomc_drift import (  # noqa: E402
    FomcDriftStrategy,
    _event_signals,
)
from ggTrader.lab.strategies.indicators import extract_close  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    default_out = REPO_ROOT / "docs" / "research" / "_fomc_clustered_event_study_20261020.json"
    parser = argparse.ArgumentParser(
        description="Day-clustered FOMC drift event study on the treasury ETF universe."
    )
    parser.add_argument(
        "--out",
        "--output",
        type=Path,
        default=default_out,
        help=f"Path to output JSON file (default: {default_out}).",
    )
    return parser.parse_args(argv)


def run_clustered_study() -> dict[str, Any]:
    """Run the day-clustered FOMC drift event study."""
    cfg = LabConfig()
    es = pd.Timestamp(FOMC_EVAL_START, tz="UTC")
    ee = pd.Timestamp(FOMC_EVAL_END, tz="UTC")
    data_start = str((es - pd.Timedelta(days=FOMC_WARMUP_DAYS)).date())

    universe = equity_universe_between(es, ee, universe=FOMC_UNIVERSE)
    ohlcv = load_ohlcv(universe + ["SPY"], data_start, FOMC_EVAL_END, use_negative_cache=True)
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv_stocks = ohlcv[sym_cols]

    symbols = sorted(
        [s for s in universe if s in ohlcv_stocks.columns.get_level_values(0).unique()]
    )
    close = extract_close(ohlcv_stocks, symbols)
    strat = FomcDriftStrategy(cfg)
    events = strat._load_events()

    eval_start_d = pd.Timestamp(FOMC_EVAL_START).date()
    eval_end_d = pd.Timestamp(FOMC_EVAL_END).date()
    if hasattr(close.index, "tz") and close.index.tz is not None:
        idx_dates = close.index.tz_convert("UTC").date
    else:
        idx_dates = pd.to_datetime(close.index).date
    eval_mask = (idx_dates >= eval_start_d) & (idx_dates <= eval_end_d)
    close_eval = close.loc[eval_mask]

    # Primary pair only: entry_offset_days=1, hold_days=0
    entry_offset = 1
    hold = 0
    pair_name = f"entry_{entry_offset}_hold_{hold}"

    signals = _event_signals(close_eval.index, symbols, events, entry_offset, hold)
    mask = in_position_mask(signals.entries, signals.exits)
    returns_eval = close_eval.pct_change()

    # Gross: split into daily mean series and test
    event_gross, nonevent_gross = split_event_returns_by_date(returns_eval, mask)
    gross_res = event_study_test(event_gross, nonevent_gross)

    # Net: subtract FOMC_ROUNDTRIP_COST from first held day before averaging by date
    first_held_mask = mask & (~mask.shift(1, fill_value=False))
    net_returns = returns_eval.copy()
    net_returns[first_held_mask] = net_returns[first_held_mask] - FOMC_ROUNDTRIP_COST
    event_net, nonevent_net = split_event_returns_by_date(net_returns, mask)
    net_res = event_study_test(event_net, nonevent_net)

    ref_path = REPO_ROOT / "docs" / "research" / "_audit_reruns_20261019.json"
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_json = json.load(f)
    unclustered_ref = ref_json["fomc_drift"]["primary_event_study"]

    return {
        "universe": FOMC_UNIVERSE,
        "eval_start": FOMC_EVAL_START,
        "eval_end": FOMC_EVAL_END,
        "pair": pair_name,
        "n_event_dates": gross_res.n_event,
        "gross": gross_res._asdict(),
        "net": net_res._asdict(),
        "unclustered_reference": unclustered_ref,
    }


def main(argv: list[str] | None = None) -> int:
    """Run entrypoint."""
    args = parse_args(argv)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_clustered_study()

    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    tmp_path.replace(out_path)

    print(f"Wrote clustered event study to {out_path}")
    print(f"  n_event_dates: {result['n_event_dates']}")
    print(f"  gross: t={result['gross']['t_stat']:.3f}, p={result['gross']['p_value']:.4f}")
    print(f"  net:   t={result['net']['t_stat']:.3f}, p={result['net']['p_value']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
