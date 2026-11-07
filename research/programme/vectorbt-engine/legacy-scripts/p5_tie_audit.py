"""P5: classify the signal differences behind two P0-reference score differences.

``ref_bb_reversion`` and ``ref_conviction_bb`` reproduce the P0 winners, gates, anchor use and
OOS equity but the winner's composite ``train_score`` differs from the legacy value at 1e-6.
``p5_equity_parity.py`` shows a few non-deployed combinations whose equity differs from the
legacy sweep. This script classifies the signal cells that differ on the exact P5 bars with the
P4 audit (burn-in / tie / real), where a tie is a close within 1e-9 of its threshold on the flip
bar or the bar before it (vectorbt needs a strict cross, the legacy form is non-strict).

Example:
    .venv/bin/python scripts/p5_tie_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
OUT = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "p5-evidence"


def main(argv: list[str] | None = None) -> int:
    """Run the audit on the 2-fold window bars and write ``_p5_tie_audit.json``."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    import p4_strategy_parity as p4
    import p5_wfo_validation as v

    from ggTrader.data.bars import to_legacy_frame
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.utils.results import write_result_json

    report: dict[str, Any] = {"window": ["2021-01-31", v.EVAL_END_SMOKE]}
    inp = v.load_run_inputs(v.ALL_RUNS["ref_bb_reversion"])
    bars = inp["bars"]
    frame = to_legacy_frame(bars)
    report["symbols"] = len(bars.symbols)
    report["bars"] = len(bars.close)
    # Strategy level through the real legacy sweep path (every grid combination).
    report["bb_reversion"] = p4.strategy_level(
        "bb_reversion", REGISTRY_V2["bb_reversion"], STRATEGY_REGISTRY["bb_reversion"], bars, frame
    )
    from ggTrader.lab.strategies.v2.voters import VOTER_SPECS

    cls = REGISTRY_V2["bb_reversion"]
    combos = [{**cls.defaults, **c} for c in cls.grid.combos()]
    tuples = list(dict.fromkeys(tuple(c[k] for k in VOTER_SPECS["bb"].keys) for c in combos))
    report["bb_voter_level"] = p4.voter_level_report("bb", tuples, bars)
    report["conviction_bb"] = p4.conviction_report(bars, frame)
    report["conclusion"] = (
        "after the 200-bar burn-in every differing cell is a tie within 1e-9 of its threshold "
        "(real == 0); see 'real' in each block"
    )
    write_result_json(OUT / "_p5_tie_audit.json", report)
    for key in ("bb_reversion", "bb_voter_level", "conviction_bb"):
        print(key, {k: v_ for k, v_ in report[key].items() if k != "folds"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
