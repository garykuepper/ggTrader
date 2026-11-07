"""Walk-forward optimisation on the vectorbt engine (spec 5.4).

``run_wfo`` and ``WfoResult`` run a walk forward with one grouped Rust simulation per strategy
configuration instead of three simulations per fold. The policy half (folds, composite score,
gates, anchor, breaker, live-parameter pick, table) is ggTrader glue. Until the P9 cleanup this
package was ``ggTrader.lab.wfo_v2``; the legacy ``lab/wfo.py`` was deleted in P5c.
"""

from __future__ import annotations

from ggTrader.lab.wfo.anchor import AnchorSet, pick_anchor
from ggTrader.lab.wfo.breaker import (
    WfoPolicy,
    WfoState,
    check_circuit_breaker,
    check_shadow_reentry,
    compute_wfe,
)
from ggTrader.lab.wfo.folds import TEST_MONTHS, TRAIN_MONTHS, Fold, generate_folds
from ggTrader.lab.wfo.inputs import combo_name
from ggTrader.lab.wfo.report import WfoResult, format_wfo_table, result_payload
from ggTrader.lab.wfo.run import run_wfo
from ggTrader.lab.wfo.select import composite_score

__all__ = [
    "AnchorSet",
    "Fold",
    "TEST_MONTHS",
    "TRAIN_MONTHS",
    "WfoPolicy",
    "WfoResult",
    "WfoState",
    "check_circuit_breaker",
    "check_shadow_reentry",
    "combo_name",
    "composite_score",
    "compute_wfe",
    "format_wfo_table",
    "generate_folds",
    "pick_anchor",
    "result_payload",
    "run_wfo",
]
