import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

REF = {
    "xs_momentum": "results/monthly_wf/sp500_xs_momentum",
    "dual_momentum": "results/monthly_wf/sp500_dual_momentum",
}


def _old_selections(run_dir: str):
    out = {}
    for d in sorted(Path(run_dir).glob("month=*")):
        doc = json.loads((d / "selections.json").read_text())
        asof = pd.Timestamp(doc["asof"], tz="UTC")
        out[asof] = [(s["symbol"], round(float(s["weight"]), 10)) for s in doc["selections"]]
    return out


@pytest.mark.parametrize("strategy", ["xs_momentum", "dual_momentum"])
def test_lab_reproduces_old_selections(strategy):
    ref_dir = REF[strategy]
    if not Path(ref_dir).exists():
        pytest.skip(f"reference run {ref_dir} not present")
    old = _old_selections(ref_dir)

    from ggTrader.lab.data import eligible_at, equity_universe_between, load_ohlcv
    from ggTrader.lab.strategies.momentum import build_strategy
    from ggTrader.lab.strategy import LabConfig

    cfg = LabConfig(top_n=50, lookback=252, skip=21, min_history_bars=400)
    strat = build_strategy(strategy, cfg)
    eval_start = pd.Timestamp("2021-01-31", tz="UTC")
    eval_end = pd.Timestamp("2026-05-31", tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end)
    ohlcv = load_ohlcv(universe, str(data_start.date()), str(eval_end.date()))

    mismatches = []
    for asof in sorted(old):
        past = ohlcv.loc[:asof]
        eligible, _ = eligible_at(asof, past, cfg)
        new = [
            (s["symbol"], round(float(s["weight"]), 10)) for s in strat.select(asof, past, eligible)
        ]
        if new != old[asof]:
            mismatches.append(str(asof.date()))
    assert not mismatches, f"selection mismatches at: {mismatches[:5]} ({len(mismatches)} total)"
