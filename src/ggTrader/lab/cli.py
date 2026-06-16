"""CLI for the lab research bench: run a strategy over the equity universe."""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.harness import walkforward
from ggTrader.lab.strategies.momentum import STRATEGY_NAMES, build_strategy
from ggTrader.lab.strategies.signals import SIGNAL_STRATEGY_NAMES, build_signal_strategy
from ggTrader.lab.strategy import LabConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a lab strategy walk-forward.")
    p.add_argument(
        "--strategy",
        choices=tuple(STRATEGY_NAMES) + tuple(SIGNAL_STRATEGY_NAMES),
        required=True,
    )
    p.add_argument("--market", default="equity")
    p.add_argument("--eval-start", default="2021-01-31")
    p.add_argument("--eval-end", default=None)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--max-stocks", type=int, default=None)
    return p


def run_lab(argv: List[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    cfg = LabConfig(
        top_n=args.top_n, lookback=args.lookback, skip=args.skip, max_stocks=args.max_stocks
    )
    if args.strategy in SIGNAL_STRATEGY_NAMES:
        strat = build_signal_strategy(args.strategy, cfg)
    else:
        strat = build_strategy(args.strategy, cfg)

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = (
        pd.Timestamp(args.eval_end, tz="UTC")
        if args.eval_end
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    # Window must cover the eligibility requirement (min_history_bars), not just
    # the momentum lookback — else the first selection dates are starved of history.
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end)
    ohlcv = load_ohlcv(universe + ["SPY"], str(data_start.date()), str(eval_end.date()))
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    run_id = walkforward(
        [strat],
        ohlcv,
        spy_close,
        eval_start=str(eval_start.date()),
        eval_end=str(eval_end.date()),
        market=args.market,
        freq="monthly",
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg)[0],
        base_config=dict(STOCK_BASE_CONFIG),
    )
    print(f"lab run complete: {run_id}")
    return run_id


if __name__ == "__main__":
    run_lab()
