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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run parameter sweep instead of single walk-forward.",
    )
    mode.add_argument(
        "--wfo",
        action="store_true",
        default=False,
        help="Walk-forward optimization: rolling train/test folds with OOS scoring.",
    )
    p.add_argument(
        "--sweep-param",
        action="append",
        default=[],
        help="Override sweep range: --sweep-param ema_fast=5,10,20",
    )
    return p


def run_lab(argv: List[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    cfg = LabConfig(
        top_n=args.top_n, lookback=args.lookback, skip=args.skip, max_stocks=args.max_stocks
    )

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

    if args.sweep:
        from ggTrader.lab.strategies.ensemble import EnsembleSignal
        from ggTrader.lab.strategies.momentum import CrossSectionalMomentum, DualMomentum
        from ggTrader.lab.strategies.signals import (
            BollingerReversionSignal,
            EmaCrossSignal,
            RsiReversionSignal,
            WfoTournamentSignal,
        )
        from ggTrader.lab.sweep import build_grid, run_sweep

        cls_map = {
            "ema_cross": EmaCrossSignal,
            "wfo_tournament": WfoTournamentSignal,
            "bb_reversion": BollingerReversionSignal,
            "rsi_reversion": RsiReversionSignal,
            "ensemble": EnsembleSignal,
            "xs_momentum": CrossSectionalMomentum,
            "dual_momentum": DualMomentum,
        }
        strategy_cls = cls_map[args.strategy]
        overrides = _parse_sweep_params(args.sweep_param)
        grid = build_grid(strategy_cls, overrides=overrides if overrides else None)
        print(f"Sweep: {args.strategy} | {len(grid)} param combos")
        return run_sweep(
            args.strategy,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=str(eval_start.date()),
            eval_end=str(eval_end.date()),
            market=args.market,
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )

    if args.wfo:
        from ggTrader.lab.strategies.ensemble import EnsembleSignal as _WfoEnsemble
        from ggTrader.lab.strategies.signals import (
            SIGNAL_STRATEGY_NAMES as _SIG_NAMES,
        )
        from ggTrader.lab.strategies.signals import (
            BollingerReversionSignal,
            EmaCrossSignal,
            RsiReversionSignal,
            WfoTournamentSignal,
        )
        from ggTrader.lab.sweep import build_grid
        from ggTrader.lab.wfo import run_wfo

        if args.strategy not in _SIG_NAMES:
            raise SystemExit(f"--wfo only supports signal strategies: {_SIG_NAMES}")
        cls_map = {
            "ema_cross": EmaCrossSignal,
            "wfo_tournament": WfoTournamentSignal,
            "bb_reversion": BollingerReversionSignal,
            "rsi_reversion": RsiReversionSignal,
            "ensemble": _WfoEnsemble,
        }
        strategy_cls = cls_map[args.strategy]
        overrides = _parse_sweep_params(args.sweep_param)
        grid = build_grid(strategy_cls, overrides=overrides if overrides else None)
        print(f"WFO: {args.strategy} | {len(grid)} param combos")
        result = run_wfo(
            args.strategy,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=str(eval_start.date()),
            eval_end=str(eval_end.date()),
            market=args.market,
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )
        return result

    if args.strategy in SIGNAL_STRATEGY_NAMES:
        strat = build_signal_strategy(args.strategy, cfg)
    else:
        strat = build_strategy(args.strategy, cfg)

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


def _parse_sweep_params(raw: List[str]) -> dict[str, list]:
    """Parse CLI '--sweep-param key=v1,v2,v3' into {key: [v1, v2, v3]}."""
    result: dict[str, list] = {}
    for item in raw:
        key, _, vals = item.partition("=")
        if not key or not vals:
            raise ValueError(f"Invalid --sweep-param: {item!r} (expected key=v1,v2,...)")
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        result[key.strip()] = parsed
    return result


if __name__ == "__main__":
    run_lab()
