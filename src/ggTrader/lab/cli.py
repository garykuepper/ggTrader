"""CLI for the lab research bench: walk-forward a strategy (or blend sleeves) on the engine.

``ggt lab --wfo`` and ``ggt lab --blend`` run on the vectorbt engine (``lab.wfo``) and read the
frozen tape snapshot. The pre-rebuild ``--engine``, ``--sweep``, ``--max-sector-count`` and the
plain (no mode) run were removed in P5c; strategies that were not ported stay runnable at git tag
``legacy-engine-vbt0285`` (``scripts/legacy_env.sh``).
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from ggTrader.lab.data import CRYPTO_BASE_CONFIG, STOCK_BASE_CONFIG
from ggTrader.lab.strategies import registry
from ggTrader.lab.strategy import LabConfig

UNIVERSE_CHOICES = (
    "sp500",
    "nasdaq100",
    "russell2000",
    "midcap400",
    "fx_hedge",
    "fomc_treasury",
    "commodity_trend",
    "treasury_curve",
    "cross_asset_sleeve",
)


def check_strategy(name: str) -> str:
    """Return ``name`` if it is ported; otherwise exit with a message that says where it runs."""
    if name in registry.REGISTRY:
        return name
    if name in registry.LEGACY_ONLY_STRATEGIES:
        raise SystemExit(
            f"strategy {name!r} is not ported to the engine contract; it stays runnable at git "
            f"tag {registry.LEGACY_TAG} (scripts/legacy_env.sh)"
        )
    raise SystemExit(f"Unknown strategy {name!r}; ported strategies: {sorted(registry.REGISTRY)}")


def _parse_blend_sleeves(spec: str) -> List[tuple[str, str]]:
    """Parse 'strat@univ,strat@univ' into [(strategy, universe), ...]; validate."""
    sleeves: List[tuple[str, str]] = []
    for part in spec.split(","):
        item = part.strip()
        if item.count("@") != 1:
            raise SystemExit(f"Bad sleeve {item!r}; expected 'strategy@universe'")
        strategy, universe = item.split("@")
        check_strategy(strategy)
        if universe not in UNIVERSE_CHOICES:
            raise SystemExit(f"Unknown universe {universe!r}; choices: {UNIVERSE_CHOICES}")
        sleeves.append((strategy, universe))
    return sleeves


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a lab strategy walk-forward (or blend sleeves) on the vectorbt engine."
    )
    p.add_argument(
        "--strategy",
        default=None,
        metavar="NAME",
        help="Ported strategy (required with --wfo): " + ", ".join(sorted(registry.REGISTRY)),
    )
    p.add_argument("--market", default="equity")
    p.add_argument(
        "--universe",
        choices=UNIVERSE_CHOICES,
        default="sp500",
        help="Stock universe to select from (default: sp500).",
    )
    p.add_argument("--eval-start", default="2021-01-31")
    p.add_argument("--eval-end", default=None)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--max-stocks", type=int, default=None)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--wfo",
        action="store_true",
        default=False,
        help="Walk-forward optimization: rolling train/test folds with OOS scoring.",
    )
    mode.add_argument(
        "--blend",
        default=None,
        metavar="S@U,...",
        help="Blend sleeves: --blend ensemble@sp500,xs_momentum@nasdaq100",
    )
    p.add_argument("--target-vol", type=float, default=0.068)
    p.add_argument("--blend-window", type=int, default=60)
    # 1.0 is the deployed value (the old 2.0 default overstated blends; the trap was hit twice).
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument(
        "--snapshot",
        default=None,
        metavar="PATH",
        help="Parquet tape to read (default: the newest in data/snapshots).",
    )
    p.add_argument(
        "--result-json",
        default=None,
        metavar="PATH",
        help="--wfo: where to write the result JSON with its provenance block "
        "(default results/lab/wfo_<strategy>_<universe>_<start>_<end>_vbt.json).",
    )
    p.add_argument(
        "--sweep-param",
        action="append",
        default=[],
        help="Override a grid axis: --sweep-param min_agree=2,3",
    )
    return p


def run_lab(argv: List[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    cfg = LabConfig(
        top_n=args.top_n,
        lookback=args.lookback,
        skip=args.skip,
        max_stocks=args.max_stocks,
    )

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = (
        pd.Timestamp(args.eval_end, tz="UTC")
        if args.eval_end
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    base_config = CRYPTO_BASE_CONFIG if args.market == "crypto" else STOCK_BASE_CONFIG

    if args.blend:
        from ggTrader.lab.blend import run_blend

        sleeves = _parse_blend_sleeves(args.blend)
        result = run_blend(
            sleeves,
            cfg,
            str(eval_start.date()),
            str(eval_end.date()),
            market=args.market,
            base_config=dict(base_config),
            target_vol=args.target_vol,
            window=args.blend_window,
            max_leverage=args.max_leverage,
            snapshot=args.snapshot,
        )
        print(result.table)
        print(f"blend run complete: {result.run_id}")
        return result.run_id

    if not args.strategy:
        raise SystemExit("--wfo needs --strategy NAME")
    check_strategy(args.strategy)
    from ggTrader.lab.wfo.cli_support import run_vbt_wfo

    print(f"WFO: {args.strategy} | vectorbt engine", flush=True)
    outcome = run_vbt_wfo(
        args.strategy,
        args.universe,
        cfg,
        eval_start,
        eval_end,
        market=args.market,
        base_config=dict(base_config),
        overrides=_parse_sweep_params(args.sweep_param) or None,
        snapshot=args.snapshot,
        result_json=args.result_json,
    )
    return outcome if isinstance(outcome, str) else outcome.table


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
            if v == "None":
                parsed.append(None)
            elif v == "True":
                parsed.append(True)
            elif v == "False":
                parsed.append(False)
            else:
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
