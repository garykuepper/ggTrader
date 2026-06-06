"""Offline robustness harness for the experimental cross-sectional momentum strategy.

Runs ``CrossSectionalMomentum`` through the standalone ``WalkForwardOptimizer`` on
real Binance.US 4h OHLCV from TimescaleDB and emits a robustness summary plus an
IS-vs-OOS Sharpe scatter, so we can judge whether the cross-sectional alpha is
worth a (separately-scoped) live integration.

This is VERIFICATION ONLY. It is not wired into the live trader, writes results
under ``run_type='cross_sectional_research'`` (never ``'research'``, so the live
trader's discovery ignores it), and keeps the HMM regime filter disabled
(``hmm_filter_enabled=False``) because its emission features are not yet real.

Run on the host (``host.docker.internal`` auto-rewrites to ``localhost``):

    .venv/bin/python scripts/run_cross_sectional_research.py --help

Smoke test (fast, ~12 symbols, 1-combo grid):

    .venv/bin/python scripts/run_cross_sectional_research.py \
        --symbols BTC-USD,ETH-USD,SOL-USD,DOGE-USD,XRP-USD,ADA-USD,LINK-USD,DOT-USD \
        --start 2024-06-01 --end 2025-06-01 \
        --in-sample-bars 540 --out-of-sample-bars 180 --step-bars 180 --smoke

Full run (auto-universe, deep history):

    .venv/bin/python scripts/run_cross_sectional_research.py \
        --start 2023-01-01 --min-history-bars 4000 --recent-after 2025-09-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

# Allow running from project root or scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.backtesting.wfo import WalkForwardOptimizer, WFOConfig  # noqa: E402
from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader  # noqa: E402
from ggTrader.strategies.momentum.config import MomentumConfig  # noqa: E402
from ggTrader.strategies.momentum.cross_sectional import CrossSectionalMomentum  # noqa: E402
from ggTrader.utils.results_manager import ResultsManager  # noqa: E402

# Weights are held fixed at the for_crypto() default (0.6/0.4) so every combo
# satisfies MomentumConfig's w_momentum+w_liquidity==1.0 validator. To sweep
# weights, use --sweep-weights (runs once per pair). The grid only contains
# orthogonal, individually-valid fields.
DEFAULT_GRID = {
    "formation_window": [20, 30, 40],
    "exclusion_gap": [3, 5],
    "entry_percentile": [0.85, 0.90],
    "hold_bars": [3, 5],
}  # 24 combos

SMOKE_GRID = {
    "formation_window": [30],
    "entry_percentile": [0.90],
}  # 1 combo

# (w_momentum, w_liquidity) pairs used only when --sweep-weights is set.
WEIGHT_PAIRS = [(0.6, 0.4), (0.7, 0.3), (0.5, 0.5)]

# Used only if the universe auto-query returns nothing (e.g. unexpected schema).
# Deep-history, still-live Binance.US names; BTC-USD is mandatory for beta strip.
FALLBACK_UNIVERSE = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "DOGE-USD",
    "XRP-USD",
    "ADA-USD",
    "XLM-USD",
    "SUI-USD",
    "PEPE-USD",
    "LINK-USD",
    "DOT-USD",
    "AVAX-USD",
    "LTC-USD",
    "BCH-USD",
    "NEAR-USD",
    "ICP-USD",
    "FET-USD",
    "CRV-USD",
]


def resolve_universe(
    loader: TimescaleDBLoader,
    explicit: list[str] | None,
    venue: str,
    interval: str,
    min_history_bars: int,
    recent_after: pd.Timestamp,
) -> list[str]:
    """Return the BASE-USD symbols to trade. BTC-USD is always included.

    If ``explicit`` is given it is used verbatim. Otherwise distinct symbols are
    queried from the ``ohlcv`` table for the venue/interval, keeping only those
    with enough history AND a recent bar (excludes dead coins). The 2-bar junk
    ``BTC`` symbol is excluded by the ``-USD`` suffix filter.
    """
    if explicit:
        syms = {s.strip().upper() for s in explicit if s.strip()}
        syms.add("BTC-USD")
        return sorted(syms)

    query = text(
        """
        SELECT symbol
        FROM ohlcv
        WHERE venue = :venue AND interval = :interval AND symbol LIKE '%-USD'
        GROUP BY symbol
        HAVING COUNT(*) >= :min_history_bars AND MAX(timestamp) >= :recent_after
        ORDER BY COUNT(*) DESC;
        """
    )
    try:
        with loader.engine.connect() as conn:
            rows = conn.execute(
                query,
                {
                    "venue": venue,
                    "interval": interval,
                    "min_history_bars": min_history_bars,
                    "recent_after": recent_after.to_pydatetime(),
                },
            )
            symbols = [r[0] for r in rows]
    except Exception as e:  # pragma: no cover - depends on live DB
        print(f"Universe query failed ({e}); using fallback list.")
        symbols = []

    if not symbols:
        symbols = list(FALLBACK_UNIVERSE)

    out = sorted(set(symbols) | {"BTC-USD"})
    return out


def load_universe_frames(
    loader: TimescaleDBLoader,
    symbols: list[str],
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    venue: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV and return (close_df, volume_df) with single-level symbol columns."""
    raw = loader.fetch_ohlcv(
        symbols, interval=interval, start_date=start, end_date=end, venue=venue
    )
    if raw.empty:
        raise ValueError(
            f"No OHLCV returned for venue={venue} interval={interval}. "
            "Check the venue string (expected 'binanceus_spot') and symbols."
        )

    # fetch_ohlcv returns two-level columns (symbol, metric); metric is level 1.
    close_df = raw.xs("close", axis=1, level=1).sort_index()
    volume_df = raw.xs("volume", axis=1, level=1).reindex_like(close_df)

    # Hygiene: drop all-NaN columns, align, forward-fill short gaps, drop empty rows.
    close_df = close_df.dropna(axis=1, how="all")
    volume_df = volume_df.reindex(columns=close_df.columns)
    close_df = close_df.dropna(how="all")
    volume_df = volume_df.reindex(index=close_df.index, columns=close_df.columns)
    close_df = close_df.ffill()
    volume_df = volume_df.fillna(0.0)

    if "BTC-USD" not in close_df.columns:
        raise ValueError(
            "BTC-USD missing after load — required for crypto beta-stripping. "
            f"Got columns: {list(close_df.columns)}"
        )

    return close_df, volume_df


def build_base_config(w_momentum: float, w_liquidity: float) -> MomentumConfig:
    """Crypto base config (HMM off), optionally with overridden composite weights."""
    cfg = MomentumConfig.for_crypto()
    if (w_momentum, w_liquidity) != (cfg.w_momentum, cfg.w_liquidity):
        data = cfg.model_dump()
        data["w_momentum"] = w_momentum
        data["w_liquidity"] = w_liquidity
        cfg = MomentumConfig(**data)
    return cfg


def run_optimizer(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    base_config: MomentumConfig,
    param_grid: dict,
    wfo_config: WFOConfig,
) -> tuple[WalkForwardOptimizer, dict]:
    """Run one WFO pass with the HMM filter explicitly disabled."""
    opt = WalkForwardOptimizer(
        CrossSectionalMomentum, param_grid, wfo_config, base_config=base_config
    )
    # Bool kwarg is forwarded (not iloc-sliced) to CrossSectionalMomentum.run().
    # Never pass regime_gate so the HMM code path is untouched.
    opt.run(close_df, volume_df, hmm_filter_enabled=False)
    return opt, opt.summary()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline WFO robustness harness for the cross-sectional momentum "
            "strategy (Binance.US 4h, HMM disabled). Verification only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--symbols", default=None, help="Comma-separated BASE-USD list; overrides auto-universe."
    )
    p.add_argument("--venue", default="binanceus_spot", help="DB venue column value.")
    p.add_argument("--interval", default="4h", help="Bar interval in the DB.")
    p.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD); default today UTC.")
    p.add_argument(
        "--min-history-bars",
        type=int,
        default=4000,
        help="Auto-universe: min total 4h bars per symbol.",
    )
    p.add_argument(
        "--recent-after",
        default="2025-09-01",
        help="Auto-universe: require a bar on/after this date (drops dead coins).",
    )
    p.add_argument("--in-sample-bars", type=int, default=1080, help="IS window (bars).")
    p.add_argument("--out-of-sample-bars", type=int, default=360, help="OOS window (bars).")
    p.add_argument("--step-bars", type=int, default=360, help="Roll step (bars).")
    p.add_argument(
        "--sharpe-threshold",
        type=float,
        default=1.0,
        help="deploy_ready requires mean OOS Sharpe above this.",
    )
    p.add_argument(
        "--max-dd-threshold",
        type=float,
        default=0.15,
        help="deploy_ready requires max OOS drawdown below this (fraction).",
    )
    p.add_argument(
        "--sweep-weights",
        action="store_true",
        help="Run once per (w_mom,w_liq) pair and report each.",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Use a 1-combo grid for a fast end-to-end check."
    )
    p.add_argument("--output-dir", default=None, help="Explicit run dir (else timestamped).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    recent_after = pd.Timestamp(args.recent_after, tz="UTC")

    loader = TimescaleDBLoader()
    symbols = resolve_universe(
        loader,
        args.symbols.split(",") if args.symbols else None,
        args.venue,
        args.interval,
        args.min_history_bars,
        recent_after,
    )
    print(f"Universe ({len(symbols)} symbols): {', '.join(symbols)}")

    close_df, volume_df = load_universe_frames(
        loader, symbols, args.interval, start, end, args.venue
    )
    print(
        f"Loaded {close_df.shape[0]} bars x {close_df.shape[1]} symbols "
        f"[{close_df.index.min()} -> {close_df.index.max()}]"
    )

    wfo_config = WFOConfig(
        in_sample_bars=args.in_sample_bars,
        out_of_sample_bars=args.out_of_sample_bars,
        step_bars=args.step_bars,
        deployment_sharpe_threshold=args.sharpe_threshold,
        deployment_max_dd_threshold=args.max_dd_threshold,
    )
    param_grid = SMOKE_GRID if args.smoke else DEFAULT_GRID

    weight_pairs = WEIGHT_PAIRS if args.sweep_weights else [(0.6, 0.4)]
    runs: list[dict] = []
    for w_mom, w_liq in weight_pairs:
        label = f"w_mom={w_mom},w_liq={w_liq}"
        print(f"\n=== WFO pass [{label}] ===")
        base_config = build_base_config(w_mom, w_liq)
        opt, summary = run_optimizer(close_df, volume_df, base_config, param_grid, wfo_config)
        print(f"Summary [{label}]: {summary}")
        runs.append(
            {
                "weights": {"w_momentum": w_mom, "w_liquidity": w_liq},
                "summary": summary,
                "windows": opt.results,
                "_opt": opt,
                "_base_config": base_config,
            }
        )

    # Best pass = highest mean OOS Sharpe; its optimizer drives the plot.
    best = max(runs, key=lambda r: r["summary"].get("mean_oos_sharpe", float("-inf")))
    best_opt = best["_opt"]

    rm = ResultsManager(
        script_name="cross_sectional_research",
        pipeline_stage="research",
        explicit_run_dir=args.output_dir,
    )
    plot_path = rm.get_plot_path("wfo_robustness.png")
    try:
        best_opt.plot_robustness(save_path=str(plot_path))
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"Warning: could not render robustness plot: {e}")

    params = {
        "param_grid": param_grid,
        "weight_pairs": [list(p) for p in weight_pairs],
        "base_config": best["_base_config"].model_dump(),
    }
    metrics = {
        "best_weights": best["weights"],
        "best_summary": best["summary"],
        "passes": [
            {"weights": r["weights"], "summary": r["summary"], "windows": r["windows"]}
            for r in runs
        ],
    }
    metadata = {
        "START_DATE": args.start,
        "END_DATE": str(end.date()),
        "INTERVAL": args.interval,
        "VENUE": args.venue,
        "SYMBOLS": list(close_df.columns),
        "ASSET_CLASS": "crypto",
        "hmm_filter_enabled": False,
        "wfo_config": wfo_config.model_dump(),
    }

    try:
        out_path = rm.save_run_results(params=params, metrics=metrics, metadata=metadata)
    except Exception as e:
        # Tolerate a read-only DB (the ResultsManager mirrors to the runs table).
        print(f"Warning: ResultsManager.save_run_results failed ({e}); writing JSON directly.")
        out_path = rm.run_dir / "run_results.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "run_id": rm.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "script_name": "cross_sectional_research",
                    "asset_class": "crypto",
                    "configuration": metadata,
                    "strategy_parameters": params,
                    "results": metrics,
                },
                f,
                indent=4,
                default=str,
            )

    print("\n=== Best pass summary ===")
    rm.print_summary(best["summary"])
    deploy = best["summary"].get("deploy_ready", False)
    print(f"\ndeploy_ready: {deploy}  (weights {best['weights']})")
    print(f"Run dir:        {rm.run_dir}")
    print(f"Results JSON:   {out_path}")
    print(f"Robustness PNG: {plot_path}")


if __name__ == "__main__":
    main()
