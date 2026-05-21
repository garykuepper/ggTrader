"""CLI: ``ggt backtest-strategy`` — run a new-architecture Strategy backtest.

Phase 3.5: feature store widened (DataFrame return), Pricer removed, rolls
detected and charged a single fee. Currently wired for ``cash_and_carry_btc``
against synthetic basis data. Real-data backtest lands after the Kraken
Futures backfill (separate followup).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ggTrader.backtest.vectorized import run_backtest
from ggTrader.core.calendars import Crypto24x7Calendar
from ggTrader.core.instrument import Instrument
from ggTrader.features.base import InstrumentArg, _normalize_instruments, pair_column_label
from ggTrader.features.derivatives_synthetic import (
    SyntheticBasisConfig,
    SyntheticFeatureStore,
)
from ggTrader.strategies.carry.cash_and_carry import CashAndCarryBTC
from ggTrader.strategies.loader import build_strategy_from_yaml, load_strategy_yaml

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "strategies"


def register_backtest_strategy_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "backtest-strategy",
        help="Backtest a new-architecture Strategy from a YAML config",
    )
    parser.add_argument(
        "strategy_id",
        type=str,
        help="Strategy id matching a yaml in src/ggTrader/config/strategies/",
    )
    parser.add_argument("--start", type=str, default=None, help="ISO UTC date")
    parser.add_argument("--end", type=str, default=None, help="ISO UTC date")
    parser.add_argument(
        "--starting-equity",
        type=str,
        default="100000",
        help="Starting USD equity per leg (default 100000)",
    )
    parser.add_argument(
        "--equity-curve-out",
        type=str,
        default=None,
        help="Write equity curve CSV to this path",
    )


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class _BroadcastFutureFeatureStore:
    """Synthetic-data broadcast wrapper: route every quarterly contract's
    ``basis_apr`` / ``mid_price`` to the same underlying series. Phase 3.5
    expediency. Real implementation reads per-contract series from TimescaleDB.
    """

    def __init__(
        self,
        inner: SyntheticFeatureStore,
        spot: Instrument,
        futures: list[Instrument],
    ) -> None:
        self._inner = inner
        self._spot = spot
        self._futures = futures
        self._anchor = futures[0]

    def get(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        insts = _normalize_instruments(instruments, start)
        if feature_name == "basis_apr" and len(insts) == 2:
            # Map any (spot, future_X) → (spot, anchor) for the inner store.
            return self._inner.get("basis_apr", [self._spot, self._anchor], start, end).rename(
                columns={pair_column_label([self._spot, self._anchor]): pair_column_label(insts)}
            )
        return self._inner.get(feature_name, insts, start, end)

    def get_at(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        ts: datetime,
    ) -> pd.Series:
        df = self.get(feature_name, instruments, ts, ts)
        try:
            return df.loc[ts]
        except KeyError as exc:
            raise LookupError(f"no value for {feature_name} at {ts.isoformat()}") from exc


def run_backtest_strategy(args: argparse.Namespace) -> None:
    yaml_path = _CONFIG_DIR / f"{args.strategy_id}.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"No strategy config at {yaml_path}")

    raw = load_strategy_yaml(yaml_path)
    strategy = build_strategy_from_yaml(yaml_path)

    backtest_cfg = raw.get("backtest", {})
    start = _parse_date(args.start or backtest_cfg.get("start"))
    end = _parse_date(args.end or backtest_cfg.get("end"))
    starting_equity = Decimal(str(args.starting_equity))

    # Phase 4: FundingCarryBTC reads from TimescaleDB. Legacy CashAndCarryBTC
    # still uses the synthetic basis feature store (Phase 3.5 behavior).
    from ggTrader.strategies.carry.funding_carry import FundingCarryBTC

    if isinstance(strategy, FundingCarryBTC):
        from ggTrader.features.timescale_store import TimescaleFeatureStore

        feature_store = TimescaleFeatureStore()
    elif isinstance(strategy, CashAndCarryBTC):
        spot = strategy.carry_universe.spot
        futures = strategy.carry_universe.futures
        inner = SyntheticFeatureStore(
            config=SyntheticBasisConfig(
                start=start, end=end, seed=int(backtest_cfg.get("synthetic_seed", 42))
            ),
            spot_instrument=spot,
            future_instruments=futures,
        )
        feature_store = _BroadcastFutureFeatureStore(inner, spot, futures)
    else:
        raise SystemExit(
            f"This CLI doesn't yet know how to wire a backtest for {type(strategy).__name__}"
        )

    carry_fn = getattr(strategy, "position_carry", None)
    result = run_backtest(
        strategy=strategy,
        feature_store=feature_store,
        start=start,
        end=end,
        starting_equity=starting_equity,
        calendar=Crypto24x7Calendar(),
        position_carry_fn=carry_fn,
    )

    metrics = result.metrics()
    basis_apr_capture = _summarize_basis_capture(result.trades)
    total_trade_fees = sum((t.fee for t in result.trades), Decimal("0"))
    total_roll_fees = sum((r.spread_fee for r in result.rolls), Decimal("0"))

    print("=== Backtest Summary ===")
    print(f"Strategy:        {strategy.strategy_id}")
    print(f"Period:          {start.date()} → {end.date()}  ({metrics['years']:.2f} years)")
    print(f"Starting equity: ${metrics['starting_equity']:,.2f}")
    print(f"Ending equity:   ${metrics['ending_equity']:,.2f}")
    print(f"Total return:    {metrics['total_return']:.2%}")
    print(f"CAGR:            {metrics['cagr']:.2%}")
    print(f"Sharpe:          {metrics['sharpe']:.2f}")
    print(f"Sortino:         {metrics['sortino']:.2f}")
    print(f"Max drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"Calmar:          {metrics['calmar']:.2f}")
    print(f"Trades:          {int(metrics['n_trades'])}")
    print(f"Rolls:           {int(metrics['n_rolls'])}")
    print(f"Trade fees:      ${float(total_trade_fees):,.2f}")
    print(f"Roll fees:       ${float(total_roll_fees):,.2f}")
    is_synthetic = isinstance(feature_store, _BroadcastFutureFeatureStore)
    if basis_apr_capture:
        print(f"Avg entry basis: {basis_apr_capture:.2%} APR")
    print()
    if is_synthetic:
        print("NOTE: Backtest ran against SYNTHETIC basis data (Phase 3.5).")
    else:
        print(
            f"NOTE: Real-data backtest via TimescaleFeatureStore "
            f"(feature_store={type(feature_store).__name__})."
        )

    if args.equity_curve_out:
        result.equity_curve.to_csv(args.equity_curve_out, header=True)
        print(f"\nEquity curve written to {args.equity_curve_out}")
        out_json = Path(args.equity_curve_out).with_suffix(".metrics.json")
        out_json.write_text(json.dumps(metrics, indent=2))


def _summarize_basis_capture(trades: list) -> float:
    entry_aprs: list[float] = []
    for t in trades:
        if t.reason != "entry":
            continue
        basis = t.metadata.get("basis_apr")
        if basis is None:
            continue
        try:
            entry_aprs.append(float(basis))
        except (TypeError, ValueError):
            continue
    if not entry_aprs:
        return 0.0
    return sum(entry_aprs) / len(entry_aprs)
