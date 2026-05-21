"""Phase 4 three-backtest comparison + proxy-credibility analysis.

Runs:
  A. Real funding, 2025-05-15 → 2026-05-15  (full API-permitted window)
  B. Basis proxy, 2023-01-01 → 2025-05-15   (older regime; no real funding)
  C. Basis proxy, 2025-05-15 → 2026-05-15   (overlap; validates B)

Then:
  - Correlation of (real funding APR) vs (basis premium APR) on the overlap.
  - Trade-event overlap between A and C.
  - Regime breakdown of B by calendar year.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from ggTrader.backtest.vectorized import BacktestResult, run_backtest
from ggTrader.core.calendars import Crypto24x7Calendar
from ggTrader.features.timescale_store import TimescaleFeatureStore
from ggTrader.strategies.loader import build_strategy_from_yaml

load_dotenv()
CONFIGS = "src/ggTrader/config/strategies"


def _conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ggtrader"),
        password=os.getenv("DB_PASS", "ggtrader"),
        dbname=os.getenv("DB_NAME", "ggtrader"),
    )


def run(yaml: str, start: str, end: str) -> tuple[BacktestResult, dict]:
    strat = build_strategy_from_yaml(f"{CONFIGS}/{yaml}")
    fs = TimescaleFeatureStore()
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    carry_fn = getattr(strat, "position_carry", None)
    r = run_backtest(
        strat,
        fs,
        start_dt,
        end_dt,
        Decimal("100000"),
        calendar=Crypto24x7Calendar(),
        position_carry_fn=carry_fn,
    )
    fs.close()
    return r, r.metrics()


def fmt_row(label: str, m: dict, fees: float) -> str:
    return (
        f"  {label:35s}  "
        f"return={m['total_return'] * 100:+6.2f}%  "
        f"sharpe={m['sharpe']:5.2f}  "
        f"max_dd={m['max_drawdown'] * 100:+5.2f}%  "
        f"trades={int(m['n_trades']):3d}  "
        f"fees=${fees:6.2f}"
    )


def main() -> None:
    print("Running A: real funding, 2025-05-15 → 2026-05-15 ...")
    rA, mA = run("funding_carry_btc_real.yaml", "2025-05-15", "2026-05-15")

    print("Running B: basis proxy, 2023-01-01 → 2025-05-15 ...")
    rB, mB = run("funding_carry_btc_basis.yaml", "2023-01-01", "2025-05-15")

    print("Running C: basis proxy, 2025-05-15 → 2026-05-15 (overlap) ...")
    # Reuse basis YAML, override window
    rC, mC = run("funding_carry_btc_basis.yaml", "2025-05-15", "2026-05-15")

    feesA = float(sum((t.fee for t in rA.trades), Decimal("0")))
    feesB = float(sum((t.fee for t in rB.trades), Decimal("0")))
    feesC = float(sum((t.fee for t in rC.trades), Decimal("0")))

    print()
    print("=== Side-by-side ===")
    print(fmt_row("A. Real funding   (1y, 2025-2026)", mA, feesA))
    print(fmt_row("B. Basis proxy    (2.4y, 2023-2025)", mB, feesB))
    print(fmt_row("C. Basis proxy   (overlap, 2025-2026)", mC, feesC))

    # ---- Proxy quality: A vs C ----
    conn = _conn()
    df = pd.read_sql(
        """SELECT f."timestamp" AS ts,
                  f.funding_apr_30d AS funding,
                  b.premium_apr_30d AS basis
           FROM funding_apr f
           JOIN basis_series b ON b."timestamp" = f."timestamp"
                              AND b.perp_symbol = f.symbol
           WHERE f.symbol = 'PF_XBTUSD'
           ORDER BY f."timestamp" """,
        conn,
    )
    df["ts"] = pd.to_datetime(df["ts"])
    df["funding"] = pd.to_numeric(df["funding"])
    df["basis"] = pd.to_numeric(df["basis"])
    corr = df["funding"].corr(df["basis"])
    bias = (df["basis"] - df["funding"]).mean()
    rmse = ((df["basis"] - df["funding"]) ** 2).mean() ** 0.5
    print()
    print(f"=== Proxy quality (basis vs real funding, n={len(df):,} hourly bars) ===")
    print(f"  Correlation:       {corr:+.3f}")
    print(f"  Mean bias (basis - funding):  {bias * 100:+.2f}% APR")
    print(f"  RMSE:              {rmse * 100:.2f}% APR")

    # ---- Trade-overlap A vs C ----
    a_events = [(t.ts.date(), t.reason, t.instrument_symbol) for t in rA.trades]
    c_events = [(t.ts.date(), t.reason, t.instrument_symbol) for t in rC.trades]
    a_set, c_set = set(a_events), set(c_events)
    overlap = a_set & c_set
    print()
    print("=== Trade-event overlap A vs C ===")
    print(f"  A events: {len(a_set)}    C events: {len(c_set)}    common: {len(overlap)}")

    # ---- Regime breakdown of B by calendar year ----
    print()
    print("=== Regime breakdown of B (basis-proxy 3-year) ===")
    eq = rB.equity_curve.copy()
    eq.index = pd.to_datetime(eq.index)
    for year, g in eq.groupby(eq.index.year):
        if len(g) < 2:
            continue
        ret = float(g.iloc[-1] / g.iloc[0] - 1.0)
        dd = float((g / g.cummax() - 1).min())
        print(f"  {year}: return {ret * 100:+6.2f}%   max_dd {dd * 100:+5.2f}%   days={len(g)}")

    conn.close()


if __name__ == "__main__":
    main()
