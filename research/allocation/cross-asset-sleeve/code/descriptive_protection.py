"""Compare fixed cross-asset and Treasury-bill portfolios on frozen adjusted closes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("SPY", "BIL", "TLT", "GLD", "PDBC")
WEIGHTS = {
    "cross_asset": {"SPY": 0.8, "TLT": 0.2 / 3, "GLD": 0.2 / 3, "PDBC": 0.2 / 3},
    "spy_bil": {"SPY": 0.8, "BIL": 0.2},
    "spy": {"SPY": 1.0},
}


def portfolio(prices: pd.DataFrame, weights: dict[str, float], fee: float) -> pd.Series:
    """Compound shares and rebalance at month-end closes; charge turnover costs."""
    prices = prices[list(weights)]
    target = pd.Series(weights)
    values = target * (1 - fee)
    curve = [float(values.sum())]
    for i in range(1, len(prices)):
        values = values * (prices.iloc[i] / prices.iloc[i - 1])
        total = float(values.sum())
        if i < len(prices) - 1 and prices.index[i].month != prices.index[i + 1].month:
            turnover = float((target * total - values).abs().sum())
            total -= turnover * fee
            values = target * total
        curve.append(total)
    return pd.Series(curve, index=prices.index)


def summarize(curve: pd.Series) -> dict:
    """Report wealth, maximum drawdown and recovery duration in trading sessions."""
    running = curve.cummax()
    drawdown = curve / running - 1
    trough = int(np.argmin(drawdown.to_numpy()))
    peak = int(np.argmax(curve.iloc[: trough + 1].to_numpy()))
    peak_value = float(curve.iloc[peak])
    recovered = np.flatnonzero(curve.iloc[trough:].to_numpy() >= peak_value)
    end = trough + int(recovered[0]) if len(recovered) else len(curve) - 1
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    return {
        "terminal_wealth_per_1": float(curve.iloc[-1]),
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": float(drawdown.iloc[trough]),
        "peak": str(curve.index[peak].date()),
        "trough": str(curve.index[trough].date()),
        "recovered": bool(len(recovered)),
        "recovery_or_observed_end": str(curve.index[end].date()),
        "underwater_sessions": end - peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs, series = {}, {}
    for symbol in SYMBOLS:
        path = args.input_dir / f"{symbol}.csv"
        inputs[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        dates = (
            pd.to_datetime(frame["date"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        series[symbol] = pd.Series(frame["Adj Close"].to_numpy(), index=dates)
    prices = pd.DataFrame(series).loc["2021-04-30":"2026-04-30"]
    if prices.isna().any().any() or (prices <= 0).any().any() or prices.index.has_duplicates:
        raise ValueError("Missing, invalid or duplicate adjusted prices")
    if str(prices.index[0].date()) != "2021-04-30" or str(prices.index[-1].date()) != "2026-04-30":
        raise ValueError("Input does not cover both evaluation endpoints")
    curves = {name: portfolio(prices, weights, 0.0001) for name, weights in WEIGHTS.items()}
    # Complete calendar years only; all portfolios are measured over identical dates.
    annual = {}
    for year in range(2022, 2026):
        annual[str(year)] = {
            name: float(curve.loc[str(year)].iloc[-1] / curve.loc[: f"{year}-01-01"].iloc[-1] - 1)
            for name, curve in curves.items()
        }
    result = {
        "status": "descriptive; seen data; no pass/fail",
        "window": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "input_sha256": inputs,
        "source": "Yahoo Finance through yfinance Ticker.history(auto_adjust=False, actions=True)",
        "yfinance_version": importlib.metadata.version("yfinance"),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assumptions": (
            "Yahoo adjusted closes; monthly close rebalance; 1bp per dollar traded "
            "including initial purchases; no tax or terminal liquidation cost"
        ),
        "portfolios": {name: summarize(curve) for name, curve in curves.items()},
        "calendar_year_returns": annual,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
