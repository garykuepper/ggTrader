"""Top-N coin correlation matrix visualizer.

Loads the top-N USD coins by Kraken volume, computes a pairwise return
correlation matrix from TimescaleDB OHLCV history, and renders a clustered
heatmap. Used to inform the BTC-leader regime tier (which coins are in the
BTC cluster vs. moving independently).

Run via Docker so ``host.docker.internal`` resolves to the host's TimescaleDB:

    docker compose run --rm ggtrader_live python -u scripts/coin_correlation_matrix.py

Optional flags:
    --top N            number of coins (default 50)
    --interval 4h      bar interval (default 4h)
    --start YYYY-MM-DD start date (default 2024-01-01)
    --end YYYY-MM-DD   end date (default today)
    --threshold T      highlight corrs ≥ T (default 0.7, matches LEADER_CORR_THRESHOLD)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Allow running from project root or scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader  # noqa: E402

UNIVERSE_FILE = ROOT / "data" / "top_50_ccxt_volume.json"


def load_top_n_symbols(top_n: int) -> list[str]:
    with open(UNIVERSE_FILE) as f:
        items = json.load(f)
    items = sorted(items, key=lambda x: x["rank"])[:top_n]
    return [f"{x['symbol']}-USD" for x in items]


def fetch_returns(
    symbols: list[str], interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    loader = TimescaleDBLoader()
    ohlcv = loader.fetch_ohlcv(symbols=symbols, interval=interval, start_date=start, end_date=end)
    if ohlcv.empty:
        raise SystemExit("No OHLCV data returned from TimescaleDB.")
    close = ohlcv.xs("close", axis=1, level=1)
    # Drop coins with too little history (< 50% of bars)
    min_bars = int(len(close) * 0.5)
    keep = [c for c in close.columns if close[c].dropna().shape[0] >= min_bars]
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        print(f"  [universe] Dropping {len(dropped)} coin(s) with <50% history: {dropped}")
    kept_df: pd.DataFrame = close.loc[:, keep]
    return kept_df.pct_change().dropna(how="all")


def btc_anchored_order(corr: pd.DataFrame) -> list[str]:
    """BTC pinned first, then remaining coins sorted by descending corr-to-BTC.

    Keeps rows/columns aligned so the diagonal stays meaningful, while making
    the BTC cluster easy to read top-to-bottom and left-to-right.
    """
    bench: str = "BTC-USD" if "BTC-USD" in corr.columns else str(corr.columns[0])
    btc_corrs: pd.Series = corr[bench]
    others = [str(s) for s in btc_corrs.drop(labels=[bench]).sort_values(ascending=False).index]
    return [bench, *others]


def render_heatmap(corr: pd.DataFrame, threshold: float, out_path: Path) -> None:
    """Discrete 10-band diverging heatmap, 0.2 increments mirrored across 0.

    Reds mirror the blue scale below zero, so |corr|=0.9 reads with the same
    visual weight whether positive or negative.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n = len(corr)
    fig_side = max(8, n * 0.32)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))

    boundaries = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    band_colors = [
        # Reds: darkest at -1.0, fading to lightest near 0.
        "#67000d",  # -1.0 – -0.8
        "#a50f15",  # -0.8 – -0.6
        "#cb181d",  # -0.6 – -0.4
        "#fb6a4a",  # -0.4 – -0.2
        "#fee5d9",  # -0.2 –  0.0
        # Blues: lightest near 0, darkening toward +1.0 (mirrors red scale).
        "#f7fbff",  #  0.0 –  0.2
        "#deebf7",  #  0.2 –  0.4
        "#9ecae1",  #  0.4 –  0.6
        "#4292c6",  #  0.6 –  0.8
        "#08306b",  #  0.8 –  1.0
    ]
    cmap = ListedColormap(band_colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)

    # Black gridlines around each cell. Minor ticks at half-offsets so the
    # grid sits on cell boundaries (imshow centres each value on integer ticks).
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    centres = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=centres)
    cbar.ax.set_yticklabels(
        [
            "-1.0 – -0.8",
            "-0.8 – -0.6",
            "-0.6 – -0.4",
            "-0.4 – -0.2",
            "-0.2 –  0.0",
            " 0.0 –  0.2",
            " 0.2 –  0.4",
            " 0.4 –  0.6",
            " 0.6 –  0.8",
            " 0.8 –  1.0",
        ]
    )
    cbar.set_label("Pearson correlation")

    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            if i != j and abs(v) >= threshold:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white")

    ax.set_title(
        f"Coin return correlation matrix (n={n}, BTC-gate threshold={threshold:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def print_btc_ranking(corr: pd.DataFrame, threshold: float) -> None:
    if "BTC-USD" not in corr.columns:
        print("  [btc-rank] BTC-USD missing from correlation matrix — skipping.")
        return
    btc_col = corr["BTC-USD"]
    if isinstance(btc_col, pd.DataFrame):
        btc_col = btc_col.iloc[:, 0]
    series_raw: pd.Series = pd.Series(btc_col).drop(labels=["BTC-USD"])
    pairs = sorted(
        ((str(sym), float(v)) for sym, v in series_raw.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    above = [(s, v) for s, v in pairs if v >= threshold]
    below = [(s, v) for s, v in pairs if v < threshold]
    print(f"\n=== BTC correlation ranking (threshold={threshold:.2f}) ===")
    print(f"\nGated by BTC regime (corr ≥ {threshold:.2f}) — {len(above)} coin(s):")
    for sym, c in above:
        print(f"  {sym:<14} {c:+.3f}")
    print(f"\nFree tier (corr < {threshold:.2f}) — {len(below)} coin(s):")
    for sym, c in below:
        print(f"  {sym:<14} {c:+.3f}")


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise SystemExit(f"Could not parse timestamp: {value!r}")
    assert isinstance(ts, pd.Timestamp)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def main() -> None:
    desc = (__doc__ or "").strip().splitlines()[0]
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--interval", default="4h")
    p.add_argument("--start", default="2026-04-01")
    p.add_argument("--end", default=None)
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()

    start = _parse_ts(args.start)
    end = _parse_ts(args.end) if args.end else pd.Timestamp.now(tz="UTC")

    symbols = load_top_n_symbols(args.top)
    print(
        f"  [universe] Top {args.top} USD coins, {args.interval} bars, "
        f"{start.date()} → {end.date()}"
    )
    print(f"  [universe] Fetching returns for: {symbols}")
    returns = fetch_returns(symbols, args.interval, start, end)
    corr = returns.corr()

    # BTC pinned first, others sorted by descending corr-to-BTC
    order = btc_anchored_order(corr)
    corr = corr.reindex(index=order, columns=order)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"correlation_matrix_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = out_dir / "heatmap.png"
    render_heatmap(corr, args.threshold, heatmap)
    corr.to_csv(out_dir / "correlation_matrix.csv")
    print(f"\n  [output] heatmap → {heatmap}")
    print(f"  [output] csv     → {out_dir / 'correlation_matrix.csv'}")

    print_btc_ranking(corr, args.threshold)


if __name__ == "__main__":
    main()
