"""Strategy-usage analyzer for WFO research runs.

Aggregates the WFO-selected `best_strategy` across recent research runs and
prints a ranked table so we can see which entry strategies consistently win
coins (and which never do).

Two data sources, in order of preference:
  1. ``runs`` table in TimescaleDB (the source of truth).
  2. ``results/research/research_*/run_results.json`` on disk (fallback / when
     running outside Docker).

Caveat about the win count: the WFO ``MAX_COINS_PER_STRATEGY`` diversity cap
(default 10) truncates each strategy's portfolio share *after* selection, so
a strategy that wins exactly the cap every run is probably more dominant than
the count alone suggests. The "median robustness when selected" column is the
better quality signal — a strategy with 5 wins at robustness 0.3 is genuinely
weaker than one with 5 wins at 0.6.

Usage:
    docker compose run --rm ggtrader_live python -u scripts/strategy_usage_stats.py
    python scripts/strategy_usage_stats.py --last 5
    python scripts/strategy_usage_stats.py --from-disk
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

# Allow running from project root or scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _per_coin_from_blob(blob: Any) -> dict[str, dict[str, Any]]:
    """Normalise either ``{per_coin: {...}}`` or a flat ``{sym: {...}}`` dict."""
    if not isinstance(blob, dict):
        return {}
    if "per_coin" in blob and isinstance(blob["per_coin"], dict):
        return blob["per_coin"]
    return blob


def _iter_runs_from_db(limit: Optional[int]) -> Iterable[tuple[str, str, dict]]:
    """Yield (run_id, timestamp_iso, per_coin_dict) for research runs in DB order."""
    from sqlalchemy import text  # type: ignore

    from ggTrader.utils.result_db_manager import ResultDBManager

    m = ResultDBManager()
    sql = """
        SELECT run_id, timestamp, strategy_params
        FROM runs
        WHERE run_type = 'research'
          AND COALESCE(status, 'success') = 'success'
          AND strategy_params IS NOT NULL
        ORDER BY timestamp DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with m.engine.connect() as conn:
        for row in conn.execute(text(sql)).fetchall():
            yield str(row[0]), str(row[1]), _per_coin_from_blob(row[2])


def _iter_runs_from_disk(limit: Optional[int]) -> Iterable[tuple[str, str, dict]]:
    """Fallback: read ``results/research/research_*/run_results.json``."""
    base = ROOT / "results" / "research"
    if not base.exists():
        return
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "run_results.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if limit:
        runs = runs[: int(limit)]
    for run_dir in runs:
        try:
            with open(run_dir / "run_results.json") as f:
                data = json.load(f)
        except Exception:
            continue
        sp = data.get("strategy_parameters", {})
        per_coin = sp.get("per_coin") or data.get("per_coin_results") or {}
        yield run_dir.name, str(data.get("timestamp", "")), per_coin


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_strategies(
    runs_iter: Iterable[tuple[str, str, dict]],
) -> tuple[dict, list[tuple[str, str, int]]]:
    """Return (per_strategy_stats, per_run_summary).

    per_strategy_stats[strategy] = {
        "wins": int,                   # times chosen as best_strategy across runs
        "robustness": list[float],     # one entry per coin where it was chosen
        "exits": Counter,              # which exits paired with this entry
        "coins": Counter,              # which coins picked it (multi-run dupes counted)
    }
    """
    from collections import Counter

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "robustness": [], "exits": Counter(), "coins": Counter()}
    )
    run_summary: list[tuple[str, str, int]] = []
    for run_id, ts, per_coin in runs_iter:
        n = 0
        for sym, info in per_coin.items():
            if not isinstance(info, dict):
                continue
            strat = info.get("best_strategy")
            if not strat:
                continue
            n += 1
            s = stats[strat]
            s["wins"] += 1
            s["coins"][sym] += 1
            exit_name = info.get("best_exit") or "<unknown>"
            s["exits"][exit_name] += 1
            r = info.get("robustness_score")
            try:
                rf = float(r) if r is not None else None
                if rf is not None and rf == rf:  # not NaN
                    s["robustness"].append(rf)
            except (TypeError, ValueError):
                pass
        run_summary.append((run_id, ts, n))
    return stats, run_summary


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "n/a"


def _median(xs: list[float]) -> str:
    return f"{statistics.median(xs):.3f}" if xs else "n/a"


def _quantile(xs: list[float], q: float) -> str:
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    qs = statistics.quantiles(xs, n=100, method="inclusive")
    idx = max(0, min(98, int(q * 100) - 1))
    return f"{qs[idx]:.3f}"


def render_table(
    stats: dict[str, dict[str, Any]],
    total_wins: int,
    *,
    full_registry: list[str],
) -> None:
    """Print a fixed-width table sorted by wins desc."""
    rows = []
    for strat in full_registry:
        s = stats.get(strat) or {"wins": 0, "robustness": [], "exits": {}, "coins": {}}
        wins = int(s["wins"])
        rob = list(s["robustness"])
        rows.append(
            (
                strat,
                wins,
                _pct(wins, total_wins),
                _median(rob),
                _quantile(rob, 0.25),
                _quantile(rob, 0.75),
                len(s["coins"]),
            )
        )
    rows.sort(key=lambda r: r[1], reverse=True)

    headers = ["strategy", "wins", "share", "med_rob", "p25_rob", "p75_rob", "uniq_coins"]
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def render_exit_pairings(stats: dict[str, dict[str, Any]]) -> None:
    """For each entry strategy, show which exits it most often pairs with."""
    print("\n=== Exit pairings (per entry strategy) ===")
    for strat in sorted(stats, key=lambda s: stats[s]["wins"], reverse=True):
        s = stats[strat]
        if s["wins"] == 0:
            continue
        pairs = ", ".join(f"{e}={n}" for e, n in s["exits"].most_common())
        print(f"  {strat:<22} {pairs}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--last", type=int, default=10, help="Aggregate the last N research runs (default: 10)")
    p.add_argument(
        "--from-disk",
        action="store_true",
        help="Read run_results.json from disk instead of querying the runs table",
    )
    args = p.parse_args()

    # Pull the entry registry so the output includes 0-win strategies (the whole point).
    try:
        from ggTrader.indicators.strategies import ENTRY_REGISTRY
        full_registry = list(ENTRY_REGISTRY.keys())
    except Exception:
        full_registry = []

    if args.from_disk:
        runs_iter = list(_iter_runs_from_disk(args.last))
        source = f"disk ({ROOT / 'results' / 'research'})"
    else:
        try:
            runs_iter = list(_iter_runs_from_db(args.last))
            source = "TimescaleDB runs table"
        except Exception as e:
            print(f"  [warn] DB query failed ({e!r}); falling back to disk")
            runs_iter = list(_iter_runs_from_disk(args.last))
            source = f"disk fallback ({ROOT / 'results' / 'research'})"

    if not runs_iter:
        print("No research runs found.")
        return

    stats, run_summary = aggregate_strategies(iter(runs_iter))
    total_wins = sum(int(v["wins"]) for v in stats.values())

    print(f"\n=== Strategy usage across {len(run_summary)} research run(s) — source: {source} ===")
    for run_id, ts, n in run_summary:
        print(f"  {run_id:<40} {ts:<28} {n} coin(s) selected")

    # Make sure 0-win registered strategies still appear in the table.
    # (Aggregator only creates an entry on first hit; pad here for visibility.)
    for strat in full_registry:
        stats.setdefault(strat, {"wins": 0, "robustness": [], "exits": {}, "coins": {}})

    render_table(
        stats,
        total_wins=total_wins,
        full_registry=full_registry or list(stats.keys()),
    )
    render_exit_pairings(stats)

    # Pruning candidates: <2% share AND median robustness <0.2 (or never selected at all).
    print("\n=== Pruning candidates (consistently weak or never selected) ===")
    weak: list[str] = []
    for strat, s in stats.items():
        wins = int(s["wins"])
        rob = list(s["robustness"])
        if wins == 0:
            weak.append(f"{strat} (0 wins across {len(run_summary)} runs)")
            continue
        share = wins / total_wins if total_wins else 0.0
        med = statistics.median(rob) if rob else float("nan")
        if share < 0.02 and (med != med or med < 0.2):
            weak.append(f"{strat} (share={share:.1%}, median_robustness={med:.3f})")
    if not weak:
        print("  (none — every strategy has either non-trivial share or robustness ≥ 0.2)")
    else:
        for w in weak:
            print(f"  {w}")
    print()


if __name__ == "__main__":
    main()
