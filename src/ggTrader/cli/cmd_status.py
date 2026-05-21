"""CLI Command: ggt status — show live research pipeline progress."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def register_status_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("status", help="Show live research pipeline progress")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to research run directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh every 30 s until all workers finish",
    )


def _find_latest_research_dir() -> Optional[Path]:
    from ggTrader.utils.paths import find_project_root

    base = find_project_root() / "results" / "research"
    if not base.exists():
        return None
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("research_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_elapsed_secs(elapsed_str: str) -> Optional[int]:
    """Parse 'H:MM:SS' or 'M:SS' elapsed string into total seconds."""
    try:
        parts = elapsed_str.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _estimate_remaining_secs(log_path: Path) -> Optional[int]:
    """
    Estimate remaining seconds for a worker.

    Priority:
    1. Most recent "WFO complete in Xs | ETA H:MM:SS" line — emitted after each coin
       and already accounts for all remaining coins via a running average.
    2. Fall back to the per-combo ETA in the tail (covers only the current coin).
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read()
    except OSError:
        return None

    # "WFO complete" ETA: emitted after every coin, already a full-worker estimate
    wfo_complete = re.findall(r"WFO complete in \d+s \| ETA (\S+) \(est\.", tail)
    if wfo_complete:
        return _parse_elapsed_secs(wfo_complete[-1])

    # Fallback: per-combo ETA (covers only current coin — shown when on coin 1)
    eta_matches = re.findall(r"ETA (\S+) \(est\. .+?\)", tail)
    if eta_matches:
        return _parse_elapsed_secs(eta_matches[-1])

    return None


def _worker_summary(log_path: Path) -> tuple[str, Optional[int]]:
    """Return (one-line status string, estimated remaining seconds or None)."""
    if not log_path.exists():
        return "not started", None

    try:
        # Read last 4 KB for efficiency
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read()
    except OSError:
        return "unreadable", None

    # Did it finish?
    if "Requested WFO phases completed" in tail or "WFO Results saved to" in tail:
        coins = re.findall(r"Optimizing (\S+) \((\d+)/(\d+)\)", tail)
        if coins:
            last = coins[-1]
            return f"DONE ({last[0]} was last, {last[1]}/{last[2]} coins)", None
        return "DONE", None

    # Current coin
    coin_matches = re.findall(r"Optimizing (\S+) \((\d+)/(\d+)\)", tail)
    combo_matches = re.findall(r"Testing: (\S+) \((\d+)/(\d+)\)", tail)
    fold_matches = re.findall(r"Fold (\d+)/(\d+) done", tail)
    eta_matches = re.findall(r"ETA (\S+) \(est\. (.+?)\)", tail)

    parts = []
    if coin_matches:
        sym, ci, ct = coin_matches[-1]
        parts.append(f"coin {ci}/{ct} ({sym})")
    if combo_matches:
        name, xi, xt = combo_matches[-1]
        parts.append(f"combo {xi}/{xt} ({name})")
    if fold_matches:
        fi, ft = fold_matches[-1]
        parts.append(f"fold {fi}/{ft}")
    if eta_matches:
        eta, wall = eta_matches[-1]
        parts.append(f"ETA {eta} (~{wall})")

    remaining_secs = _estimate_remaining_secs(log_path)
    return "  |  ".join(parts) if parts else "starting...", remaining_secs


def _count_selected_strategies(run_dir: Path) -> dict:
    """Count strategy selections from run_results.json if present."""
    import json

    rj = run_dir / "run_results.json"
    if not rj.exists():
        return {}
    try:
        with open(rj) as f:
            data = json.load(f)
        per_coin = data.get("strategy_parameters", {}).get("per_coin", {})
        counts: dict = {}
        for info in per_coin.values():
            s = info.get("best_strategy", "unknown")
            e = info.get("best_exit", "")
            key = f"{s}+{e}" if e else s
            counts[key] = counts.get(key, 0) + 1
        return counts
    except Exception:
        return {}


def run_status(args: argparse.Namespace) -> None:
    import time

    try:
        import psutil

        _psutil = True
    except ImportError:
        _psutil = False

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_research_dir()
    if run_dir is None:
        print("No research run directory found. Run `ggt research` first.")
        return

    worker_logs = sorted(run_dir.glob("worker_*.log"))

    def _show() -> bool:
        """Print status, return True if all workers are done."""

        print(f"\n{'=' * 60}")
        print(f"  Research Run: {run_dir.name}")
        print(f"{'=' * 60}")

        # Worker process check
        if _psutil:
            alive = 0
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmd = " ".join(proc.info["cmdline"] or [])
                    if "run_walk_forward_optimization" in cmd and str(run_dir) in cmd:
                        alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            total_workers = len(worker_logs)
            done_count = total_workers - alive
            print(f"  Workers: {alive} running, {done_count}/{total_workers} finished\n")

        all_done = True
        max_remaining_secs: Optional[int] = None
        for log in worker_logs:
            summary, remaining_secs = _worker_summary(log)
            done = summary.startswith("DONE")
            if not done:
                all_done = False
                if remaining_secs is not None:
                    if max_remaining_secs is None or remaining_secs > max_remaining_secs:
                        max_remaining_secs = remaining_secs
            icon = "[done]" if done else "[running]"
            print(f"  {icon} {log.name}: {summary}")

        # Overall ETA — max remaining time across all running workers
        if max_remaining_secs is not None:
            if max_remaining_secs > 0:
                h, rem = divmod(max_remaining_secs, 3600)
                m, s = divmod(rem, 60)
                eta_str = f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"
                wall = datetime.now().timestamp() + max_remaining_secs
                wall_str = datetime.fromtimestamp(wall).strftime("%I:%M %p").lstrip("0")
                print(f"\n  Overall ETA: ~{eta_str} (est. {wall_str})")
            else:
                print("\n  Overall ETA: finishing soon...")

        # run_results.json strategy counts (written incrementally)
        strategy_counts = _count_selected_strategies(run_dir)
        if strategy_counts:
            print("\n  Strategies selected so far:")
            for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
                print(f"    {strat}: {count} coin(s)")

        # Final report
        report = run_dir / "research_report.md"
        if report.exists():
            print(f"\n  Report ready: {report}")
            plots = list((run_dir / "plots").glob("*.png")) if (run_dir / "plots").exists() else []
            if plots:
                print(f"  Plots: {len(plots)} PNG(s) in {run_dir / 'plots'}")

        print(f"{'=' * 60}\n")
        return all_done

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                done = _show()
                if done:
                    print("All workers complete.")
                    break
                print("  (refreshing every 30s — Ctrl+C to exit)\n")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        _show()
