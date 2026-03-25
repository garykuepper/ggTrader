import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


def chunk_list(lst: List, n: int) -> List[List]:
    """Split list into n approximately equal chunks."""
    if not lst:
        return []
    n = max(1, min(n, len(lst)))
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def register_research_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'research' subcommand."""
    parser = subparsers.add_parser(
        "research", help="Run the Grand Walk-Forward Optimization pipeline"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Train lookup window in days (default: 1095 / 3 years)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of coins to test based on live exchange volume (default: 50)",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="30d",
        choices=["24h", "7d", "30d"],
        help="Volume aggregation window for asset selection (default: 30d)",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel worker processes (default: 5)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Run sequentially instead of in parallel"
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")


def run_research(args: argparse.Namespace):
    """Executes the research pipeline in parallel by default."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    research_dir = Path(f"results/research_{timestamp}")
    research_dir.mkdir(parents=True, exist_ok=True)

    universe_path = research_dir / "top_ccxt_volume.json"

    print(
        f"\n[{datetime.now()}] Step 1: Fetching Live CCXT Universe for Research "
        f"({args.top} coins, {args.window} window)..."
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/update_universe_ccxt.py",
            "--limit",
            str(args.top),
            "--out",
            str(universe_path),
            "--window",
            args.window,
        ],
        check=True,
    )

    # Load the freshly generated symbols
    try:
        with open(universe_path, "r") as f:
            data = json.load(f)
            # Symbols might be list of strings or list of objects
            if isinstance(data[0], dict):
                symbols = [item["symbol"] for item in data]
            else:
                symbols = data
    except Exception as e:
        print(f"Error loading universe for chunking: {e}")
        return

    # Ensure -USD suffix (as required by the historical loader)
    symbols = [s if "-" in s else f"{s}-USD" for s in symbols]

    # Calculate dynamic training window
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    if args.no_parallel or args.workers <= 1:
        print(
            f"\n[{datetime.now()}] Step 2: Initiating Sequential Walk-Forward "
            f"Optimization ({start_date} to {end_date})..."
        )
        cmd = [
            sys.executable,
            "scripts/run_walk_forward_optimization.py",
            "--symbols",
            ",".join(symbols),
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--phase1",
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        subprocess.run(cmd, check=True)
    else:
        print(
            f"\n[{datetime.now()}] Step 2: Initiating Parallel Walk-Forward "
            f"Optimization ({args.workers} workers)..."
        )
        symbol_chunks = chunk_list(symbols, args.workers)

        processes = []
        log_handles = []

        for i, chunk in enumerate(symbol_chunks):
            if not chunk:
                continue

            chunk_str = ",".join(chunk)
            worker_log = research_dir / f"worker_{i + 1}.log"

            # Note: We run Phase 1 (WFO) in parallel.
            # Phase 2/3 (Portfolio Validation) requires aggregated results,
            # so they are typically run after all workers finish.
            cmd = [
                sys.executable,
                "-u",
                "scripts/run_walk_forward_optimization.py",
                "--symbols",
                chunk_str,
                "--phase1",
                "--no-progress",
            ]

            f = open(worker_log, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, bufsize=1)
            processes.append(p)
            log_handles.append(f)
            print(f"  > Launched worker {i + 1} (processing {len(chunk)} coins)...")

        print("-" * 50)
        print("All workers launched. Monitoring progress...")

        try:
            while True:
                alive = [p.poll() is None for p in processes]
                if not any(alive):
                    break

                done_count = alive.count(False)
                print(f"Status: {done_count}/{len(processes)} workers finished...", end="\r")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nTerminating research workers...")
            for p in processes:
                p.terminate()
        finally:
            for f in log_handles:
                f.close()

        print(f"\n[{datetime.now()}] All parallel workers finished.")
        print(f"Logs available in: {research_dir}")

    print(f"\n[{datetime.now()}] Research Pipeline complete.")
