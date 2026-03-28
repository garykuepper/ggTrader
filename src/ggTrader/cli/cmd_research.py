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
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    research_dir = project_root / f"results/research/research_{timestamp}"
    research_dir.mkdir(parents=True, exist_ok=True)

    universe_path = research_dir / "top_ccxt_volume.json"

    # Check for a same-day cached universe file (keyed by date + top-N + window).
    today = datetime.now().strftime("%Y%m%d")
    cache_name = f"universe_cache_{today}_top{args.top}_{args.window}.json"
    universe_cache_path = project_root / "results" / "research" / cache_name

    if universe_cache_path.exists():
        print(
            f"\n[{datetime.now()}] Step 1: Using cached universe from today "
            f"({universe_cache_path.name}) — skipping live fetch."
        )
        import shutil
        shutil.copy(universe_cache_path, universe_path)
    else:
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
        # Cache for subsequent runs today
        import shutil
        shutil.copy(universe_path, universe_cache_path)
        print(f"  Cached universe to {universe_cache_path.name} for today's runs.")

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
            "--run-dir",
            str(research_dir.absolute()),
            "--pipeline-stage",
            "research",
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
        log_paths = []

        for i, chunk in enumerate(symbol_chunks):
            if not chunk:
                continue

            chunk_str = ",".join(chunk)
            worker_log = research_dir / f"worker_{i + 1}.log"
            log_paths.append(worker_log)

            cmd = [
                sys.executable,
                "-u",
                "scripts/run_walk_forward_optimization.py",
                "--symbols",
                chunk_str,
                "--phase1",
                "--no-progress",
                "--run-dir",
                str(research_dir.absolute()),
                "--pipeline-stage",
                "research",
            ]

            f = open(worker_log, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, bufsize=1)
            processes.append(p)
            log_handles.append(f)
            print(f"  > Launched worker {i + 1} (processing {len(chunk)} coins)...")

        print("-" * 50)
        print("All workers launched. Monitoring progress...")

        import re
        from tqdm import tqdm

        # Regex patterns for progress extraction
        re_sym = re.compile(r"--- Optimizing (\S+) \((\d+)/(\d+)\) ---")
        re_combo = re.compile(r"Testing: \S+ \((\d+)/(\d+)\)")
        re_fold = re.compile(r"Fold (\d+)/(\d+) done")

        # Main progress bar for finished workers
        main_pbar = tqdm(
            total=len(processes),
            desc="Total Workers",
            unit="worker",
            position=0,
            leave=True
        )

        # Per-worker progress bars
        worker_pbars = []
        for i in range(len(processes)):
            pbar = tqdm(
                total=100,
                desc=f"Worker {i+1}",
                position=i + 1,
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
            )
            worker_pbars.append(pbar)

        try:
            while True:
                alive = [p.poll() is None for p in processes]
                done_count = alive.count(False)
                main_pbar.n = done_count
                main_pbar.refresh()

                if not any(alive):
                    break

                # Update per-worker status by reading log tails
                for i, log_path in enumerate(log_paths):
                    if not alive[i]:
                        worker_pbars[i].set_description(f"Worker {i+1}: Finished")
                        worker_pbars[i].n = 100
                        worker_pbars[i].refresh()
                        continue

                    try:
                        # Read last 20 lines of the log
                        with open(log_path, "r", encoding="utf-8") as lf:
                            # Use offset to read near the end for performance
                            lf.seek(0, 2)
                            size = lf.tell()
                            lf.seek(max(0, size - 2048))
                            tail = lf.read()

                        # Extract progress state
                        sym_match = list(re_sym.finditer(tail))
                        combo_match = list(re_combo.finditer(tail))
                        fold_match = list(re_fold.finditer(tail))

                        msg = f"Worker {i+1}: "
                        total_progress = 0.0

                        if sym_match:
                            s_name, s_idx, s_total = sym_match[-1].groups()
                            msg += f"[{s_name} {s_idx}/{s_total}] "
                            # Coin progress (e.g. 1/5 = 20%)
                            total_progress += (int(s_idx) - 1) / int(s_total) * 100
                            
                            if combo_match:
                                c_idx, c_total = combo_match[-1].groups()
                                msg += f"C{c_idx}/{c_total} "
                                # Combo contribution within current coin
                                combo_prog = (int(c_idx) - 1) / int(c_total) * (100 / int(s_total))
                                total_progress += combo_prog

                                if fold_match:
                                    f_idx, f_total = fold_match[-1].groups()
                                    msg += f"F{f_idx}/{f_total}"
                                    # Fold contribution within current combo
                                    fold_prog = int(f_idx) / int(f_total) * (100 / int(s_total) / int(c_total))
                                    total_progress += fold_prog

                        worker_pbars[i].set_description(msg)
                        worker_pbars[i].n = min(99, total_progress) # Save 100 for true finish
                        worker_pbars[i].refresh()
                    except (OSError, ValueError, IndexError):
                        pass

                time.sleep(2)
        except KeyboardInterrupt:
            print("\nTerminating research workers...")
            for p in processes:
                p.terminate()
        finally:
            main_pbar.close()
            for pbar in worker_pbars:
                pbar.close()
            # Move cursor past the closed progress bars
            print("\n" * (len(processes) + 1))
            for f in log_handles:
                f.close()

    print(f"\n[{datetime.now()}] All parallel workers finished.")
    print(f"Intermediate WFO results available in: {research_dir}")

    # Step 3: Global Validation & Recent Data Performance
    # We now run Phase 2 & 3 ONCE using the accumulated results dir
    print(
        f"\n[{datetime.now()}] Step 3: Initiating Full Training/Test Validation (Phase 2) "
        "and YTD Performance (Phase 3)..."
    )
    # The run_walk_forward_optimization.py script knows how to find its own results 
    # if we point it to the same run-dir and symbols-file.
    final_cmd = [
        sys.executable,
        "scripts/run_walk_forward_optimization.py",
        "--symbols-file", str(universe_path.absolute()),
        "--phase2", 
        "--phase3",
        "--run-dir", str(research_dir.absolute()),
        "--pipeline-stage", "research",
        "--no-progress"
    ]
    subprocess.run(final_cmd, check=True)

    print(f"\n[{datetime.now()}] Research Pipeline complete.")
