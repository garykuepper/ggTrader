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
        "--end-date",
        type=str,
        default=None,
        help="End date for the research window in YYYY-MM-DD format (default: today)",
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
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated explicit symbol list (e.g. BTC-USD,ETH-USD,TRX-USD). "
             "When set, skips the volume-based universe fetch and uses these symbols directly. "
             "Useful for small-N iteration on diagnostic coins.",
    )


class WorkerResultMissing(RuntimeError):
    """A worker exited 0 but its result file is missing or unreadable.

    Distinguishes a real plumbing failure (lost output, mid-write crash) from
    a legitimate "this batch had no winners" outcome (file exists, empty
    per_coin). The first should abort phase 2/3; the second is fine.
    """


def merge_worker_results(research_dir: Path, num_workers: int) -> int:
    """Merge per-worker result files into a single run_results.json.

    Each worker writes worker_N_results.json after phase 1. This function
    reads all of them, unions their ``strategy_parameters.per_coin`` dicts,
    and writes the merged result to ``run_results.json`` for phase 2/3.

    Returns the number of coins merged. Raises ``WorkerResultMissing`` if
    any worker's result file is missing or unreadable — the caller should
    treat that as fatal rather than silently continuing with partial data.
    Legitimate empty per_coin (worker ran, found nothing robust) is fine
    and merely contributes 0 coins to the union.
    """
    merged_per_coin: dict = {}
    merged_base: dict | None = None
    missing: list[str] = []
    empty: list[str] = []

    for n in range(1, num_workers + 1):
        worker_file = research_dir / f"worker_{n}_results.json"
        if not worker_file.exists():
            missing.append(f"{worker_file.name} (file not found)")
            continue
        try:
            with open(worker_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            missing.append(f"{worker_file.name} (unreadable: {e!r})")
            continue

        sp = data.get("strategy_parameters", {})
        per_coin = sp.get("per_coin", {})
        if not per_coin:
            empty.append(worker_file.name)

        merged_per_coin.update(per_coin)
        if merged_base is None:
            merged_base = data  # use first worker's structure as the template

    if missing:
        raise WorkerResultMissing(
            f"{len(missing)}/{num_workers} worker result file(s) missing or unreadable: "
            + ", ".join(missing)
        )

    if empty:
        # Not fatal — just surface it so the user knows which shards contributed nothing.
        print(f"  [merge] Note: {len(empty)} worker(s) had empty per_coin: {empty}")

    if merged_base is None:
        # All workers had empty per_coin — every coin failed gates everywhere.
        print("  [merge] ERROR: every worker had empty per_coin — phase 2/3 has no data.")
        return 0

    # Write merged result: patch per_coin into the template structure
    merged_base["strategy_parameters"]["per_coin"] = merged_per_coin
    # Update the symbols list in configuration to reflect all coins
    if "configuration" in merged_base:
        merged_base["configuration"]["symbols"] = list(merged_per_coin.keys())

    out_path = research_dir / "run_results.json"
    with open(out_path, "w") as f:
        json.dump(merged_base, f, indent=4)

    # Persist the merged run as a row in the ``runs`` table so the live trader's
    # state_manager.get_latest_research_run can discover it. Per-worker rows
    # already exist (one per shard), but the consolidated set of per-coin params
    # only lives in the merged JSON.
    try:
        from datetime import datetime as _dt

        from ggTrader.utils.result_db_manager import ResultDBManager

        ps_path = research_dir / "phase_stats.json"
        phase_stats = None
        if ps_path.exists():
            try:
                phase_stats = json.loads(ps_path.read_text())
            except Exception:
                phase_stats = None

        ts_str = merged_base.get("timestamp")
        try:
            ts = _dt.fromisoformat(ts_str) if ts_str else _dt.now()
        except Exception:
            ts = _dt.now()

        ResultDBManager().add_run(
            run_id=merged_base.get("run_id") or research_dir.name,
            run_type="research",
            script_name=merged_base.get("script_name", "research"),
            parameters=merged_base.get("configuration", {}).get("_raw_config", {}),
            metadata=merged_base.get("configuration", {}),
            metrics=merged_base.get("results", {}),
            pipeline_stage="research",
            asset_class=merged_base.get("asset_class")
                or merged_base.get("configuration", {})
                .get("_raw_config", {}).get("ASSET_CLASS", "crypto"),
            strategy_params=merged_base.get("strategy_parameters", {}),
            phase_stats=phase_stats,
            run_dir=str(research_dir),
            status="success",
            timestamp=ts,
        )
        print(f"  [merge] Persisted merged run to runs table ({merged_base.get('run_id')}).")
    except Exception as e:
        print(f"  [merge] WARNING: failed to upsert merged run into DB: {e}")

    n_coins = len(merged_per_coin)
    print(f"  [merge] Merged {n_coins} coins from {num_workers} workers -> {out_path.name}")
    return n_coins


def run_research(args: argparse.Namespace):
    """Executes the research pipeline in parallel by default."""
    asset_class = "crypto"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    research_dir = project_root / f"results/research/research_{timestamp}"
    research_dir.mkdir(parents=True, exist_ok=True)

    universe_path = research_dir / "top_ccxt_volume.json"
    universe_script = "scripts/update_universe_ccxt.py"

    # Same-day universe is keyed by (asset_class, snapshot_date, top, window) in
    # the universe_cache table.
    from datetime import date as _date

    from sqlalchemy import text as _text

    from ggTrader.utils.result_db_manager import ResultDBManager

    rm = ResultDBManager()
    cache_key = f"{asset_class}_top{args.top}_{args.window}"
    today_d = _date.today()

    if args.symbols:
        # Explicit symbol list overrides the volume-based universe fetch entirely.
        # Used for small-N iteration on diagnostic coins; skips DB caching since
        # the universe is user-specified, not derived from market state.
        explicit = [s.strip() for s in args.symbols.split(",") if s.strip()]
        explicit = [s if "-" in s else f"{s}-USD" for s in explicit]
        print(
            f"\n[{datetime.now()}] Step 1: Using explicit --symbols list "
            f"({len(explicit)} coins): {','.join(explicit)}"
        )
        with open(universe_path, "w") as f:
            json.dump(explicit, f, indent=2)
    else:
        cached_payload = None
        try:
            with rm.engine.connect() as conn:
                row = conn.execute(
                    _text(
                        """
                        SELECT symbols FROM universe_cache
                        WHERE asset_class = :ac AND snapshot_date = :d
                          AND symbols->>'cache_key' = :ck
                        """
                    ),
                    {"ac": asset_class, "d": today_d, "ck": cache_key},
                ).fetchone()
            if row is not None:
                cached_payload = row[0]
        except Exception:
            cached_payload = None

        if cached_payload is not None:
            print(
                f"\n[{datetime.now()}] Step 1: Using cached universe from today "
                f"(universe_cache table, {cache_key}) — skipping live fetch."
            )
            with open(universe_path, "w") as f:
                json.dump(cached_payload.get("entries", []), f, indent=2)
        else:
            print(
                f"\n[{datetime.now()}] Step 1: Fetching Live Crypto Universe for Research "
                f"({args.top} assets, {args.window} window)..."
            )
            subprocess.run(
                [
                    sys.executable,
                    universe_script,
                    "--limit",
                    str(args.top),
                    "--out",
                    str(universe_path),
                    "--window",
                    args.window,
                ],
                check=True,
            )
            try:
                with open(universe_path, "r") as f:
                    entries = json.load(f)
                payload = {"cache_key": cache_key, "entries": entries}
                with rm.engine.begin() as conn:
                    conn.execute(
                        _text(
                            """
                            INSERT INTO universe_cache (asset_class, snapshot_date, symbols)
                            VALUES (:ac, :d, CAST(:p AS JSONB))
                            ON CONFLICT (asset_class, snapshot_date) DO UPDATE
                              SET symbols = EXCLUDED.symbols, created_at = now()
                            """
                        ),
                        {"ac": asset_class, "d": today_d, "p": json.dumps(payload)},
                    )
                print(f"  Cached universe to universe_cache table ({cache_key}) for today's runs.")
            except Exception as e:
                print(f"  WARNING: failed to cache universe to DB: {e}")

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

    # Ensure -USD suffix (required by the historical loader)
    symbols = [s if "-" in s else f"{s}-USD" for s in symbols]

    # Calculate dynamic training window
    if args.end_date:
        end_date_str = args.end_date
        try:
            end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid --end-date '{end_date_str}'. Must be in YYYY-MM-DD format.")
            return
    else:
        end_date_obj = datetime.now()
        end_date_str = end_date_obj.strftime("%Y-%m-%d")

    end_date = end_date_str
    start_date = (end_date_obj - timedelta(days=args.days)).strftime("%Y-%m-%d")

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
                "--worker-id",
                str(i + 1),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
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
                dynamic_ncols=True,
                bar_format="{desc} {percentage:3.0f}%|{bar:20}| [{elapsed}]"
            )
            worker_pbars.append(pbar)

        # Worker persistent state
        worker_states = {}
        for i in range(len(processes)):
            worker_states[i] = {"msg": "", "progress": 0.0}

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

                        msg = ""
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
                                    fold_prog = (
                                        int(f_idx) / int(f_total)
                                        * (100 / int(s_total) / int(c_total))
                                    )
                                    total_progress += fold_prog

                            # Update persistent state
                            worker_states[i] = {"msg": msg, "progress": total_progress}
                        else:
                            # Use last known state if markers scrolled out of the buffer
                            msg = worker_states[i]["msg"]
                            total_progress = worker_states[i]["progress"]

                        desc_str = msg if msg else "Init..."
                        # Prevent console wrap by truncating long descriptions
                        if len(desc_str) > 40:
                             desc_str = desc_str[:37] + "..."

                        worker_pbars[i].set_description(f"W{i+1} {desc_str}")
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

    # Verify every worker exited cleanly. Subprocess.Popen does not raise on
    # non-zero exit; without this check, an OOM-killed or exception-thrown
    # worker would silently drop its 10-coin batch and phase 2/3 would run
    # on a partial universe with no signal.
    if not args.no_parallel and args.workers > 1:
        worker_failures: list[tuple[int, int, Path]] = []
        for i, p in enumerate(processes):
            rc = p.returncode
            if rc != 0:
                worker_failures.append((i + 1, rc if rc is not None else -1, log_paths[i]))
        if worker_failures:
            print("\nERROR: one or more research workers exited with a non-zero status.")
            for w_id, rc, log in worker_failures:
                print(f"  ✗ Worker {w_id}: exit code {rc}  (log: {log})")
            print(
                "Refusing to run phase 2/3 on partial data. Inspect the worker "
                "log(s) above and re-run after fixing the root cause."
            )
            sys.exit(1)

    # Merge per-worker result files into a single run_results.json before phase 2/3.
    # Without this, whichever worker wrote last would be the only one seen by phase 2/3.
    if not args.no_parallel and args.workers > 1:
        print(f"\n[{datetime.now()}] Merging worker results...")
        try:
            # actual workers launched (≤ args.workers)
            n_merged = merge_worker_results(research_dir, len(symbol_chunks))
        except WorkerResultMissing as e:
            print(f"\nERROR: {e}")
            print(
                "  Worker(s) exited 0 but their result file is missing or corrupt — "
                "this is a plumbing failure, not a no-edge result. Phase 2/3 aborted."
            )
            sys.exit(1)
        if n_merged == 0:
            print("  ERROR: merge produced 0 coins. Phase 2/3 will be skipped.")
            return

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
        "--no-progress",
        "--start-date", start_date,
        "--end-date", end_date,
    ]
    subprocess.run(final_cmd, check=True)

    print(f"\n[{datetime.now()}] Research Pipeline complete.")
