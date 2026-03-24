"""CLI Command: Research Pipeline (The Grand WFO)"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def register_research_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'research' subcommand."""
    parser = subparsers.add_parser("research", help="Run the Grand Walk-Forward Optimization pipeline")
    
    parser.add_argument(
        "--days", type=int, default=1095, 
        help="Train lookup window in days (default: 1095 / 3 years)"
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of coins to test based on live 24h volume (default: 50)"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar"
    )

def run_research(args: argparse.Namespace):
    """Executes the research pipeline."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    research_dir = Path(f"results/research_{timestamp}")
    research_dir.mkdir(parents=True, exist_ok=True)
    
    universe_path = research_dir / "top_ccxt_volume.json"
    
    print(f"\n[{datetime.now()}] Step 1: Fetching Live CCXT Universe for Research ({args.top} coins)...")
    subprocess.run([
        sys.executable, "scripts/update_universe_ccxt.py", 
        "--limit", str(args.top), 
        "--out", str(universe_path)
    ], check=True)
    
    # We call run_walk_forward_optimization.py, but passing our CCXT universe file
    print(f"\n[{datetime.now()}] Step 2: Initiating Grand Walk-Forward Optimization...")
    cmd = [
        sys.executable, "scripts/run_walk_forward_optimization.py",
        "--symbols-file", str(universe_path),
        "--phase1", "--phase2", "--phase3"
    ]
    if args.no_progress:
        cmd.append("--no-progress")
        
    subprocess.run(cmd, check=True)
    print(f"\n[{datetime.now()}] Research Pipeline complete.")
