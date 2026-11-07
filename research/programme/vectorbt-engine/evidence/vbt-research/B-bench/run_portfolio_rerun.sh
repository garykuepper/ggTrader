#!/bin/bash
# second, interleaved pass for variance estimate
cd /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/B-bench
for pass in 2 3; do
 for cfg in "venv0285 none" "venv120 numba" "venv120 rust"; do set -- $cfg
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/$1/bin/python -I /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/B-bench/bench_portfolio.py --engine $2 --repeat 5 --out /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/B-bench/portfolio_$1_$2_pass$pass.json > /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/B-bench/portfolio_$1_$2_pass$pass.txt 2>&1
 done
done
echo done > /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/B-bench/portfolio_rerun.done
