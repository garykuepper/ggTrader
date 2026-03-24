"""
Merge parallel WFO results from multiple worker directories into a single parameter JSON.
"""

import json
import glob
import os
from pathlib import Path

def merge_results():
    pattern = "results/run_wfo_per_coin_multi_strategy_20260324_0804*"
    dirs = glob.glob(pattern)
    merged_params = {}
    
    print(f"Found {len(dirs)} worker result directories.")
    
    for d in dirs:
        results_path = Path(d) / "run_results.json"
        if not results_path.exists():
            continue
            
        with open(results_path, "r") as f:
            data = json.load(f)
            per_coin = data.get("strategy_parameters", {}).get("per_coin", {})
            print(f"  {d}: {len(per_coin)} symbols.")
            merged_params.update(per_coin)
            
    output_file = "data/iteration_7_best_params_top_50.json"
    with open(output_file, "w") as f:
        json.dump(merged_params, f, indent=4)
        
    print(f"\nSuccessfully merged {len(merged_params)} symbols into {output_file}.")
    
    # Simple scorecard
    print("\n--- Iteration 7 Unicorn Scorecard (Best Performers) ---")
    scorecard = []
    for symbol, r in merged_params.items():
        scorecard.append((
            symbol, 
            r.get('best_strategy'), 
            r.get('best_exit'),
            r.get('robustness_score', 0.0)
        ))
        
    # Sort by robustness score
    scorecard.sort(key=lambda x: x[3] if x[3] is not None else -999, reverse=True)
    
    for s, ent, exi, score in scorecard:
        s_str = f"{s:12}"
        ent_str = f"{str(ent):15}"
        exi_str = f"{str(exi):12}"
        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {s_str} | {ent_str} | {exi_str} | {score_str}")

if __name__ == "__main__":
    merge_results()
