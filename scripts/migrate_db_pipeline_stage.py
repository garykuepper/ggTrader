"""One-time migration script to update pipeline_stage for older results."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from sqlalchemy import text
from ggTrader.utils.result_db_manager import ResultDBManager

def migrate():
    print("Starting DB migration...")
    db = ResultDBManager()
    
    queries = [
        ("UPDATE runs SET pipeline_stage = 'research' WHERE pipeline_stage is null AND run_type IN ('wfo', 'research', 'run_wfo', 'run_wfo_per_coin_multi_strategy', 'run_wfo_per_coin');",),
        ("UPDATE runs SET pipeline_stage = 'backtest' WHERE pipeline_stage is null AND run_type IN ('backtest_simulation', 'run_backtest', 'backtest');",),
        ("UPDATE runs SET pipeline_stage = 'production' WHERE pipeline_stage is null AND run_type IN ('production_run', 'production', 'recalibration');",),
        ("UPDATE runs SET pipeline_stage = 'trade' WHERE pipeline_stage is null AND run_type IN ('live_trader', 'trade', 'execution_engine');",),
    ]
    
    try:
        with db.engine.begin() as conn:
            for q in queries:
                result = conn.execute(text(q[0]))
                print(f"Updated {result.rowcount} rows with rule: {q[0][:50]}...")
        print("Migration complete!")
    except Exception as e:
        print(f"Migration error: {e}")
        
if __name__ == '__main__':
    migrate()
