import mlflow, tempfile, pandas as pd, numpy as np, os
d=tempfile.mkdtemp(dir="/tmp/claude-1000"); mlflow.set_tracking_uri(f"sqlite:///{d}/m.db"); mlflow.set_experiment("s8")
with mlflow.start_run(run_name="fold01") as r:
    mlflow.log_params({"min_agree":3,"bb_alpha":2.0}); mlflow.log_metric("oos_sharpe",0.81); 
    mlflow.log_table(pd.DataFrame({"a":[1,2]}), "t.json")
print("OK   mlflow", mlflow.__version__, "run logged; search ->", len(mlflow.search_runs()))
