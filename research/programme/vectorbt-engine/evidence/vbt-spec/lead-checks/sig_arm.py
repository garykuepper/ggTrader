import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S5-evidence")
import s5_signals as s5
arm, out = sys.argv[1], sys.argv[2]
close = pd.read_parquet("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S5-evidence/tape_close.parquet"); volume = pd.read_parquet("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S5-evidence/tape_volume.parquet")
rng = np.random.default_rng(99); cols = sorted(rng.choice(close.columns, 40, replace=False))
close, volume = close[cols], volume[cols]
ind = s5.load_ggt("/home/flynn/ggTrader/src/ggTrader/lab/strategies/indicators.py") if arm == "ggt" else None
v = s5.run(arm, close, volume, ind); e, x = s5.ensemble(v)
pd.concat({"e": e.astype(bool), "x": x.astype(bool)}, axis=1).to_parquet(out)
print(arm, pd.__version__, close.shape, int(e.values.sum()), int(x.values.sum()))
