"""Dump scenarios.run_all() for engine ARGV[1] (rust|numba) to ARGV[2]; run from the repo copy."""
import json, sys, warnings
warnings.simplefilter("ignore")
from ggTrader.engine import config
from tests.engine import scenarios
config.set_engine(sys.argv[1])
res = scenarios.run_all()
def conv(o):
    import numpy as np
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if hasattr(o, "tolist"): return o.tolist()
    return o
json.dump(conv(res), open(sys.argv[2], "w"))
