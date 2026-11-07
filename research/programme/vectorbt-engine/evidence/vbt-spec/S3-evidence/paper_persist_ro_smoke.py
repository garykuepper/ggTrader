"""READ-ONLY smoke of paper/persist.py getters (SQLAlchemy 2.0.x vs 2.1.x). Prints only types and
sha1 of repr (no account values). Engine is forced to postgresql_readonly; no init_paper_schema()."""
import hashlib
import json
import sys
import warnings

import sqlalchemy
from sqlalchemy import create_engine

warnings.simplefilter("ignore")
from ggTrader.paper import persist  # noqa: E402
from ggTrader.utils.config import get_db_connection_string  # noqa: E402

eng = create_engine(get_db_connection_string()).execution_options(postgresql_readonly=True)
persist._get_engine = lambda: eng
names = [
    "get_latest_snapshot",
    "get_earliest_snapshot",
    "get_latest_snapshot_positions",
    "get_latest_snapshot_run_date",
    "get_peak_value",
    "get_rebalance_state",
    "get_snapshot_history",
    "get_last_buy_dates",
    "get_accrued_dividend_keys",
    "get_total_dividend_accrual",
    "get_open_split_corrections",
    "get_open_split_states",
    "get_pending_orders",
]
out = {"sqlalchemy": sqlalchemy.__version__}
for n in names:
    try:
        v = getattr(persist, n)()
        r = repr(sorted(v.items(), key=str)) if isinstance(v, dict) else repr(sorted(v, key=str)) if isinstance(v, (set, list)) else repr(v)
        out[n] = {"type": type(v).__name__, "len": len(v) if hasattr(v, "__len__") else None, "sha1": hashlib.sha1(r.encode()).hexdigest()[:10]}
    except Exception as e:  # noqa: BLE001
        out[n] = {"error": repr(e)[:200]}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
