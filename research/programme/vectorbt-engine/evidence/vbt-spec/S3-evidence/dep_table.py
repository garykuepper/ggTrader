"""Per-dependency table: old resolved version, new resolved version, declared pandas/numpy bounds."""
import re
import sys
from importlib import metadata

E = sys.argv[1]
declared = """ccxt defusedxml fsspec joblib matplotlib mplfinance numba numpy optuna pandas pandas-market-calendars
pandas-ta plotly psycopg2-binary pyarrow pypdf python-dotenv pytrends requests scikit-learn seaborn sqlalchemy ta
tabulate tqdm vectorbt xlrd xlsxwriter yfinance alpaca-py pydantic typer typing-extensions hmmlearn lightgbm""".split()
extra_seen = "scipy beautifulsoup4 statsmodels vectorbt-rust llvmlite exchange-calendars".split()


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def load_freeze(p):
    d = {}
    for line in open(p):
        if "==" in line:
            n, v = line.strip().split("==", 1)
            d[norm(n)] = v
    return d


old, new = load_freeze(f"{E}/05_old_stack_freeze.txt"), load_freeze(f"{E}/05_new_stack_freeze.txt")
print("| package | declared in pyproject | old (2.3.3 stack) | new (resolved) | declared pandas / numpy / numba bounds (new wheel metadata) |")
print("|---|---|---|---|---|")
for n in declared + extra_seen:
    nn = norm(n)
    try:
        reqs = metadata.requires(nn) or []
    except metadata.PackageNotFoundError:
        reqs = []
    b = []
    for r in reqs:
        if "extra ==" in r:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", r)
        if m and norm(m.group(1)) in ("pandas", "numpy", "numba"):
            b.append(f"{norm(m.group(1))}{m.group(2).split(';')[0].strip()}")
    print(f"| {n} | {'yes' if n in declared else 'transitive'} | {old.get(nn, '-')} | {new.get(nn, '-')} | {', '.join(b) or 'none declared'} |")
