import json
import os

nb_path = "research/notebooks/walkforward_optimization.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r") as f:
        nb = json.load(f)

    # Standardize Cell 1 (Imports)
    nb["cells"][1]["source"] = [
        "import sys\n",
        "import os\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import vectorbt as vbt\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from tabulate import tabulate\n",
        "\n",
        "# Ensure project root is in path\n",
        "project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'src'))\n",
        "if project_root not in sys.path:\n",
        "    sys.path.append(project_root)\n",
        "\n",
        "from ggTrader.utils.setup import load_data_and_setup\n",
        "from ggTrader.core.fast_backtest import FastBacktest\n",
        "from ggTrader.utils.utils import make_end_anchored_tscv, plot_cv_indices",
    ]

    # Standardize Cell 2 (Configuration)
    nb["cells"][2]["source"] = [
        "# --- Configuration ---\n",
        "CONSTANTS = {\n",
        '    "SYMBOLS": None,  # Set to a list like ["BTC", "ETH"] to override JSON\n',
        '    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",\n',
        '    "INTERVAL": "4h",\n',
        '    "START_DATE": "2023-01-01",\n',
        '    "END_DATE": "2025-06-01",\n',
        '    "N_SPLITS": 3,\n',
        '    "TEST_RATIO": 0.334,\n',
        '    "START_CASH": 10000,\n',
        "}\n",
        "\n",
        'print("WFO Configuration loaded.")',
    ]

    # Standardize Cell 3 (Data Loading)
    nb["cells"][3]["source"] = [
        'print("Loading data...")\n',
        "try:\n",
        "    data_df = load_data_and_setup(CONSTANTS)\n",
        '    print(f"Loaded {len(data_df)} rows for {len(data_df.columns.levels[0])} symbols.")\n',
        "except Exception as e:\n",
        '    print(f"Error loading data: {e}")',
    ]

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook standardized successfully.")
else:
    print("Notebook path not found.")
