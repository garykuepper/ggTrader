import json
import os
import re

def update_notebook(filepath):
    print(f"Updating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                original_line = line
                
                # --- Kraken Data ---
                line = line.replace('from utils.KrakenHistoricalData import', 'from ggTrader.data.kraken.historical_data import')
                line = line.replace('from ggTrader.data.KrakenHistoricalData import', 'from ggTrader.data.kraken.historical_data import')
                
                line = line.replace('from utils.KrakenData import', 'from ggTrader.data.kraken.data_manager import')
                line = line.replace('from ggTrader.data.KrakenData import', 'from ggTrader.data.kraken.data_manager import')
                
                # --- Core ---
                line = line.replace('from ggTrader.Portfolio import', 'from ggTrader.core.portfolio import')
                line = line.replace('from ggTrader.Screener import', 'from ggTrader.core.screener import')
                line = line.replace('from ggTrader.Backtest import', 'from ggTrader.core.backtest import')
                line = line.replace('from ggTrader.Trading import', 'from ggTrader.core.trading import')
                line = line.replace('from ggTrader.Position import', 'from ggTrader.core.position import')
                
                # --- Indicators ---
                line = line.replace('from ggTrader.Signals import', 'from ggTrader.indicators.signals import')
                
                # --- Utils ---
                line = line.replace('from ggTrader.Utils import', 'from ggTrader.utils.utils import')
                
                # Handle direct imports if any
                line = line.replace('import ggTrader.Portfolio', 'import ggTrader.core.portfolio')
                line = line.replace('import ggTrader.Trading', 'import ggTrader.core.trading')

                if line != original_line:
                    changed = True
                new_source.append(line)
            cell['source'] = new_source

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {filepath}")
    else:
        print(f"No changes needed for {filepath}")

# Paths (absolute for safety)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
notebooks_dir = os.path.join(project_root, 'notebooks')

if os.path.exists(notebooks_dir):
    for filename in os.listdir(notebooks_dir):
        if filename.endswith('.ipynb'):
            update_notebook(os.path.join(notebooks_dir, filename))
else:
    print(f"Notebooks directory not found: {notebooks_dir}")
