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
                # Replace KrakenHistoricalData
                line = line.replace('from utils.KrakenHistoricalData import', 'from ggTrader.data.KrakenHistoricalData import')
                # Replace KrakenData
                line = line.replace('from utils.KrakenData import', 'from ggTrader.data.KrakenData import')
                # Replace Signals (assuming from ggTrader.Signals)
                line = line.replace('from ggTrader.Signals import', 'from ggTrader.indicators.Signals import')
                # Replace Utils (assuming from ggTrader.Utils)
                line = line.replace('from ggTrader.Utils import', 'from ggTrader.utils.Utils import')
                # Replace Portfolio
                line = line.replace('from ggTrader.Portfolio import', 'from ggTrader.core.Portfolio import')
                # Replace Screener
                line = line.replace('from ggTrader.Screener import', 'from ggTrader.core.Screener import')
                
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

notebooks_dir = 'notebooks'
for filename in os.listdir(notebooks_dir):
    if filename.endswith('.ipynb'):
        update_notebook(os.path.join(notebooks_dir, filename))
