import json
from pathlib import Path


def patch_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 0: Markdown instructions
    nb["cells"][0]["source"] = [
        "# Parameter Sensitivity Explorer\n",
        "\n",
        "This notebook performs a **vectorized grid search** using the `ggTrader` orchestrator api. It visualizes the profitability landscape to find robust parameter regions.\n",
        "\n",
        "### Exporting with Plots\n",
        "To ensure plots are included in your export:\n",
        "1. **HTML**: Use `File > Save and Export Notebook As... > HTML`. The `notebook_connected` renderer is enabled by default below.\n",
        "2. **PDF/WebPDF**: Use `File > Save and Export Notebook As... > WebPDF`. This is the most reliable way to capture interactive plots as static images in a PDF.",
    ]

    # Cell 1: Code initialization
    # Looking for the cell with 'import vectorbt as vbt'
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any(
            "import vectorbt as vbt" in line for line in cell["source"]
        ):
            new_source = []
            for line in cell["source"]:
                if "import plotly.graph_objects as go" in line:
                    new_source.append(line)
                    new_source.append("import plotly.io as pio\n")
                elif "from tabulate import tabulate" in line:
                    new_source.append(line)
                    new_source.append("\n")
                    new_source.append("# Configure Plotly renderer for export compatibility\n")
                    new_source.append(
                        "# 'notebook' embeds Plotly.js directly in the file (safest for export)\n"
                    )
                    new_source.append('pio.renderers.default = "notebook"\n')
                    new_source.append("\n")
                    new_source.append("# Ensure VectorBT uses the default plotly renderer\n")
                    new_source.append("vbt.settings.plotting['layout']['template'] = 'vbt'\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            break

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    patch_notebook(
        r"C:\Users\gkuep\PycharmProjects\ggTrader\notebooks\parameter_sensitivity_explorer.ipynb"
    )
    print("Notebook patched successfully.")
