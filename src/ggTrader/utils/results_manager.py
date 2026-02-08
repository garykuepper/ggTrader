import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.io as pio

class ResultsManager:
    """
    Manages the creation of timestamped result directories and the saving of 
    metadata, metrics, and plots.
    """
    def __init__(self, script_name, results_dir="results"):
        self.script_name = script_name
        self.project_root = self._find_project_root()
        self.base_results_dir = self.project_root / results_dir
        self.run_dir = self._create_run_directory()
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _find_project_root(self):
        """Finds the project root by looking for pyproject.toml."""
        current = Path(os.getcwd()).absolute()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        return current

    def _create_run_directory(self):
        """Creates a timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.script_name}_{timestamp}"
        run_dir = self.base_results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_metadata(self, metadata):
        """Saves run metadata (parameters, script info) to run_metadata.json."""
        meta_path = self.run_dir / "run_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        return meta_path

    def save_params(self, params):
        """Saves optimal parameters to params.json for transfer."""
        params_path = self.run_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        return params_path

    def load_params(self, params_file_path):
        """Loads parameters from a JSON file."""
        with open(params_file_path, 'r') as f:
            return json.load(f)

    def save_metrics(self, df, filename="metrics.csv"):
        """Saves performance metrics to a CSV file."""
        path = self.run_dir / filename
        df.to_csv(path, index=False)
        return path

    def get_plot_path(self, filename):
        """Returns the full path for a plot file in the plots directory."""
        if not filename.endswith(('.png', '.html', '.pdf')):
            filename += '.png'
        return self.plots_dir / filename

    def save_plotly_figure(self, fig, filename):
        """
        Saves a Plotly figure as both interactive HTML and static PNG.
        """
        html_path = self.get_plot_path(filename.replace('.png', '') + '.html')
        png_path = self.get_plot_path(filename.replace('.html', '') + '.png')
        
        # Save interactive HTML
        fig.write_html(str(html_path))
        
        # Save static PNG (requires kaleido)
        try:
            fig.write_image(str(png_path))
        except Exception as e:
            print(f"Warning: Could not save static image for {filename}: {e}")
            
        return html_path, png_path

    def save_excel(self, data_dict, filename="results.xlsx"):
        """
        Saves multiple DataFrames into a single Excel workbook with multiple sheets.
        
        Args:
            data_dict (dict): Dictionary mapping {sheet_name: DataFrame}
            filename (str): Output filename.
        """
        excel_path = self.run_dir / filename
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for sheet_name, df in data_dict.items():
                if isinstance(df, dict):
                    df = pd.DataFrame.from_dict(df, orient='index', columns=['Value'])
                
                # Trim sheet name to max 31 characters (Excel limit)
                safe_name = sheet_name[:31]

                # Strip timezones for Excel compatibility (xlsxwriter requirement)
                df = df.copy()
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                for col in df.columns:
                    if pd.api.types.is_datetime64tz_dtype(df[col]):
                        df[col] = df[col].dt.tz_localize(None)

                df.to_excel(writer, sheet_name=safe_name)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[safe_name]
                for i, col in enumerate(df.columns):
                    column_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.set_column(i + 1, i + 1, column_len)
                # Adjust index column as well
                index_len = max(df.index.astype(str).str.len().max(), 10) + 2
                worksheet.set_column(0, 0, index_len)

        return excel_path

    def __str__(self):
        return f"ResultsManager(run_dir={self.run_dir})"

def get_latest_params(base_results_dir="results"):
    """Finds the latest params.json in the results directory."""
    results_path = Path(base_results_dir).absolute()
    if not results_path.exists():
        return None
    
    param_files = list(results_path.glob("**/params.json"))
    if not param_files:
        return None
    
    # Sort by directory name (which has timestamp)
    param_files.sort(key=lambda x: x.parent.name, reverse=True)
    return param_files[0]
