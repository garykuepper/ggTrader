import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.io as pio
import matplotlib.pyplot as plt
import uuid
from .result_db_manager import ResultDBManager


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

        # New: UUID for database tracking
        self.run_id = str(uuid.uuid4())

        # New: Database and Log Manager
        self.db_manager = ResultDBManager(
            log_path=self.base_results_dir / "runs_log.csv",
        )

    def _find_project_root(self):
        """Finds the project root by looking for pyproject.toml."""
        # Start from the location of THIS file
        current = Path(__file__).absolute().parent
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent

        # Fallback to CWD if not found via parents
        current_cwd = Path(os.getcwd()).absolute()
        for parent in [current_cwd] + list(current_cwd.parents):
            if (parent / "pyproject.toml").exists():
                return parent

        return current_cwd

    def _create_run_directory(self):
        """Creates a timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.script_name}_{timestamp}"
        run_dir = self.base_results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_run_results(
        self, params: dict, metrics: dict, metadata: dict = None
    ) -> Path:
        """
        Saves run results to a single JSON file with strict separation of concerns.

        Structure:
            {
                "run_id": ...,
                "timestamp": ...,
                "configuration": { ... },   # Metadata, Config, Consts
                "strategy_parameters": { ... }, # Signal/Strategy Params
                "results": { ... }          # Metrics
            }
        """
        if metadata is None:
            metadata = {}

        # 1. Structure the output
        output_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "script_name": self.script_name,
            "configuration": {
                "start_date": metadata.get("START_DATE"),
                "end_date": metadata.get("END_DATE"),
                "interval": metadata.get("INTERVAL"),
                "symbols_file": metadata.get("SYMBOLS_FILE"),
                "symbols": metadata.get("SYMBOLS"),
                "capital": metadata.get("START_CASH"),
                "fees": metadata.get("FEES"),
                # Include full raw config for debugging if needed, but keep top-level clean
                "_raw_config": metadata,
            },
            "strategy_parameters": params,
            "results": metrics,
        }

        output_path = self.run_dir / "run_results.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        # 2. Mirror to DB
        # Pass separated components to add_run
        self.db_manager.add_run(
            run_id=self.run_id,
            run_type=self.script_name,
            script_name=self.script_name,
            parameters=params,
            metadata=output_data["configuration"],
            metrics=metrics,
        )

        # Also mirror metrics largely
        self.db_manager.add_metrics(self.run_id, metrics)

        return output_path

    # Deprecated methods
    def save_metadata(self, metadata):
        """DEPRECATED: Use save_run_results instead."""
        pass

    def save_params(self, params):
        """Saves optimal parameters to params.json for transfer."""
        params_path = self.run_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        return params_path

    def load_params(self, params_file_path):
        """Loads parameters from a JSON file."""
        with open(params_file_path, "r") as f:
            return json.load(f)

    def save_metrics(
        self, df: pd.DataFrame, filename: str = "metrics.csv", save_csv: bool = False
    ) -> Path:
        """
        Saves performance metrics to a CSV file (optional) and mirrored to DuckDB if relevant.
        """
        path = self.run_dir / filename
        if save_csv:
            df.to_csv(path, index=False)

        # Mirror to DB if it's WFO results
        if filename == "wfo_results.csv":
            self.db_manager.add_wfo_results(self.run_id, df)
        elif filename == "sensitivity_results.csv":
            # Potentially handle sensitivity results in DB?
            # For now, CSV is fine.
            pass

        return path

    def print_summary(self, metrics: dict) -> None:
        """Prints a summary of the performance metrics to the console."""
        print("\n--- Performance Summary ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("---------------------------")

    def save_trades(self, df):
        """Saves detailed trade history to the database."""
        self.db_manager.add_trades(self.run_id, df)

    def save_equity_curve(self, series, filename="equity_curve"):
        """Saves equity curve to DB."""
        self.db_manager.add_equity_curve(self.run_id, series)

    def get_plot_path(self, filename):
        """Returns the full path for a plot file in the plots directory."""
        if not filename.endswith((".png", ".html", ".pdf", ".jpg")):
            # default extension?
            pass
        return self.plots_dir / filename

    def save_plot(self, fig, filename):
        """
        Universal plot saver for Matplotlib and Plotly figures.
        Auto-detects backend.
        """
        path = self.get_plot_path(filename)

        # 1. Plotly Figure
        if hasattr(fig, "write_image") or hasattr(fig, "write_html"):
            # Save HTML (Interactive)
            html_path = path.with_suffix(".html")
            try:
                fig.write_html(str(html_path))
            except Exception as e:
                print(f"Warning: Could not save HTML for {filename}: {e}")

            # Save PNG (Static)
            png_path = path.with_suffix(".png")
            try:
                # Requires kaleido
                fig.write_image(str(png_path))
            except Exception as e:
                print(
                    f"Warning: Could not save PNG for {filename}: {e} (Install kaleido?)"
                )

        # 2. Matplotlib Figure
        elif hasattr(fig, "savefig"):
            try:
                fig.savefig(str(path))
                plt.close(fig)  # Close to free memory
            except Exception as e:
                print(f"Warning: Could not save Matplotlib figure {filename}: {e}")

        else:
            print(f"Warning: Unknown figure type for {filename}")

    def save_vbt_dashboard(self, pf, filename="plots"):
        """
        Saves a VectorBT Portfolio dashboard (Equity, Drawdowns, Trades).
        Args:
            pf: vectorbt.Portfolio object
            filename: base filename
        """
        try:
            # 1. Main Dashboard (Equity, Drawdown, etc.)
            # pf.plot() returns a Plotly FigureWidget
            fig = pf.plot(subplots=["drawdowns", "value"])
            self.save_plot(fig, filename)

            # 2. Trade Signals (Optional but useful)
            # fig_trades = pf.plot_trade_signals()
            # self.save_plot(fig_trades, f"{filename}_trades")

        except Exception as e:
            print(f"Error saving VBT dashboard: {e}")

    def save_plotly_figure(self, fig, filename):
        """Deprecated alias for save_plot."""
        self.save_plot(fig, filename)

    def save_excel(self, data_dict, filename="results.xlsx"):
        """Saves multiple DataFrames into a single Excel workbook."""
        excel_path = self.run_dir / filename
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            for sheet_name, df in data_dict.items():
                if isinstance(df, dict):
                    df = pd.DataFrame.from_dict(df, orient="index", columns=["Value"])

                safe_name = sheet_name[:31]
                df = df.copy()
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                for col in df.columns:
                    if pd.api.types.is_datetime64tz_dtype(df[col]):
                        df[col] = df[col].dt.tz_localize(None)

                df.to_excel(writer, sheet_name=safe_name)
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
