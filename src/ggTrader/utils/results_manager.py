"""Results management and output persistence."""

import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from ggTrader.utils.paths import find_project_root

from .result_db_manager import ResultDBManager


class ResultsManager:
    """
    Manages the creation of timestamped result directories and the saving of
    metadata, metrics, and plots.
    """

    def __init__(self, script_name: str, results_dir: str = "results") -> None:
        self.script_name = script_name
        self.project_root = find_project_root()
        self.base_results_dir = self.project_root / results_dir
        self.run_dir = self._create_run_directory()

        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = str(uuid.uuid4())
        self.db_manager = ResultDBManager(
            log_path=self.base_results_dir / "runs_log.csv",
        )

    def _create_run_directory(self) -> Path:
        """Creates a timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.script_name}_{timestamp}"
        run_dir = self.base_results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_run_results(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves run results to a single JSON file and mirrors to the database.
        """
        if metadata is None:
            metadata = {}

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
                "_raw_config": metadata,
            },
            "strategy_parameters": params,
            "results": metrics,
        }

        output_path = self.run_dir / "run_results.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        self.db_manager.add_run(
            run_id=self.run_id,
            run_type=self.script_name,
            script_name=self.script_name,
            parameters=params,
            metadata=output_data["configuration"],
            metrics=metrics,
        )
        self.db_manager.add_metrics(self.run_id, metrics)

        return output_path

    # =========================================================================
    # Deprecated / Simple Persistence
    # =========================================================================

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """DEPRECATED: Use save_run_results instead."""
        pass

    def save_params(self, params: Dict[str, Any]) -> Path:
        """Saves optimal parameters to params.json for transfer."""
        params_path = self.run_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        return params_path

    def load_params(self, params_file_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads parameters from a JSON file."""
        with open(params_file_path, "r") as f:
            return json.load(f)

    def save_metrics(
        self, df: pd.DataFrame, filename: str = "metrics.csv", save_csv: bool = False
    ) -> Path:
        """Saves performance metrics to a CSV and mirrors WFO to DB."""
        path = self.run_dir / filename
        if save_csv:
            df.to_csv(path, index=False)

        if filename == "wfo_results.csv":
            self.db_manager.add_wfo_results(self.run_id, df)

        return path

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Prints a summary of the performance metrics to the console."""
        print("\n--- Performance Summary ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("---------------------------")

    def save_trades(self, df: pd.DataFrame) -> None:
        self.db_manager.add_trades(self.run_id, df)

    def save_equity_curve(self, series: pd.Series, filename: str = "equity_curve") -> None:
        self.db_manager.add_equity_curve(self.run_id, series)

    # =========================================================================
    # Plotting & Exports
    # =========================================================================

    def get_plot_path(self, filename: str) -> Path:
        """Returns the full path for a plot file in the plots directory."""
        return self.plots_dir / filename

    def _save_plotly_figure(self, fig: Any, path: Path, filename: str) -> None:
        """Helper to save interactive HTML and static PNG Plotly figures."""
        try:
            fig.write_html(str(path.with_suffix(".html")))
        except Exception as e:
            print(f"Warning: Could not save HTML for {filename}: {e}")

        try:
            fig.write_image(str(path.with_suffix(".png")))
        except Exception as e:
            print(f"Warning: Could not save PNG for {filename}: {e} (Install kaleido?)")

    def _save_matplotlib_figure(self, fig: Any, path: Path, filename: str) -> None:
        """Helper to save and safely close Matplotlib figures."""
        try:
            fig.savefig(str(path), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not save Matplotlib figure {filename}: {e}")

    def save_plot(self, fig: Any, filename: str) -> None:
        """Universal plot saver routing to specific backend handlers."""
        path = self.get_plot_path(filename)

        if hasattr(fig, "write_image") or hasattr(fig, "write_html"):
            self._save_plotly_figure(fig, path, filename)
        elif hasattr(fig, "savefig"):
            self._save_matplotlib_figure(fig, path, filename)
        else:
            print(f"Warning: Unknown figure type for {filename}")

    def save_vbt_dashboard(self, pf: Any, filename: str = "plots") -> None:
        """Saves a VectorBT Portfolio dashboard."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = pf.plot(subplots=["drawdowns", "value"])
            self.save_plot(fig, filename)
        except Exception as e:
            print(f"Error saving VBT dashboard: {e}")

    def save_plotly_figure(self, fig: Any, filename: str) -> None:
        """Deprecated alias for save_plot."""
        self.save_plot(fig, filename)

    def _prepare_df_for_excel(self, df: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Helper to format DataFrames and strip timezone info for Excel compatibility."""
        if isinstance(df, dict):
            df = pd.DataFrame.from_dict(df, orient="index", columns=["Value"])

        df = df.copy()

        # Strip timezone from index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Strip timezone from columns
        for col in df.columns:
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        return df

    def save_excel(self, data_dict: Dict[str, Any], filename: str = "results.xlsx") -> Path:
        """Saves multiple DataFrames into a single Excel workbook."""
        excel_path = self.run_dir / filename
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            for sheet_name, df in data_dict.items():
                clean_df = self._prepare_df_for_excel(df)
                safe_name = str(sheet_name)[:31]
                clean_df.to_excel(writer, sheet_name=safe_name)
        return excel_path

    def __str__(self) -> str:
        return f"ResultsManager(run_dir={self.run_dir})"


def get_latest_params(base_results_dir: str = "results") -> Optional[Path]:
    """Finds the latest params.json in the results directory."""
    project_root = find_project_root()
    results_path = project_root / base_results_dir
    if not results_path.exists():
        return None

    param_files = list(results_path.glob("**/params.json"))
    if not param_files:
        return None

    param_files.sort(key=lambda x: x.parent.name, reverse=True)
    return param_files[0]
