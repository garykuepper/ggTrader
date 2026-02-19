"""Smoke tests for plotting utilities."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import matplotlib.pyplot as plt
from ggTrader.utils.plotting import plot_optimization_landscape


def test_plot_optimization_landscape_smoke():
    df = pd.DataFrame(
        {"param1": [1, 1, 2, 2], "param2": [10, 20, 10, 20], "Sharpe Ratio": [0.5, 0.6, 0.7, 0.8]}
    )

    # We just want to ensure it doesn't crash
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
        plot_optimization_landscape(
            df,
            params_to_plot=["param1", "param2"],
            metric_name="Sharpe Ratio",
            results_manager=None,
        )
    plt.close("all")
