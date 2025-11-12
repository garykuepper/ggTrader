import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit
import math
import pandas as pd
import vectorbt as vbt

# -------Train Splits
def make_end_anchored_tscv(n_samples, n_splits, test_ratio):
    """
    Create an end-anchored TimeSeriesSplit with a desired test_ratio per sliding window.
    test_ratio: fraction of each window used for testing (0 < test_ratio < 1)
    """
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be in (0, 1).")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    k = (1 - test_ratio) / test_ratio  # train_size : test_size ratio
    # Maximal integer test_size that fits all folds
    test_size = int(math.floor(n_samples / (n_splits + k)))
    if test_size < 1:
        raise ValueError("Not enough samples for the requested n_splits and test_ratio.")

    # Compute train cap to enforce ratio; round to nearest int
    max_train_size = int(round(k * test_size))
    if max_train_size < 1:
        max_train_size = 1  # safety

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        max_train_size=max_train_size
    )
    return tscv, test_size, max_train_size


def plot_cv_indices(cv, X, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    cmap_cv = plt.cm.coolwarm

    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        indices = np.empty(len(X))
        indices[:] = np.nan
        indices[tr] = 0  # Train indices
        indices[tt] = 1  # Test indices

        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker='_',
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2
        )

        # Optional: annotate ranges
        ax.text(tt[0] * .7, ii + 0.7,
                f"Train:{tr[0]}–{tr[-1]} | Test:{tt[0]}–{tt[-1]}",
                fontsize=8)

    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=[f'Fold {i + 1}' for i in range(n_splits)],
        xlabel='Sample index',
        ylabel='CV iteration',
        title='TimeSeriesSplit (Sliding Window) Visualization'
    )
    ax.set_xlim([0, len(X)])
    ax.set_ylim([0, n_splits + 0.2])
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Legend mapping exactly matches 0 (train) and 1 (test)
    legend_elements = [
        Patch(color=plt.cm.coolwarm(0.0), label='Training set'),
        Patch(color=plt.cm.coolwarm(1.0), label='Testing set'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

#
def periods_per_year_from_interval(interval: str) -> int:
    # Handles "4h", "1h", "1d" and similar
    if interval.endswith("h"):
        hours = int(interval[:-1])
        per_day = 24 // max(1, hours)
        return per_day * 365
    if interval.endswith("d"):
        days = int(interval[:-1])
        per_day = 1 // max(1, days) if days > 0 else 1
        return per_day * 365
    # Fallbacks for your common choices
    mapping = {"4h": 6 * 365, "1h": 24 * 365, "1d": 365}
    return mapping.get(interval, 6 * 365)


def reassign_columns_value(stats_df: pd.DataFrame, trial_num: int, level: int = 0):
    new_level_0 = stats_df.columns.levels[level].to_numpy().copy()
    new_level_0[:] = trial_num
    new_levels = [pd.Index(new_level_0), stats_df.columns.levels[1]]
    new_columns = stats_df.columns.set_levels(new_levels)

    # Assign back
    stats_df.columns = new_columns

    stats_df.columns = stats_df.columns.set_names(['id', 'symbol'])
    return stats_df


def assign_column_labels(stats_df: pd.DataFrame, labels: list):
    return stats_df.rename_axis(labels, axis=1)


def stats_df_to_wide(df: pd.DataFrame, trial_num: int):
    df = reassign_columns_value(df, trial_num, level=0)
    df = assign_column_labels(df, labels=['id', 'symbol'])
    df = df.T.reset_index()
    return convert_cols_to_numeric(df)


def convert_cols_to_numeric(df: pd.DataFrame):
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')
    df['Period'] = pd.to_timedelta(df['Period'], errors='coerce')
    duration_str = 'Duration'
    for col in df.columns:
        # Skip columns we already converted to datetime/timedelta
        if col in ['Start', 'End', 'Period']:
            continue

        # Skip columns that are already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if duration_str in col:
            df[col] = pd.to_timedelta(df[col], errors='coerce')
            continue
        # Try converting to numeric, catching exceptions
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')

        except (ValueError, TypeError):
            # If conversion fails, leave the column as-is
            pass
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df



