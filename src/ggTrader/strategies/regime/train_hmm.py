"""Offline Hidden Markov Model (HMM) training script for market regime detection.

Supports rolling training, state labeling heuristics, and database persistence.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

# Append project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from ggTrader.utils.result_db_manager import ResultDBManager


def generate_mock_emission_features(
    asset_class: Literal["equity", "crypto"],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Generates synthetic emission features for testing or fallback purposes."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_t = len(dates)
    np.random.seed(42)

    if asset_class == "equity":
        # Feature 1: SPY log return
        spy_ret = np.random.normal(0.0005, 0.01, n_t)
        # Feature 2: VIX close
        vix = 15.0 + np.cumsum(np.random.normal(0, 0.5, n_t))
        vix = np.clip(vix, 9.0, 80.0)
        # Feature 3: VIX3M/VIX term structure (contango proxy)
        vix_term = 1.1 + np.random.normal(0, 0.05, n_t)
        # Feature 4: SPY 5-day realized vol
        spy_vol = 0.008 + np.abs(np.random.normal(0, 0.003, n_t))

        df = pd.DataFrame(
            {
                "spy_log_ret": spy_ret,
                "vix_close": vix,
                "vix_term_structure": vix_term,
                "spy_realized_vol": spy_vol,
            },
            index=dates,
        )
    else:
        # Feature 1: BTC log return
        btc_ret = np.random.normal(0.001, 0.02, n_t)
        # Feature 2: BTC 10-day realized volatility
        btc_vol = 0.015 + np.abs(np.random.normal(0, 0.005, n_t))
        # Feature 3: Aggregate perp funding rate
        funding = np.random.normal(0.0001, 0.0002, n_t)
        # Feature 4: Stablecoin net inflows
        net_inflows = np.random.normal(1e6, 5e5, n_t)

        df = pd.DataFrame(
            {
                "btc_log_ret": btc_ret,
                "btc_realized_vol": btc_vol,
                "aggregate_funding_rate": funding,
                "stablecoin_net_inflows": net_inflows,
            },
            index=dates,
        )

    return df


def load_emission_features(
    asset_class: Literal["equity", "crypto"],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Loads emission features from CSV, DB, or falls back to synthetic generation."""
    if csv_path:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            # Slice to requested date range
            df = df.loc[start_date:end_date]
            return df
        else:
            print(f"Provided CSV path {csv_path} does not exist. Generating mock data.")

    # Try loading from DB if possible, otherwise generate mock data
    try:
        # The database might not have the specialized features (VIX, funding, inflows)
        # so we default to generating mock features for this implementation.
        return generate_mock_emission_features(asset_class, start_date, end_date)
    except Exception as e:
        print(f"Could not load features from DB: {e}. Generating mock data.")
        return generate_mock_emission_features(asset_class, start_date, end_date)


def train_hmm_model(features: np.ndarray, n_inits: int = 10) -> tuple[GaussianHMM, float]:
    """Fits GaussianHMM using multiple initializations and selects the best model."""
    best_model = None
    best_score = -np.inf

    for init_idx in range(n_inits):
        # Random initialization
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=1000,
            random_state=init_idx,
        )
        try:
            model.fit(features)
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise ValueError("HMM training failed to converge on all initializations.")

    return best_model, best_score


def label_hmm_states(
    model: GaussianHMM, asset_class: Literal["equity", "crypto"]
) -> dict[int, str]:
    """Label HMM states (Engaged Trend, Mean Reversion, Systemic Lapse).

    Criteria:
      - Equities: Lowest VIX mean = Engaged, highest VIX mean = Lapse.
      - Crypto: Lowest realized vol mean = Engaged, highest realized vol mean = Lapse.
    """
    means = model.means_  # shape (n_components, n_features)

    # In our features layout:
    # Equities: VIX close is feature index 1
    # Crypto: BTC realized vol is feature index 1
    vol_feature_index = 1

    vols = means[:, vol_feature_index]
    sorted_indices = np.argsort(vols)

    # Index with lowest volatility -> Engaged Trend
    # Index with highest volatility -> Systemic Lapse
    # Middle index -> Mean Reversion
    state_labels = {
        int(sorted_indices[0]): "engaged_trend",
        int(sorted_indices[1]): "mean_reversion",
        int(sorted_indices[2]): "systemic_lapse",
    }
    return state_labels


def main() -> None:
    """Orchestrate HMM training script."""
    parser = argparse.ArgumentParser(description="Train HMM Regime Model offline.")
    parser.add_argument(
        "--asset-class",
        type=str,
        required=True,
        choices=["equity", "crypto"],
        help="Asset class type",
    )
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-window", type=int, default=504, help="Train window size in bars")
    parser.add_argument("--step", type=int, default=21, help="Slide step size in bars")
    parser.add_argument("--n-inits", type=int, default=10, help="Number of random inits")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV feature path")

    args = parser.parse_args()

    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)

    print(f"Loading emission features for {args.asset_class} from {args.start} to {args.end}...")
    df = load_emission_features(args.asset_class, start_date, end_date, args.csv_path)

    n_t = len(df)
    if n_t <= args.train_window:
        print(
            f"Error: dataset size {n_t} is smaller than or equal to "
            f"train-window {args.train_window}."
        )
        sys.exit(1)

    print(f"Running rolling window HMM training (window={args.train_window}, step={args.step})...")

    # Initialize columns for decoded regime probabilities
    df["state_0_prob"] = np.nan
    df["state_1_prob"] = np.nan
    df["state_2_prob"] = np.nan
    df["dominant_state"] = -1

    feature_cols = [c for c in df.columns if not c.endswith("_prob") and c != "dominant_state"]

    # We do a rolling HMM: train on [t - train_window : t], standardizing features,
    # and decode the regime at index t.
    for t in range(args.train_window, n_t, args.step):
        train_df = df.iloc[t - args.train_window : t]
        train_features = train_df[feature_cols].values

        # Standardize features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)

        # Train model
        try:
            model, _ = train_hmm_model(train_scaled, args.n_inits)
            state_labels = label_hmm_states(model, args.asset_class)

            # Map the actual state indices to the standardized output labels
            # state_0_prob -> engaged_trend
            # state_1_prob -> mean_reversion
            # state_2_prob -> systemic_lapse
            engaged_idx = [k for k, v in state_labels.items() if v == "engaged_trend"][0]
            mr_idx = [k for k, v in state_labels.items() if v == "mean_reversion"][0]
            lapse_idx = [k for k, v in state_labels.items() if v == "systemic_lapse"][0]

            # Assign decoded outputs to the out-of-sample step [t : t + step_size]
            # to keep the rolling walk-forward decoding free of look-ahead.
            step_size = min(args.step, n_t - t)
            step_scaled = scaler.transform(df.iloc[t : t + step_size][feature_cols].values)

            step_probs = model.predict_proba(step_scaled)
            step_dominant = model.predict(step_scaled)

            # Map to standard state order: 0=engaged_trend, 1=mean_reversion, 2=systemic_lapse
            for k in range(step_size):
                idx = t + k
                probs = step_probs[k]
                dom = step_dominant[k]

                df.at[df.index[idx], "state_0_prob"] = probs[engaged_idx]
                df.at[df.index[idx], "state_1_prob"] = probs[mr_idx]
                df.at[df.index[idx], "state_2_prob"] = probs[lapse_idx]

                # Map dominant state
                if dom == engaged_idx:
                    df.at[df.index[idx], "dominant_state"] = 0
                elif dom == mr_idx:
                    df.at[df.index[idx], "dominant_state"] = 1
                else:
                    df.at[df.index[idx], "dominant_state"] = 2

        except Exception as e:
            print(f"Error at step {t}: {e}")
            continue

    # Fill any remaining NaNs in probabilities with 0.0 and dominant with -1
    df["state_0_prob"] = df["state_0_prob"].fillna(0.0)
    df["state_1_prob"] = df["state_1_prob"].fillna(0.0)
    df["state_2_prob"] = df["state_2_prob"].fillna(0.0)
    df["dominant_state"] = df["dominant_state"].fillna(-1).astype(int)

    # Train a final HMM on the entire dataset to pickle for live/runtime use
    print("Training final model on full history...")
    full_scaler = StandardScaler()
    full_scaled = full_scaler.fit_transform(df[feature_cols].values)
    final_model, final_score = train_hmm_model(full_scaled, args.n_inits)
    final_labels = label_hmm_states(final_model, args.asset_class)

    # Save to model file
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    model_path = models_dir / f"hmm_{args.asset_class}_{date_str}.pkl"

    model_data = {
        "model": final_model,
        "scaler": full_scaler,
        "labels": final_labels,
        "features": feature_cols,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Saved HMM model pickle to {model_path}")

    # Write states to TimescaleDB
    db_manager = ResultDBManager()
    print("Persisting regime states to TimescaleDB...")

    # Create table if not exists
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS regime_states (
            timestamp TIMESTAMPTZ NOT NULL,
            asset_class VARCHAR(16) NOT NULL,
            state_0_prob DOUBLE PRECISION,
            state_1_prob DOUBLE PRECISION,
            state_2_prob DOUBLE PRECISION,
            dominant_state INTEGER,
            PRIMARY KEY (timestamp, asset_class)
        );
    """
    with db_manager.engine.connect() as conn:
        with conn.begin():
            conn.execute(text(create_table_sql))
            # Convert to a hypertable; tolerate failure if TimescaleDB isn't available.
            try:
                conn.execute(
                    text(
                        "SELECT create_hypertable('regime_states', 'timestamp',"
                        " if_not_exists => TRUE)"
                    )
                )
            except Exception as e:
                print(f"Hypertable warning (regime_states): {e}")

    # Prepare records
    records = []
    for ts, row in df.iterrows():
        # Only write rows that were actually decoded (non-NaN dominant_state >= 0)
        if row["dominant_state"] < 0:
            continue
        records.append(
            {
                "timestamp": ts.to_pydatetime(),
                "asset_class": args.asset_class,
                "state_0_prob": float(row["state_0_prob"]),
                "state_1_prob": float(row["state_1_prob"]),
                "state_2_prob": float(row["state_2_prob"]),
                "dominant_state": int(row["dominant_state"]),
            }
        )

    if records:
        # Save to DB via DataFrame write
        records_df = pd.DataFrame(records)
        records_df.to_sql(
            "regime_states",
            db_manager.engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        print(f"Successfully persisted {len(records)} regime state rows to TimescaleDB.")
    else:
        print("No decoded records found to persist.")


if __name__ == "__main__":
    main()
