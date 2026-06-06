"""Runtime Hidden Markov Model (HMM) regime filtering."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sqlalchemy import text

from ggTrader.utils.result_db_manager import ResultDBManager

logger = logging.getLogger(__name__)


def load_regime_gate(
    asset_class: Literal["equity", "crypto"],
    start: pd.Timestamp,
    end: pd.Timestamp,
    engaged_threshold: float = 0.65,
) -> pd.Series:
    """Loads pre-computed regime probabilities from TimescaleDB and produces
    a boolean execution gate Series for use at runtime.

    Args:
        asset_class: 'equity' or 'crypto'.
        start: Start timestamp of the query.
        end: End timestamp of the query.
        engaged_threshold: Minimum probability of state 0 (Engaged Trend) to keep gate open.

    Returns:
        pd.Series: Boolean series indexed by timestamp.
    """
    db = ResultDBManager()
    query = text(
        """
        SELECT timestamp, state_0_prob, dominant_state
        FROM regime_states
        WHERE asset_class = :asset_class
          AND timestamp >= :start
          AND timestamp <= :end
        ORDER BY timestamp
        """
    )
    try:
        with db.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "asset_class": asset_class,
                    "start": start,
                    "end": end,
                },
            )
    except Exception as e:
        logger.warning(f"Could not load regime states from DB: {e}. Defaulting to False gate.")
        return pd.Series(dtype=bool)

    if df.empty:
        logger.warning("No regime states found in database. Defaulting to False gate.")
        return pd.Series(dtype=bool)

    # Convert timestamps to index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Timezone alignment
    if start.tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(start.tz)
        else:
            df.index = df.index.tz_convert(start.tz)
    else:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    # Filter logic: gate True where state_0_prob >= engaged_threshold,
    # and gate False where dominant_state == 2
    # Missing timestamps will default to False when reindexed by the caller
    gate = (df["state_0_prob"] >= engaged_threshold) & (df["dominant_state"] != 2)
    return gate.astype(bool)
