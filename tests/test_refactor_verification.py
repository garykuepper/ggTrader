from unittest.mock import patch

import numpy as np
import pandas as pd

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_with_movers


@patch("ggTrader.utils.setup.load_data_and_setup")
@patch("ggTrader.utils.setup.build_mover_mask")
def test_load_data_with_movers(mock_build, mock_load):
    """Verify load_data_with_movers logic."""
    mock_df = pd.DataFrame({"close": [1, 2, 3]})
    mock_load.return_value = mock_df
    mock_mask = pd.DataFrame({"mask": [True, False, True]})
    mock_build.return_value = mock_mask

    # Case 1: USE_MOVERS = 0
    config = {"USE_MOVERS": 0}
    ohlcv, mask = load_data_with_movers(config)
    assert ohlcv.equals(mock_df)
    assert mask is None
    mock_build.assert_not_called()

    # Case 2: USE_MOVERS > 0
    config = {"USE_MOVERS": 10}
    ohlcv, mask = load_data_with_movers(config)
    assert ohlcv.equals(mock_df)
    assert mask.equals(mock_mask)
    mock_build.assert_called_once()


def test_fast_backtest_helpers():
    """Verify FastBacktest helper methods exist and work (smoke test)."""
    # Create a dummy FastBacktest instance
    ohlcv = pd.DataFrame(
        np.random.randn(10, 4),
        columns=pd.MultiIndex.from_product([["BTC"], ["open", "high", "low", "close"]]),
    )
    params = {"a": 1}
    fb = FastBacktest(ohlcv, params)

    # Test _determine_grouping
    entries = pd.DataFrame(index=ohlcv.index, columns=ohlcv.columns.get_level_values(0))
    # Case 1: Single run (Index columns)
    group_by, use_sharing = fb._determine_grouping(entries)
    assert use_sharing is True  # Default
    assert isinstance(group_by, np.ndarray)

    # Case 2: Grid search (MultiIndex columns)
    entries_multi = pd.DataFrame(
        index=ohlcv.index,
        columns=pd.MultiIndex.from_tuples([(1, "BTC"), (2, "BTC")], names=["p", "s"]),
    )
    group_by, use_sharing = fb._determine_grouping(entries_multi)
    assert use_sharing is True
    # It should drop the last level "s" and keep "p"
    assert len(group_by) == 2
