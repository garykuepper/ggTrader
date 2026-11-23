"""Tests for the combined release bars interface."""

import pytest

from ggTrader.data.bars import load_bars


def test_release_id_with_db_source_is_rejected():
    with pytest.raises(ValueError, match="release_id"):
        load_bars(["AAPL"], source="db", release_id=1)


def test_combined_rejects_snapshot_path():
    with pytest.raises(ValueError, match="snapshot_path"):
        load_bars(["AAPL"], source="combined", snapshot_path="unused")
