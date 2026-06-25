from ggTrader.data.core.index_constituents import snapshot_members
from ggTrader.lab.cli import UNIVERSE_CHOICES


def test_midcap400_registered_in_universe_choices():
    assert "midcap400" in UNIVERSE_CHOICES


def test_midcap400_snapshot_loads_normalized():
    members = snapshot_members("midcap400")
    assert 380 <= len(members) <= 420
    # normalized: no dotted class tickers (MOG.A -> MOG-A)
    assert all("." not in m for m in members)
    assert members == sorted(members)
