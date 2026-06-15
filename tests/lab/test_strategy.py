from ggTrader.lab.strategy import LabConfig


def test_labconfig_defaults():
    cfg = LabConfig()
    assert cfg.top_n == 50
    assert cfg.lookback == 252
    assert cfg.skip == 21
    assert cfg.min_history_bars == 400
    assert cfg.max_stocks is None


def test_labconfig_override():
    cfg = LabConfig(top_n=10, max_stocks=20)
    assert cfg.top_n == 10
    assert cfg.max_stocks == 20
