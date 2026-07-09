"""Tests for the portfolio-blend helper and orchestrator."""

import numpy as np
import pandas as pd

from ggTrader.lab.blend import blend_curves


def _idx(n, start="2021-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _equity_from_returns(rets: pd.Series, start=100000.0) -> pd.Series:
    return (1.0 + rets).cumprod() * start


def test_blend_curves_aligns_on_intersection_and_blends():
    idx = _idx(400)
    np.random.seed(0)
    a = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 400), index=idx))
    # b starts 50 bars later -> intersection trims the blend to the common span
    b = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 350), index=idx[50:]))
    blend_eq, returns_df, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b})
    assert list(returns_df.columns) == ["A@sp500", "B@nasdaq100"]
    assert returns_df.index.min() >= idx[50]  # trimmed to the later start
    assert blend_eq.notna().all()
    assert (blend_eq > 0).all()


def test_blend_curves_equal_vol_gives_balanced_weights():
    """Two sleeves with the same vol get ~50/50 inverse-vol weight (diag)."""
    idx = _idx(400)
    rng = np.random.default_rng(1)
    a = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    b = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    _, _, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b}, window=60)
    last = diag.iloc[-1]
    assert abs(last["w_A@sp500"] - last["w_B@nasdaq100"]) < 0.15  # near-balanced


import ggTrader.lab.blend as blend_mod  # noqa: E402
from ggTrader.lab.blend import run_blend  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.wfo import WfoResult  # noqa: E402


def test_run_blend_orchestrates_and_persists(monkeypatch):
    idx = _idx(400)
    rng = np.random.default_rng(2)
    eqs = {
        "ensemble@sp500": _equity_from_returns(
            pd.Series(rng.normal(0.0004, 0.011, 400), index=idx)
        ),
        "xs_momentum@nasdaq100": _equity_from_returns(
            pd.Series(rng.normal(0.0003, 0.013, 400), index=idx)
        ),
    }
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))
    spy_ohlcv = pd.concat({"SPY": pd.DataFrame({"close": spy})}, axis=1)
    spy_ohlcv.columns.names = ["symbol", "field"]

    # universe membership + ohlcv: return a frame containing SPY + 1 dummy symbol
    monkeypatch.setattr(blend_mod, "equity_universe_between", lambda *a, **k: ["AAA"])

    def _fake_load(symbols, start, end, **k):
        frames = {"SPY": pd.DataFrame({"close": spy}, index=idx)}
        frames["AAA"] = pd.DataFrame({"close": spy.values}, index=idx)
        df = pd.concat(frames, axis=1)
        df.columns.names = ["symbol", "field"]
        return df

    monkeypatch.setattr(blend_mod, "load_ohlcv", _fake_load)
    monkeypatch.setattr(blend_mod, "build_grid", lambda cls: [{}])

    labels = iter(eqs.values())

    def _fake_wfo(name, cls, cfg, ohlcv, spy_close, **k):
        return WfoResult(oos_equity=next(labels), fold_results=[], live_params={}, table="t")

    monkeypatch.setattr(blend_mod, "run_wfo", _fake_wfo)

    calls = {"start": 0, "returns": 0, "summary": 0, "finish": 0}
    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(
        blend_mod.persist,
        "start_run",
        lambda *a, **k: calls.__setitem__("start", calls["start"] + 1) or "run123",
    )
    monkeypatch.setattr(
        blend_mod.persist,
        "write_returns_equity",
        lambda *a, **k: calls.__setitem__("returns", calls["returns"] + 1),
    )
    monkeypatch.setattr(
        blend_mod.persist,
        "write_summary",
        lambda *a, **k: calls.__setitem__("summary", calls["summary"] + 1),
    )
    monkeypatch.setattr(
        blend_mod.persist,
        "finish_run",
        lambda *a, **k: calls.__setitem__("finish", calls["finish"] + 1),
    )

    result = run_blend(
        [("ensemble", "sp500"), ("xs_momentum", "nasdaq100")],
        LabConfig(),
        "2021-01-01",
        "2022-07-01",
        market="equity",
        base_config=dict(blend_mod.STOCK_BASE_CONFIG),
    )
    assert result.run_id == "run123"
    assert set(result.sleeve_equity) == {"ensemble@sp500", "xs_momentum@nasdaq100"}
    assert result.blended_equity.notna().all()
    assert calls["start"] == 1 and calls["finish"] == 1
    assert calls["returns"] == 3  # 2 sleeves + 1 blend
    assert calls["summary"] == 1
    assert "blend" in result.table.lower()


def _make_monkeypatched_base(monkeypatch, rng, idx, spy):
    """Shared setup for run_blend guard tests."""
    monkeypatch.setattr(blend_mod, "equity_universe_between", lambda *a, **k: ["AAA"])

    def _fake_load(symbols, start, end, **k):
        frames = {"SPY": pd.DataFrame({"close": spy}, index=idx)}
        frames["AAA"] = pd.DataFrame({"close": spy.values}, index=idx)
        df = pd.concat(frames, axis=1)
        df.columns.names = ["symbol", "field"]
        return df

    monkeypatch.setattr(blend_mod, "load_ohlcv", _fake_load)
    monkeypatch.setattr(blend_mod, "build_grid", lambda cls: [{}])

    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(blend_mod.persist, "start_run", lambda *a, **k: "run-x")
    monkeypatch.setattr(blend_mod.persist, "write_returns_equity", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "write_summary", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "finish_run", lambda *a, **k: None)


def test_run_blend_raises_on_wfo_no_folds(monkeypatch):
    """run_wfo returning a bare str (no-folds path) must raise SystemExit."""
    idx = _idx(400)
    rng = np.random.default_rng(3)
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))

    _make_monkeypatched_base(monkeypatch, rng, idx, spy)

    # Simulate the no-folds return value: run_wfo returns a plain str
    monkeypatch.setattr(blend_mod, "run_wfo", lambda *a, **k: "no valid folds")

    import pytest

    with pytest.raises(SystemExit):
        run_blend(
            [("ensemble", "sp500")],
            LabConfig(),
            "2021-01-01",
            "2022-07-01",
            market="equity",
            base_config=dict(blend_mod.STOCK_BASE_CONFIG),
        )


def test_run_blend_passes_per_sleeve_universe_fn(monkeypatch):
    """Weight-based strategies need universe_fn; each sleeve's closure must be
    scoped to its own universe, not the last-iterated loop value (late-binding
    closure bug)."""
    idx = _idx(400)
    rng = np.random.default_rng(5)
    eqs = {
        "xs_momentum@sp500": _equity_from_returns(
            pd.Series(rng.normal(0.0004, 0.011, 400), index=idx)
        ),
        "idio_vol@nasdaq100": _equity_from_returns(
            pd.Series(rng.normal(0.0003, 0.013, 400), index=idx)
        ),
    }
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))

    monkeypatch.setattr(blend_mod, "equity_universe_between", lambda *a, **k: ["AAA"])

    def _fake_load(symbols, start, end, **k):
        frames = {"SPY": pd.DataFrame({"close": spy}, index=idx)}
        frames["AAA"] = pd.DataFrame({"close": spy.values}, index=idx)
        df = pd.concat(frames, axis=1)
        df.columns.names = ["symbol", "field"]
        return df

    monkeypatch.setattr(blend_mod, "load_ohlcv", _fake_load)
    monkeypatch.setattr(blend_mod, "build_grid", lambda cls: [{}])
    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(blend_mod.persist, "start_run", lambda *a, **k: "run-x")
    monkeypatch.setattr(blend_mod.persist, "write_returns_equity", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "write_summary", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "finish_run", lambda *a, **k: None)

    labels = iter(eqs.values())
    captured_universe_fns: list = []

    def _fake_wfo(name, cls, cfg, ohlcv, spy_close, **k):
        assert "universe_fn" in k
        captured_universe_fns.append(k["universe_fn"])
        return WfoResult(oos_equity=next(labels), fold_results=[], live_params={}, table="t")

    monkeypatch.setattr(blend_mod, "run_wfo", _fake_wfo)

    def _fake_eligible_at(asof, past, cfg, *, universe):
        return ([f"ELIGIBLE_{universe.upper()}"], {})

    monkeypatch.setattr(blend_mod, "eligible_at", _fake_eligible_at)

    run_blend(
        [("xs_momentum", "sp500"), ("idio_vol", "nasdaq100")],
        LabConfig(),
        "2021-01-01",
        "2022-07-01",
        market="equity",
        base_config=dict(blend_mod.STOCK_BASE_CONFIG),
    )

    assert len(captured_universe_fns) == 2
    # Each sleeve's universe_fn must resolve to ITS OWN universe, not both
    # resolving to the last-iterated "nasdaq100" (the late-binding bug).
    assert captured_universe_fns[0](None, None) == ["ELIGIBLE_SP500"]
    assert captured_universe_fns[1](None, None) == ["ELIGIBLE_NASDAQ100"]


def test_run_blend_raises_on_empty_symbols(monkeypatch):
    """No overlap between universe members and available OHLCV must raise SystemExit."""
    idx = _idx(400)
    rng = np.random.default_rng(4)
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))

    # Override equity_universe_between to return a symbol NOT in the loaded ohlcv
    monkeypatch.setattr(blend_mod, "equity_universe_between", lambda *a, **k: ["MISSING_SYM"])

    def _fake_load(symbols, start, end, **k):
        # Only loads SPY — MISSING_SYM is absent so syms will be empty
        frames = {"SPY": pd.DataFrame({"close": spy}, index=idx)}
        df = pd.concat(frames, axis=1)
        df.columns.names = ["symbol", "field"]
        return df

    monkeypatch.setattr(blend_mod, "load_ohlcv", _fake_load)
    monkeypatch.setattr(blend_mod, "build_grid", lambda cls: [{}])
    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(blend_mod.persist, "start_run", lambda *a, **k: "run-x")
    monkeypatch.setattr(blend_mod.persist, "write_returns_equity", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "write_summary", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "finish_run", lambda *a, **k: None)

    import pytest

    with pytest.raises(SystemExit):
        run_blend(
            [("ensemble", "sp500")],
            LabConfig(),
            "2021-01-01",
            "2022-07-01",
            market="equity",
            base_config=dict(blend_mod.STOCK_BASE_CONFIG),
        )
