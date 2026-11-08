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


import types  # noqa: E402

import pytest  # noqa: E402

import ggTrader.lab.blend as blend_mod  # noqa: E402
import ggTrader.lab.wfo as wfo_pkg  # noqa: E402
import ggTrader.lab.wfo.cli_support as cli_support  # noqa: E402
from ggTrader.lab.blend import run_blend  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.wfo import WfoResult  # noqa: E402


def _patch_persist(monkeypatch, calls=None):
    calls = calls if calls is not None else {}

    def bump(name, ret=None):
        return lambda *a, **k: calls.__setitem__(name, calls.get(name, 0) + 1) or ret

    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(blend_mod.persist, "start_run", bump("start", "run123"))
    monkeypatch.setattr(blend_mod.persist, "write_returns_equity", bump("returns"))
    monkeypatch.setattr(blend_mod.persist, "write_summary", bump("summary"))
    monkeypatch.setattr(blend_mod.persist, "finish_run", bump("finish"))
    return calls


def _patch_inputs(monkeypatch, idx, spy, symbols=("AAA",), seen=None):
    """Replace the tape load and the per-sleeve inputs; ``seen`` records the universes asked."""
    bars = types.SimpleNamespace(
        close=pd.DataFrame({"SPY": spy, "AAA": spy.values}, index=idx), provenance={}
    )
    monkeypatch.setattr(cli_support, "load_run_bars", lambda *a, **k: bars)

    def _sleeve_inputs(b, universe, cfg, es, ee, **kwargs):
        if seen is not None:
            seen.append(universe)
        return types.SimpleNamespace(symbols=list(symbols), universe=universe), f"mask-{universe}"

    monkeypatch.setattr(cli_support, "sleeve_inputs", _sleeve_inputs)


def _run(sleeves):
    return run_blend(
        sleeves,
        LabConfig(),
        "2021-01-01",
        "2022-07-01",
        market="equity",
        base_config=dict(blend_mod.STOCK_BASE_CONFIG),
    )


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
    _patch_inputs(monkeypatch, idx, spy)
    labels = iter(eqs.values())
    monkeypatch.setattr(
        wfo_pkg,
        "run_wfo",
        lambda *a, **k: WfoResult(
            oos_equity=next(labels), fold_results=[], live_params={}, table="t"
        ),
    )
    calls = _patch_persist(monkeypatch)

    result = _run([("ensemble", "sp500"), ("xs_momentum", "nasdaq100")])
    assert result.run_id == "run123"
    assert set(result.sleeve_equity) == {"ensemble@sp500", "xs_momentum@nasdaq100"}
    assert result.blended_equity.notna().all()
    assert calls["start"] == 1 and calls["finish"] == 1
    assert calls["returns"] == 3  # 2 sleeves + 1 blend
    assert calls["summary"] == 1
    assert "blend" in result.table.lower()


def test_run_blend_raises_on_wfo_no_folds(monkeypatch):
    """run_wfo returning a bare str (the no-folds path) must raise SystemExit."""
    idx = _idx(400)
    spy = _equity_from_returns(
        pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, 400), index=idx)
    )
    _patch_inputs(monkeypatch, idx, spy)
    _patch_persist(monkeypatch)
    monkeypatch.setattr(wfo_pkg, "run_wfo", lambda *a, **k: "no valid folds")
    with pytest.raises(SystemExit, match="no valid folds"):
        _run([("ensemble", "sp500")])


def test_run_blend_gives_each_sleeve_its_own_universe_and_mask(monkeypatch):
    """Each sleeve's inputs are scoped to its own universe (no late-binding closure bug)."""
    idx = _idx(400)
    rng = np.random.default_rng(5)
    eqs = iter(
        _equity_from_returns(pd.Series(rng.normal(0.0004, 0.011, 400), index=idx)) for _ in range(2)
    )
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))
    seen: list = []
    _patch_inputs(monkeypatch, idx, spy, seen=seen)
    _patch_persist(monkeypatch)
    masks: list = []

    def _fake_wfo(name, cls, cfg, sleeve, spy_close, **k):
        masks.append((name, sleeve.universe, k["mask"]))
        return WfoResult(oos_equity=next(eqs), fold_results=[], live_params={}, table="t")

    monkeypatch.setattr(wfo_pkg, "run_wfo", _fake_wfo)
    _run([("xs_momentum", "sp500"), ("ensemble", "nasdaq100")])
    assert seen == ["sp500", "nasdaq100"]
    assert masks == [
        ("xs_momentum", "sp500", "mask-sp500"),
        ("ensemble", "nasdaq100", "mask-nasdaq100"),
    ]


def test_run_blend_raises_on_empty_symbols(monkeypatch):
    """A universe with no symbols on the tape must raise SystemExit."""
    idx = _idx(400)
    spy = _equity_from_returns(
        pd.Series(np.random.default_rng(4).normal(0.0003, 0.01, 400), index=idx)
    )
    _patch_inputs(monkeypatch, idx, spy, symbols=())
    _patch_persist(monkeypatch)
    with pytest.raises(SystemExit, match="no symbols available"):
        _run([("ensemble", "sp500")])


def test_run_blend_refuses_a_strategy_that_is_not_ported(monkeypatch):
    idx = _idx(400)
    spy = _equity_from_returns(
        pd.Series(np.random.default_rng(6).normal(0.0003, 0.01, 400), index=idx)
    )
    _patch_inputs(monkeypatch, idx, spy)
    _patch_persist(monkeypatch)
    with pytest.raises(SystemExit, match="legacy-engine-vbt0285"):
        _run([("pead", "sp500")])


def test_run_blend_has_no_engine_argument() -> None:
    import inspect

    assert "engine" not in inspect.signature(run_blend).parameters


def test_run_wfo_json_provenance_names_the_strategy_source(tmp_path, monkeypatch):
    """The result JSON carries a hash of the strategy file(s) next to the stack fields."""
    import json

    from ggTrader.lab.strategies import registry

    monkeypatch.setattr(cli_support, "result_payload", lambda *a, **k: {"strategy": k["strategy"]})
    bars = type("B", (), {"provenance": {}, "summary": lambda self: {}})()
    out = cli_support.write_wfo_json(None, bars, str(tmp_path / "r.json"), strategy="ensemble")
    block = json.loads(out.read_text())["provenance"]
    assert "membership_sha256" not in block  # no source given: nothing is claimed
    expected = registry.source_sha256(registry.REGISTRY["ensemble"])
    assert block["strategy_sha256"] == expected["strategy_sha256"]
    assert "ensemble.py" in block["strategy_files"]


def test_run_blend_stamps_the_membership_vintage_in_the_provenance(monkeypatch):
    """The lab_runs provenance names the S&P 500 list the sleeves were built from."""
    idx = _idx(400)
    rng = np.random.default_rng(5)
    eq = _equity_from_returns(pd.Series(rng.normal(0.0004, 0.011, 400), index=idx))
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))
    _patch_inputs(monkeypatch, idx, spy)
    monkeypatch.setattr(
        wfo_pkg,
        "run_wfo",
        lambda *a, **k: WfoResult(oos_equity=eq, fold_results=[], live_params={}, table="t"),
    )
    _patch_persist(monkeypatch)
    stamps = {}
    monkeypatch.setattr(
        blend_mod.persist, "start_run", lambda *a, **k: stamps.update(k) or "run123"
    )
    _run([("ensemble", "sp500")])
    block = stamps["provenance"]
    assert block["membership"]["kind"] == "fixture"
    assert block["membership"]["fixture"] == "sp500_v1_2026-06-02"
    assert len(block["membership_sha256"]) == 64
    assert block["membership"]["age_since_verified_days"] is not None


def test_wfo_json_records_the_membership_vintage_in_provenance_only(monkeypatch, tmp_path):
    import json

    from ggTrader.data.membership import FixtureMembershipSource

    monkeypatch.setattr(cli_support, "result_payload", lambda *a, **k: {"strategy": k["strategy"]})
    bars = type("B", (), {"provenance": {}, "summary": lambda self: {}})()
    source = FixtureMembershipSource("sp500_v1_2026-06-02")
    out = cli_support.write_wfo_json(
        None,
        bars,
        str(tmp_path / "r.json"),
        strategy="ensemble",
        eval_end="2026-04-30",
        membership=source,
    )
    saved = json.loads(out.read_text())
    assert saved["strategy"] == "ensemble" and set(saved) == {"strategy", "provenance"}
    assert saved["provenance"]["membership_sha256"] == source.vintage().content_sha256
    assert saved["provenance"]["membership"]["fixture"] == "sp500_v1_2026-06-02"
    assert saved["provenance"]["membership"]["age_since_verified_days"] == -40
