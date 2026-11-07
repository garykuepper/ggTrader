"""``ggt lab --wfo`` and ``--blend`` end to end on the frozen tape (needs the real snapshot)."""

from __future__ import annotations

import json

import pytest

import ggTrader.lab.blend as blend_mod
from ggTrader.lab.cli import run_lab
from ggTrader.lab.strategy import LabConfig
from tests.lab.wfo import helpers as h

pytestmark = [pytest.mark.filterwarnings("ignore"), pytest.mark.slow]


@h.needs_real
def test_vbt_wfo_writes_a_provenance_stamped_result(tmp_path, capsys) -> None:
    out = tmp_path / "wfo.json"
    table = run_lab(
        [
            "--strategy",
            "ensemble",
            "--wfo",
            "--eval-start",
            "2021-01-31",
            "--eval-end",
            "2022-07-31",
            "--max-stocks",
            "40",
            "--result-json",
            str(out),
        ]
    )
    assert "WFO: ensemble" in table and "2 folds" in table
    data = json.loads(out.read_text())
    prov = data["provenance"]
    assert prov["engine_effective"] == "rust" and prov["baseline_epoch"] == "E1"
    assert prov["tape_sha256"] and prov["definitions"]["year_freq"] == "252 days"
    assert data["n_folds"] == 2 and data["diagnostics"]["n_simulations"] == 1
    assert data["diagnostics"]["definitions"] == "vbt"
    assert "result JSON" in capsys.readouterr().out


@h.needs_real
def test_vbt_blend_runs_the_sleeves_on_the_engine_and_stamps_the_run(monkeypatch) -> None:
    calls: dict = {}
    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(
        blend_mod.persist, "start_run", lambda *a, **k: calls.update(start=(a, k)) or "run-vbt"
    )
    monkeypatch.setattr(blend_mod.persist, "write_returns_equity", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "write_summary", lambda *a, **k: None)
    monkeypatch.setattr(blend_mod.persist, "finish_run", lambda *a, **k: None)
    result = blend_mod.run_blend(
        [("ensemble", "sp500"), ("cross_asset_sleeve", "cross_asset_sleeve")],
        LabConfig(max_stocks=40),
        "2021-01-31",
        "2022-07-31",
        market="equity",
        base_config=dict(h.BASE),
    )
    assert result.run_id == "run-vbt"
    assert set(result.sleeve_equity) == {"ensemble@sp500", "cross_asset_sleeve@cross_asset_sleeve"}
    _, kwargs = calls["start"]
    assert kwargs["provenance"]["engine_effective"] == "rust"
    assert kwargs["params"]["engine"] == "vbt" and kwargs["params"]["max_leverage"] == 1.0
