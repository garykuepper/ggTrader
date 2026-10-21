"""Check the registered factor formulas, windows, Holm family and data stops."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab import factor_screen
from ggTrader.lab.factor_screen import FULL, screen, value_momentum_combo
from ggTrader.lab.french_data import (
    FACTOR_SCHEMAS,
    VALUE_FILE,
    download_and_freeze,
    load_size_factor_vw,
)
from ggTrader.lab.momentum_screen import cagr_diff_pp, holm, net_excess, nw_t


def _panels() -> list[pd.DataFrame]:
    """Build only the histories required by the brief, without external services."""
    index = pd.date_range("1927-01-31", "2026-04-30", freq="ME")
    market = 0.005 + 0.002 * np.sin(np.arange(len(index)))
    value = pd.DataFrame({"small": market + 0.001, "big": market + 0.002}, index=index)
    common = value.loc[slice(*FULL)].copy()
    momentum = pd.DataFrame({"big_hi": market + 0.004}, index=index)
    benchmark = pd.DataFrame({"mkt": market, "rf": 0.001}, index=index)
    return [value, common.copy(), common.copy(), momentum, benchmark]


def test_h4_gross_combo_then_one_haircut_and_388_month_a2() -> None:
    """Use the monthly 50/50 gross return and one H4 cost, over the exact A2 window."""
    panels = _panels()
    value, _, _, momentum, market = panels
    combo = value_momentum_combo(value.big, momentum.big_hi)
    pd.testing.assert_series_equal(combo, (value.big + momentum.big_hi) / 2)
    result = screen(*panels)
    assert result["inputs"]["common_window"] == {"start": FULL[0], "end": FULL[1], "n_months": 754}
    assert result["inputs"]["A2_window"] == {"start": "1994-01", "end": "2026-04", "n_months": 388}
    h4 = result["hypotheses"]["H4"]
    assert h4["A1"]["t"] == pytest.approx(
        nw_t(net_excess(combo.loc[slice(*FULL)], market.mkt.loc[slice(*FULL)], 0.015))
    )
    net_wealth = (1 + combo.loc["1994-01":"2026-04"] - 0.015 / 12).prod()
    market_wealth = (1 + market.mkt.loc["1994-01":"2026-04"]).prod()
    expected = 100 * (net_wealth ** (12 / 388) - market_wealth ** (12 / 388))
    assert h4["A2"]["cagr_diff_pp"] == pytest.approx(expected)
    assert h4["A2"]["pass"] == (expected >= 1.0)
    assert h4["overall_pass"] == (h4["A1"]["pass_holm"] and h4["A2"]["pass"])
    assert h4["diagnostics"]["history_1927"]["n_months"] == len(value)
    assert h4["diagnostics"]["stress"]["post_cagr_diff_pp"] == pytest.approx(
        cagr_diff_pp(combo.loc["1994-01":], market.mkt.loc["1994-01":], 0.03)
    )
    assert set(h4["diagnostics"]["capm"]) == {"alpha_ann_pct", "beta"}
    assert set(result["small_cap_diagnostics"]) == {"SMALL HiOP", "SMALL HiBM", "SMALL LoINV"}
    for diagnostic in result["small_cap_diagnostics"].values():
        assert diagnostic["base_cost"] == 0.03
        assert diagnostic["stress"]["annual_cost"] == 0.06


def test_holm_four_sequential_stop_and_exact_thresholds() -> None:
    """Later p-values below their own hurdles still fail after the first failure."""
    assert holm({"H4": 0.001, "H2": 0.00046, "H1": 0.0001, "H3": 0.0005}) == {
        "H4": False,
        "H2": False,
        "H1": True,
        "H3": False,
    }
    assert all(holm({"H1": 0.0003375, "H2": 0.00045, "H3": 0.000675, "H4": 0.00135}).values())


@pytest.mark.parametrize(
    "panel,column,month,defect",
    [
        (0, "big", "1963-07-31", "missing"),
        (1, "big", "2026-04-30", "missing"),
        (2, "big", "2000-01-31", "duplicate"),
        (3, "big_hi", "1927-01-31", "missing"),
        (4, "rf", "2000-01-31", "nan"),
        (0, "small", "2000-01-31", "inf"),
        (1, "big", "2000-01-31", "zero_wealth"),
        (2, "small", "2000-01-31", "zero_small_wealth"),
        (4, "mkt", "2000-01-31", "negative_wealth"),
    ],
)
def test_data_stops_before_statistics(
    panel: int, column: str, month: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing, repeated, invalid or non-positive stress inputs cannot produce a verdict."""
    panels = _panels()
    date = pd.Timestamp(month)
    if defect == "missing":
        panels[panel] = panels[panel].drop(date)
    elif defect == "duplicate":
        panels[panel] = pd.concat([panels[panel], panels[panel].loc[[date]]])
    else:
        panels[panel].loc[date, column] = {
            "nan": np.nan,
            "inf": np.inf,
            "zero_wealth": -1 + 0.02 / 12,
            "zero_small_wealth": -1 + 0.06 / 12,
            "negative_wealth": -1.01,
        }[defect]

    def unexpected_statistic(*args: object, **kwargs: object) -> None:
        raise AssertionError("Data-quality stops must precede all statistics")

    monkeypatch.setattr(factor_screen, "nw_t", unexpected_statistic)
    with pytest.raises(ValueError):
        screen(*panels)


def _archive(filename: str, defect: str = "") -> bytes:
    """Provide strict six-portfolio VW schemas without using the real frozen data."""
    characteristic, _ = FACTOR_SCHEMAS[filename]
    header = (
        f",SMALL Lo{characteristic},ME1 {characteristic}2,SMALL Hi{characteristic},"
        f"BIG Lo{characteristic},ME2 {characteristic}2,BIG Hi{characteristic}"
    )
    text = f"Average Value Weighted Returns -- Monthly\n{header}\n196307,1,2,3,4,5,6\n"
    if defect == "header":
        text = text.replace("BIG Hi", "BIG Unexpected")
    elif defect == "section":
        text = text.replace("Value Weighted", "Equal Weighted")
    elif defect == "layout":
        text = text.replace("196307", "1963-07")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fixture.csv", text)
    return buffer.getvalue()


@pytest.mark.parametrize("filename", list(FACTOR_SCHEMAS))
def test_factor_loader_schema_and_checksum_stop(filename: str, tmp_path: Path) -> None:
    """Load the intended factor legs and reject any subsequent archive tampering."""
    download_and_freeze([filename], dest=tmp_path, http_get=lambda url: _archive(filename))
    frame = load_size_factor_vw(filename, tmp_path)
    assert list(frame.columns) == ["small", "big"]
    expected = [0.01, 0.04] if "INV" in filename else [0.03, 0.06]
    assert frame.iloc[0].tolist() == pytest.approx(expected)
    (tmp_path / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_size_factor_vw(filename, tmp_path)


@pytest.mark.parametrize("defect", ["header", "section", "layout"])
def test_factor_loader_strict_schema_stop(defect: str, tmp_path: Path) -> None:
    """Changed source headers, table sections and date layouts must stop parsing."""
    download_and_freeze(
        [VALUE_FILE], dest=tmp_path, http_get=lambda url: _archive(VALUE_FILE, defect)
    )
    with pytest.raises(ValueError):
        load_size_factor_vw(VALUE_FILE, tmp_path)
