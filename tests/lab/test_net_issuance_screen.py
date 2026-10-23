"""Check net-issuance source selection, fixed windows, gates, and data stops."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab import net_issuance_screen
from ggTrader.lab.french_data import download_and_freeze
from ggTrader.lab.momentum_screen import cagr_diff_pp, net_excess, nw_t, one_sided_p
from ggTrader.lab.net_issuance_screen import (
    BIG_COLUMNS,
    FULL,
    POST,
    SIZE_NI_FILE,
    UNI_NI_FILE,
    load_net_issuance_vw,
    screen,
)


def _archive(text: str) -> bytes:
    """Build a tiny frozen French CSV for exact-header tests."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fixture.csv", text)
    return buffer.getvalue()


@pytest.mark.parametrize("defect", ["", "trailing_space", "size_header", "section"])
def test_load_exact_value_weighted_columns(defect: str, tmp_path: Path) -> None:
    """Select BIG NegNI and the raw '< 0 ' field; reject changed registered labels."""
    size = (
        "Average Value Weighted Returns -- Monthly\n,"
        + ",".join(BIG_COLUMNS)
        + "\n196307,1,2,3,4,5,6,7\n"
    )
    uni = "Value Weighted Returns -- Monthly\n,< 0 ,ZERO\n196307,8,9\n"
    if defect == "trailing_space":
        uni = uni.replace("< 0 ", "< 0")
    elif defect == "size_header":
        size = size.replace("BIG NegNI", "BIG LoNI")
    elif defect == "section":
        size = size.replace("Average Value Weighted", "Average Equal Weighted")
    for filename, text in ((SIZE_NI_FILE, size), (UNI_NI_FILE, uni)):
        download_and_freeze([filename], dest=tmp_path, http_get=lambda url, raw=text: _archive(raw))
    if defect:
        with pytest.raises(ValueError):
            load_net_issuance_vw(tmp_path)
    else:
        loaded_size, loaded_uni = load_net_issuance_vw(tmp_path)
        assert list(loaded_size) == list(BIG_COLUMNS)
        assert list(loaded_uni) == ["< 0"]
        assert loaded_size.iloc[0, 0] == pytest.approx(0.01)
        assert loaded_uni.iloc[0, 0] == pytest.approx(0.08)
        (tmp_path / UNI_NI_FILE).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_net_issuance_vw(tmp_path)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build exactly the required 754 months with distinguishable cells."""
    index = pd.date_range("1963-07-31", "2026-04-30", freq="ME")
    market = pd.Series(0.005 + 0.002 * np.sin(np.arange(len(index))), index=index)
    size = pd.DataFrame(
        {name: market + 0.002 - i * 0.0002 for i, name in enumerate(BIG_COLUMNS)},
        index=index,
    )
    uni = pd.DataFrame({"< 0": market + 0.001}, index=index)
    benchmark = pd.DataFrame({"mkt": market, "rf": 0.001}, index=index)
    return size, uni, benchmark


def test_fixed_h1_gates_and_all_diagnostics() -> None:
    """Apply one H1 test and the decimal CAGR formula over exactly 388 months."""
    size, uni, market = _panels()
    result = screen(size, uni, market)
    assert result["inputs"]["full_window"] == {"start": FULL[0], "end": FULL[1], "n_months": 754}
    assert result["inputs"]["A2_window"] == {"start": POST[0], "end": POST[1], "n_months": 388}
    h1 = result["H1"]
    t = nw_t(net_excess(size["BIG NegNI"], market.mkt, 0.01))
    delta = cagr_diff_pp(size["BIG NegNI"].loc["1994-01":], market.mkt.loc["1994-01":], 0.01)
    assert h1["A1"] == {"t": t, "p": one_sided_p(t), "pass": t >= 3 and one_sided_p(t) <= 0.00135}
    assert h1["A2"] == {"cagr_diff_pp": delta, "pass": delta >= 1.0}
    assert h1["overall_pass"] == (h1["A1"]["pass"] and h1["A2"]["pass"])
    assert set(result["issuance_shape"]) == set(BIG_COLUMNS)
    assert set(result["univariate_repurchasers"]) == {
        "portfolio",
        "includes_small_caps",
        "base",
        "stress",
    }
    assert [(item["start"], item["end"]) for item in h1["diagnostics"]["subperiods"]] == [
        ("1963-07", "1993-12"),
        ("1994-01", "2008-12"),
        ("2009-01", "2026-04"),
    ]
    assert set(h1["diagnostics"]["capm"]) == {"alpha_ann_pct", "beta"}
    assert (
        h1["diagnostics"]["stress"]["full_t"]
        == result["issuance_shape"]["BIG NegNI"]["stress"]["full_t"]
    )


@pytest.mark.parametrize(
    "panel,column,month,defect",
    [
        (0, "BIG NegNI", "1963-07-31", "missing"),
        (1, "< 0", "2026-04-30", "missing"),
        (2, "mkt", "2000-01-31", "duplicate"),
        (0, "BIG HiNI", "2000-01-31", "nan"),
        (1, "< 0", "2000-01-31", "inf"),
        (2, "rf", "2000-01-31", "nan"),
        (0, "BIG ZeroNI", "2000-01-31", "zero_portfolio_wealth"),
        (1, "< 0", "2000-01-31", "zero_portfolio_wealth"),
        (2, "mkt", "2000-01-31", "zero_market_wealth"),
    ],
)
def test_data_stops_before_statistics(
    panel: int, column: str, month: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject gaps, duplicates, invalid returns, and non-positive wealth first."""
    panels = list(_panels())
    date = pd.Timestamp(month)
    if defect == "missing":
        panels[panel] = panels[panel].drop(date)
    elif defect == "duplicate":
        panels[panel] = pd.concat([panels[panel], panels[panel].loc[[date]]])
    else:
        panels[panel].loc[date, column] = {
            "nan": np.nan,
            "inf": np.inf,
            "zero_portfolio_wealth": -1 + 0.02 / 12,
            "zero_market_wealth": -1,
        }[defect]

    def unexpected_statistic(*args: object, **kwargs: object) -> None:
        raise AssertionError("Data stops must precede statistics")

    monkeypatch.setattr(net_issuance_screen, "nw_t", unexpected_statistic)
    with pytest.raises(ValueError):
        screen(*panels)
