"""Test the pre-registered monthly momentum screen with synthetic returns."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.momentum_screen import (
    cagr_diff_pp,
    capm,
    holm,
    net_excess,
    nw_t,
    one_sided_p,
    screen,
    worst_rolling_relative,
)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide complete monthly histories with a nonconstant market."""
    index = pd.date_range("1927-01-31", "2026-04-30", freq="ME")
    market = 0.005 + 0.002 * np.sin(np.arange(len(index)))
    sm = pd.DataFrame(
        {
            f"{size}_{rank}": market + 0.0001 * np.cos(np.arange(len(index)))
            for size in ("small", "big")
            for rank in ("lo", "mid", "hi")
        },
        index=index,
    )
    return sm, pd.DataFrame({"mkt": market, "rf": 0.001}, index=index)


def test_net_excess_subtracts_monthly_cost() -> None:
    """Charge one twelfth of the annual haircut each month."""
    winners = pd.Series(0.02, index=pd.date_range("2000-01-31", periods=12, freq="ME"))
    result = net_excess(winners, winners * 0.5, 0.012)
    assert result.tolist() == pytest.approx([0.009] * 12)


def test_net_excess_aligns_and_rejects_nan() -> None:
    """Use common dates without silently dropping missing observations."""
    winners = pd.Series([0.02, 0.03], index=[1, 2])
    market = pd.Series([0.01, 0.02], index=[2, 3])
    assert net_excess(winners, market, 0).to_dict() == pytest.approx({2: 0.02})
    with pytest.raises(ValueError):
        net_excess(winners, pd.Series([np.nan], index=[2]), 0)


@pytest.mark.parametrize(
    "pvals,expected",
    [
        ({"big": 0.0005, "small": 0.0010}, {"big": True, "small": True}),
        ({"big": 0.0009, "small": 0.0010}, {"big": False, "small": False}),
        ({"big": 0.0001, "small": 0.0020}, {"big": True, "small": False}),
        ({"big": 0.0010, "small": 0.0005}, {"big": True, "small": True}),
    ],
)
def test_holm_two_hypotheses(pvals: dict[str, float], expected: dict[str, bool]) -> None:
    """Stop rejecting after the first failed step, independent of input order."""
    assert holm(pvals) == expected


def test_cagr_diff_pp_constant_returns() -> None:
    """Annualize compounded monthly returns over all 120 observations."""
    winners = pd.Series([0.01] * 120)
    market = pd.Series([0.005] * 120)
    assert cagr_diff_pp(winners, market, 0) == pytest.approx((1.01**12 - 1.005**12) * 100, abs=1e-9)
    assert cagr_diff_pp(winners, market, 0.012) == pytest.approx(
        (1.009**12 - 1.005**12) * 100, abs=1e-9
    )


def test_nw_t_and_one_sided_p() -> None:
    """Use the standard error of the mean and the upper normal tail."""
    assert nw_t(pd.Series([0.01, 0.03, 0.02, 0.04]), lags=0) == pytest.approx(4.472135955)
    assert one_sided_p(3) == pytest.approx(0.0013498980316300933)
    assert nw_t(pd.Series([0.0] * 24)) == 0


def test_capm_reports_alpha_and_beta_only() -> None:
    """Recover the intercept and exposure without a mis-specified alpha t-test."""
    market = pd.Series(np.tile([-0.02, 0.0, 0.02, 0.0], 30))
    noise = np.tile([0.001, -0.001, 0.001, -0.001], 30)
    result = capm(0.002 + 1.5 * market + noise, market)
    assert result == pytest.approx({"alpha_ann_pct": 2.4, "beta": 1.5})


def test_screen_rebound_2009_uses_march_through_december_and_base_cost() -> None:
    """Exclude adjacent months and compound wealth ratios rather than return differences."""
    sm, mk = _panels()
    sm[:] = 0.0
    mk["mkt"] = 0.0
    sm.loc["2009-02", "big_hi"] = 0.9
    sm.loc["2010-01", "big_hi"] = 0.9
    sm.loc["2009-03", "big_hi"] = 0.2
    mk.loc["2009-03", "mkt"] = 0.1
    sm.loc["2009-12", "big_hi"] = -0.1
    result = screen(sm, mk)["groups"]
    assert result["big"]["rebound_2009"] == pytest.approx(
        {
            "gross_pct": (1.2 * 0.9 / 1.1 - 1) * 100,
            "net_pct": ((1.2 - 0.01 / 12) * (1 - 0.01 / 12) ** 8 * (0.9 - 0.01 / 12) / 1.1 - 1)
            * 100,
        }
    )
    assert result["small"]["rebound_2009"] == pytest.approx(
        {"gross_pct": (1 / 1.1 - 1) * 100, "net_pct": (0.9975**10 / 1.1 - 1) * 100}
    )


@pytest.mark.parametrize("recovers", [True, False])
def test_screen_relative_drawdown_tracks_peak_trough_and_recovery(recovers: bool) -> None:
    """Find a base-cost wealth-ratio drawdown and leave unrecovered durations null."""
    sm, mk = _panels()
    sm[:] = 0.01 / 12
    mk["mkt"] = 0.0
    # Big's net wealth is 1, 1.5, 1.5, 0.75, then 1.5 (or 0.75 forever).
    sm.loc["1927-02", "big_hi"] += 0.5
    sm.loc["1927-04", "big_hi"] += -0.4
    mk.loc["1927-04", "mkt"] = 0.2
    if recovers:
        sm.loc["1927-05", "big_hi"] += 1.0
    result = screen(sm, mk)["groups"]["big"]["relative_drawdown"]
    assert result["max_drawdown_pct"] == pytest.approx(-50.0)
    assert result["peak"] == "1927-03"
    assert result["trough"] == "1927-04"
    assert result["recovery"] == ("1927-05" if recovers else None)
    assert result["recovery_months"] == (1 if recovers else None)


def test_screen_relative_drawdown_includes_initial_wealth() -> None:
    """A first-month relative loss must count from the initial wealth of one."""
    sm, mk = _panels()
    sm[:] = 0.01 / 12
    mk["mkt"] = 0.0
    mk.loc["1927-01", "mkt"] = 0.25
    sm.loc["1927-02", "big_hi"] += 0.25
    result = screen(sm, mk)["groups"]["big"]["relative_drawdown"]
    assert result == {
        "max_drawdown_pct": pytest.approx(-20.0),
        "peak": "1926-12",
        "trough": "1927-01",
        "recovery": "1927-02",
        "recovery_months": 1,
    }


def test_worst_rolling_relative_compounds_wealth_ratio() -> None:
    """Identify the actual start and end of the worst compounded window."""
    index = pd.date_range("2000-01-31", periods=4, freq="ME")
    winners = pd.Series([0.1, -0.2, -0.1, 0.5], index=index)
    result = worst_rolling_relative(winners, pd.Series(0.0, index=index), months=2)
    assert result["start"] == "2000-02"
    assert result["end"] == "2000-03"
    assert result["relative_pct"] == pytest.approx(-28.0)


@pytest.mark.parametrize("panel", ["sm", "mk"])
def test_screen_window_coverage_raises(panel: str) -> None:
    """March 2026 cannot substitute for the pre-registered April endpoint."""
    sm, mk = _panels()
    if panel == "sm":
        sm = sm.iloc[:-1]
    else:
        mk = mk.iloc[:-1]
    with pytest.raises(ValueError, match="2026-04"):
        screen(sm, mk)


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_screen_rejects_missing_or_duplicate_month(defect: str) -> None:
    """Endpoint coverage alone must not hide gaps or repeated months."""
    sm, mk = _panels()
    sm = sm.drop(sm.index[100]) if defect == "missing" else pd.concat([sm, sm.iloc[[100]]])
    with pytest.raises(ValueError):
        screen(sm, mk)


@pytest.mark.parametrize("column", ["small_lo", "big_hi", "mkt", "rf"])
def test_screen_raises_on_nan_in_window(column: str) -> None:
    """Any required missing portfolio or factor value halts the screen."""
    sm, mk = _panels()
    frame = sm if column in sm else mk
    frame.loc[frame.index[100], column] = np.nan
    with pytest.raises(ValueError):
        screen(sm, mk)


def test_screen_pass_requires_both_tests() -> None:
    """A strong early history cannot rescue a negative post-publication edge."""
    sm, mk = _panels()
    sm["big_hi"] += np.where(sm.index.year < 1994, 0.02, -0.005)
    result = screen(sm, mk)
    assert result["groups"]["big"]["full_t"] >= 3.2
    assert result["holm"]["big"] is True
    assert result["groups"]["big"]["post_cagr_diff_pp"] < 1.0
    assert result["pass"]["big"] is False


def test_size_matched_is_winners_minus_same_size_mean() -> None:
    """Identical within-size portfolios have no size-matched edge or cost deduction."""
    sm, mk = _panels()
    sm[["small_lo", "small_mid", "small_hi"]] += 0.004
    result = screen(sm, mk)
    for group in ("big", "small"):
        assert result["groups"][group]["size_matched"]["cagr_diff_pp"] == pytest.approx(
            0.0, abs=1e-12
        )


@pytest.mark.parametrize(
    "big,small,next_text",
    [
        (False, False, "Momentum is closed and reported as NO-GO. No paid data."),
        (True, False, "Phase 2 on the S&P 500."),
        (
            False,
            True,
            "Skip Phase 2, since we have no small-cap PIT data. Phase 3 runs on paid small-cap "
            "data, and paper trading waits until Phase 3 passes.",
        ),
        (
            True,
            True,
            "Phases 2–3 test the S&P 500 version as the primary, and only it can authorize "
            "paper or real money. Small caps run in Phase 3 as a diagnostic only; deploying "
            "them needs a brief of their own.",
        ),
    ],
)
def test_screen_decision_and_diagnostics(big: bool, small: bool, next_text: str) -> None:
    """Route each gate outcome and preserve the requested diagnostic evidence."""
    sm, mk = _panels()
    for group, passes in (("big", big), ("small", small)):
        if passes:
            sm[f"{group}_hi"] += 0.01 + 0.001 * np.cos(np.arange(len(sm)))
    result = screen(sm, mk)
    assert result["pass"] == {"big": big, "small": small}
    assert result["next"] == next_text
    for group in ("big", "small"):
        assert set(result["groups"][group]) == {
            "full_t",
            "full_p",
            "post_cagr_diff_pp",
            "diag",
            "stress",
            "capm",
            "size_matched",
            "worst_12m",
            "rebound_2009",
            "relative_drawdown",
        }
        assert len(result["groups"][group]["diag"]) == 3
    json.dumps(result, allow_nan=False)


def test_screen_ignores_months_outside_fixed_window() -> None:
    """Extra months, including invalid values there, cannot change the screen."""
    sm, mk = _panels()
    expected = screen(sm, mk)
    sm.loc[pd.Timestamp("2026-05-31")] = np.nan
    mk.loc[pd.Timestamp("1926-12-31")] = np.nan
    assert screen(sm, mk) == expected


def test_driver_download_only_and_offline_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download mode freezes only; the default run writes provenance and screen results."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "momentum_french_screen.py"
    spec = importlib.util.spec_from_file_location("momentum_french_screen", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    sm, mk = _panels()
    manifest = {"frozen.zip": {"sha256": "test-checksum"}}
    downloads = []
    monkeypatch.setattr(driver, "download_and_freeze", lambda files: downloads.append(files))
    monkeypatch.setattr(driver, "verify_manifest", lambda: manifest)
    monkeypatch.setattr(driver, "load_size_momentum_vw", lambda: sm)
    monkeypatch.setattr(driver, "load_market_total_return", lambda: mk)
    out = tmp_path / "results" / "screen.json"
    monkeypatch.setattr("sys.argv", [str(path), "--download", "--out", str(out)])
    driver.main()
    assert downloads == [[driver.SIZE_MOM_FILE, driver.FACTORS_FILE]]
    assert not out.exists()
    monkeypatch.setattr("sys.argv", [str(path), "--out", str(out)])
    driver.main()
    result = json.loads(out.read_text())
    assert downloads == [[driver.SIZE_MOM_FILE, driver.FACTORS_FILE]]
    assert result["provenance"] == manifest
    assert result["brief"] == "docs/research/briefs/2026-10-20-momentum-retest.md"
    assert result["runtime_sec"] >= 0
    assert result["pass"] == {"big": False, "small": False}
