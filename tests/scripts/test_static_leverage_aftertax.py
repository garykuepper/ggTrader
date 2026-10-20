"""Tests for static SPY/SSO fractional leverage after-tax simulation."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest


def test_distribution_taxed_as_short_term_goes_to_st_bucket():
    from scripts.static_leverage_aftertax import Book

    b = Book(st_rate=0.24, lt_rate=0.15, wash=False)
    b.cash = 1000.0
    b.buy("SSO", 1000.0, 10.0, pd.Timestamp("2024-01-02"))
    b.distributions("SSO", 1.0, 10.0, pd.Timestamp("2024-06-03"), taxed_as="st")
    # Shares held before reinvestment: $1000 at 10 * (1 + 1 bp) (COST_BPS = 1.0), times $1.
    assert b.year_st == pytest.approx(1000.0 / 10.001, rel=1e-12)
    assert b.year_lt == 0.0


def test_distribution_default_taxed_as_lt():
    from scripts.static_leverage_aftertax import Book

    b = Book(st_rate=0.24, lt_rate=0.15, wash=False)
    b.cash = 1000.0
    b.buy("SSO", 1000.0, 10.0, pd.Timestamp("2024-01-02"))
    b.distributions("SSO", 1.0, 10.0, pd.Timestamp("2024-06-03"))
    assert b.year_lt == pytest.approx(1000.0 / 10.001, rel=1e-12)
    assert b.year_st == 0.0


def test_distribution_invalid_tax_label_raises():
    from scripts.static_leverage_aftertax import Book

    b = Book(st_rate=0.24, lt_rate=0.15, wash=False)
    b.cash = 1000.0
    b.buy("SSO", 1000.0, 10.0, pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError, match="Invalid tax label"):
        b.distributions("SSO", 1.0, 10.0, pd.Timestamp("2024-06-03"), taxed_as="ordinary")


def test_run_dist_tax_routing_and_backward_compatibility():
    from scripts.static_leverage_aftertax import run

    dates = pd.date_range("2024-01-02", "2024-03-28", freq="B")
    close = pd.DataFrame(
        {"SPY": 100.0, "SSO": 50.0, "BIL": 100.0},
        index=dates,
    )
    dist = pd.DataFrame(0.0, index=dates, columns=["SPY", "SSO", "BIL"])
    dist.loc[dates[10], "SSO"] = 1.0

    res_default = run(
        close,
        dist,
        start="2024-01-02",
        end="2024-03-28",
        w_sso=0.5,
        band=0.1,
        st=0.24,
        lt=0.15,
        wash=False,
    )
    res_lt = run(
        close,
        dist,
        start="2024-01-02",
        end="2024-03-28",
        w_sso=0.5,
        band=0.1,
        st=0.24,
        lt=0.15,
        wash=False,
        dist_tax={"SSO": "lt"},
    )
    res_st = run(
        close,
        dist,
        start="2024-01-02",
        end="2024-03-28",
        w_sso=0.5,
        band=0.1,
        st=0.24,
        lt=0.15,
        wash=False,
        dist_tax={"SSO": "st"},
    )

    # Default behaviour is identical to explicit "lt"
    assert res_default["after_tax_terminal"] == res_lt["after_tax_terminal"]
    # ST taxation at higher tax rate (24% vs 15%) reduces terminal wealth
    assert res_st["after_tax_terminal"] < res_lt["after_tax_terminal"]


def test_run_invalid_dist_tax_label_raises():
    from scripts.static_leverage_aftertax import run

    dates = pd.date_range("2024-01-02", "2024-01-10", freq="B")
    close = pd.DataFrame({"SPY": 100.0, "SSO": 50.0, "BIL": 100.0}, index=dates)
    dist = pd.DataFrame(0.0, index=dates, columns=["SPY", "SSO", "BIL"])

    with pytest.raises(ValueError, match="Invalid tax label for SSO"):
        run(
            close,
            dist,
            start="2024-01-02",
            end="2024-01-10",
            w_sso=0.5,
            band=0.1,
            st=0.24,
            lt=0.15,
            wash=False,
            dist_tax={"SSO": "invalid"},
        )


def test_cli_flags_registered():
    res = subprocess.run(
        [sys.executable, "scripts/static_leverage_aftertax.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--sso-dist-tax" in res.stdout
    assert "--bil-dist-tax" in res.stdout
