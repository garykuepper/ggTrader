"""Every driver that builds an S&P 500 universe takes ``--membership`` (slice M3).

The oracle drivers default to the frozen vintage-1 fixture so their recorded numbers reproduce;
the backfills and ``ggt lab`` have no default (``GGT_MEMBERSHIP`` or an error). The legacy-tag
runner cannot read a database, so it verifies the fixture against the file the legacy code reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ggTrader.data.membership import ORACLE_SPEC

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    """Import scripts/<name>.py by path (the scripts folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_m3_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_oracle_default_is_the_frozen_vintage_one_fixture() -> None:
    assert ORACLE_SPEC == "fixture:sp500_v1_2026-06-02"


def test_p5_defaults_to_the_oracle_fixture_and_accepts_db_one() -> None:
    p5 = _load("p5_wfo_validation")
    assert p5.parse_args([]).membership == ORACLE_SPEC
    args = p5.parse_args(["--membership", "db:1", "--expect-membership-sha", "ab"])
    assert (args.membership, args.expect_membership_sha) == ("db:1", "ab")


def test_make_tape_snapshot_defaults_to_the_oracle_fixture() -> None:
    mts = _load("make_tape_snapshot")
    assert mts.parse_args([]).membership == ORACLE_SPEC


def test_p3_and_p7_parsers_default_to_the_oracle_fixture() -> None:
    p3 = _load("p3_pit_parity")
    assert p3.build_parser().parse_args([]).membership == ORACLE_SPEC
    text = (SCRIPTS / "p7_live_golden.py").read_text()
    assert "add_membership_argument(parser, default=ORACLE_SPEC)" in text
    assert "membership=membership" in text


def test_backfills_have_no_default_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from ggTrader.data import membership
    from ggTrader.data.membership import MembershipSourceNotSelected

    membership.clear_source_cache()
    monkeypatch.delenv("GGT_MEMBERSHIP")
    backfill = _load("equity_backfill")
    assert backfill.resolve_symbols.__defaults__ == (None,)  # membership defaults to "not given"
    with pytest.raises(MembershipSourceNotSelected):
        backfill.resolve_symbols("sp500", *_span(), "SPY")  # falls to the environment: none set


def _span():
    import pandas as pd

    return pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")


def test_the_legacy_runner_verifies_the_file_the_legacy_code_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = _load("legacy_reference_runs")
    assert legacy.ORACLE_MEMBERSHIP == ORACLE_SPEC
    record = legacy.pin_membership(ORACLE_SPEC)
    assert record["membership"]["fixture"] == "sp500_v1_2026-06-02"
    assert record["membership_sha256"] == legacy.MEMBERSHIP_RECORD["membership_sha256"]
    manifest = json.loads(legacy.MEMBERSHIP_MANIFEST.read_text())["fixtures"]
    assert record["membership_sha256"] == manifest["sp500_v1_2026-06-02"]["content_sha256"]
    with pytest.raises(SystemExit, match="only fixture"):
        legacy.pin_membership("db:1")
    with pytest.raises(SystemExit, match="unknown membership fixture"):
        legacy.pin_membership("fixture:nope")
    # a legacy tree whose file is not the named fixture is refused, not recorded as v1
    from ggTrader.data.core import index_constituents as legacy_ic

    other = tmp_path / "other.csv.gz"
    other.write_bytes(b"not the fixture")
    monkeypatch.setattr(legacy_ic, "SP500_HISTORY_CSV", other)
    with pytest.raises(SystemExit, match="not fixture"):
        legacy.pin_membership(ORACLE_SPEC)
