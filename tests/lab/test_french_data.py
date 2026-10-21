"""Offline tests for frozen Ken French monthly data."""

import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ggTrader.lab.french_data import (
    FACTORS_FILE,
    FRENCH_BASE_URL,
    SIZE_MOM_FILE,
    download_and_freeze,
    load_market_total_return,
    load_size_momentum_vw,
    parse_french_monthly,
    verify_manifest,
)

SIZE_MOM_TEXT = """French library preamble

  Average Value Weighted Returns -- Monthly
,SMALL LoPRIOR,ME1 PRIOR2,SMALL HiPRIOR,BIG LoPRIOR,ME2 PRIOR2,BIG HiPRIOR
192701,1,2,3,4,5,6
192702,-1,-2,-3,-4,-5,-6

  Average Equal Weighted Returns -- Monthly
,SMALL LoPRIOR,ME1 PRIOR2,SMALL HiPRIOR,BIG LoPRIOR,ME2 PRIOR2,BIG HiPRIOR
192701,11,12,13,14,15,16
"""
FACTORS_TEXT = """French factors preamble

,Mkt-RF,SMB,HML,RF
192701,1.00,2.00,3.00,0.30
192702,-1.00,2.00,3.00,0.20
  Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,10,20,30,4
"""
VW_SECTION = "Average Value Weighted Returns -- Monthly"


def _zip_text(text: str) -> bytes:
    """Build a real in-memory CSV archive from an inline fixture."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fixture.csv", text)
    return buffer.getvalue()


def test_parse_value_weighted_section_only() -> None:
    """Select VW rather than EW and convert percent returns and month ends."""
    result = parse_french_monthly(SIZE_MOM_TEXT, VW_SECTION)
    expected = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06]],
        columns=[
            "SMALL LoPRIOR",
            "ME1 PRIOR2",
            "SMALL HiPRIOR",
            "BIG LoPRIOR",
            "ME2 PRIOR2",
            "BIG HiPRIOR",
        ],
        index=pd.to_datetime(["1927-01-31", "1927-02-28"]),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_parse_marks_missing_codes_nan() -> None:
    """Missing sentinels must not turn into very large negative returns."""
    result = parse_french_monthly(" , winner , loser , valid \n200002,-99.99,-999,-1.00\n", None)
    assert result.index[0] == pd.Timestamp("2000-02-29")
    assert result.loc["2000-02-29", ["winner", "loser"]].isna().all()
    assert result.loc["2000-02-29", "valid"] == pytest.approx(-0.01)


def test_load_size_momentum_raises_on_missing_column(tmp_path: Path) -> None:
    """An incomplete source header cannot silently drop a portfolio."""
    raw = _zip_text(SIZE_MOM_TEXT.replace("BIG HiPRIOR", "unexpected"))
    download_and_freeze([SIZE_MOM_FILE], dest=tmp_path, http_get=lambda url: raw)
    with pytest.raises(KeyError, match="BIG HiPRIOR"):
        load_size_momentum_vw(tmp_path)


def test_load_size_momentum_normalizes_headers(tmp_path: Path) -> None:
    """Case and whitespace differences retain all six portfolio identities."""
    text = SIZE_MOM_TEXT.replace("SMALL HiPRIOR", " small  hiprior ")
    raw = _zip_text(text)
    download_and_freeze([SIZE_MOM_FILE], dest=tmp_path, http_get=lambda url: raw)
    result = load_size_momentum_vw(tmp_path)
    assert list(result.columns) == [
        "small_lo",
        "small_mid",
        "small_hi",
        "big_lo",
        "big_mid",
        "big_hi",
    ]
    assert result.loc["1927-01-31"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])


def test_market_total_return_adds_rf(tmp_path: Path) -> None:
    """Market total return adds RF once and excludes the annual table."""
    raw = _zip_text(FACTORS_TEXT)
    download_and_freeze([FACTORS_FILE], dest=tmp_path, http_get=lambda url: raw)
    result = load_market_total_return(tmp_path)
    expected = pd.DataFrame(
        {"mkt": [0.013, -0.008], "rf": [0.003, 0.002]},
        index=pd.to_datetime(["1927-01-31", "1927-02-28"]),
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("filename", [SIZE_MOM_FILE, FACTORS_FILE])
def test_verify_manifest_detects_tamper(tmp_path: Path, filename: str) -> None:
    """Both verification and loading reject changed frozen bytes."""
    download_and_freeze([filename], dest=tmp_path, http_get=lambda url: _zip_text(FACTORS_TEXT))
    (tmp_path / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=filename):
        verify_manifest(tmp_path)
    loader = load_size_momentum_vw if filename == SIZE_MOM_FILE else load_market_total_return
    with pytest.raises(ValueError, match=filename):
        loader(tmp_path)


def test_download_refuses_overwrite(tmp_path: Path) -> None:
    """A collision must leave the frozen file and manifest intact."""
    download_and_freeze([SIZE_MOM_FILE], dest=tmp_path, http_get=lambda url: b"original")
    original_manifest = (tmp_path / "MANIFEST.json").read_bytes()
    with pytest.raises(FileExistsError):
        download_and_freeze(
            [FACTORS_FILE, SIZE_MOM_FILE], dest=tmp_path, http_get=lambda url: b"replacement"
        )
    assert (tmp_path / SIZE_MOM_FILE).read_bytes() == b"original"
    assert (tmp_path / "MANIFEST.json").read_bytes() == original_manifest
    assert not (tmp_path / FACTORS_FILE).exists()


def test_download_records_manifest_and_supports_force(tmp_path: Path) -> None:
    """Manifest metadata follows the bytes, including explicitly forced replacements."""
    dest = tmp_path / "french"
    download_and_freeze([SIZE_MOM_FILE], dest=dest, http_get=lambda url: b"original")
    download_and_freeze([FACTORS_FILE], dest=dest, http_get=lambda url: b"factors")
    manifest = download_and_freeze(
        [SIZE_MOM_FILE], dest=dest, http_get=lambda url: b"replacement", force=True
    )
    assert set(manifest) == {SIZE_MOM_FILE, FACTORS_FILE}
    entry = manifest[SIZE_MOM_FILE]
    assert entry["url"] == FRENCH_BASE_URL + SIZE_MOM_FILE
    assert entry["sha256"] == hashlib.sha256(b"replacement").hexdigest()
    assert entry["bytes"] == 11
    assert datetime.fromisoformat(entry["downloaded_utc"]).utcoffset().total_seconds() == 0
    assert (dest / SIZE_MOM_FILE).read_bytes() == b"replacement"
    assert json.loads((dest / "MANIFEST.json").read_text()) == manifest
    assert verify_manifest(dest) == manifest


def test_verify_manifest_detects_missing_file(tmp_path: Path) -> None:
    """Missing archives or manifests fail with a descriptive ValueError."""
    with pytest.raises(ValueError, match="MANIFEST.json"):
        verify_manifest(tmp_path)
    download_and_freeze([SIZE_MOM_FILE], dest=tmp_path, http_get=lambda url: b"original")
    (tmp_path / SIZE_MOM_FILE).unlink()
    with pytest.raises(ValueError, match=SIZE_MOM_FILE):
        verify_manifest(tmp_path)


@pytest.mark.parametrize("text,title", [(SIZE_MOM_TEXT, "absent"), ("preamble only", None)])
def test_parse_rejects_missing_table(text: str, title: str | None) -> None:
    """Missing sections or monthly rows must not yield plausible empty data."""
    with pytest.raises(ValueError):
        parse_french_monthly(text, title)
