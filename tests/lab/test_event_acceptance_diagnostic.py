"""Check the offline acceptance-source diagnostic without network or database access."""

import hashlib
import json
from pathlib import Path

import pytest

from ggTrader.lab.event_acceptance_diagnostic import compare_acceptance_pair, diagnose_cache


@pytest.mark.parametrize(
    ("header", "api", "filed", "offset", "consistent"),
    [
        ("2024-08-01T13:11:47", "2024-08-01T21:11:47Z", "2024-08-01", 8, True),
        ("2024-01-23T18:55:49", "2024-01-24T04:55:49Z", "2024-01-23", 10, False),
        ("2024-08-01T08:00:00", "2024-08-01T08:00:00Z", "2024-08-01", 0, False),
    ],
)
def test_clock_interpretations(
    header: str, api: str, filed: str, offset: int, consistent: bool
) -> None:
    result = compare_acceptance_pair(header, api, filed, filed)
    assert result["wall_clock_difference_hours"] == offset
    assert result["header_consistent"] is True
    assert result["api_consistent"] is consistent
    assert result["twice_offset_pattern"] is (offset in (8, 10))


def test_conflicting_filing_dates_do_not_pass() -> None:
    result = compare_acceptance_pair(
        "2024-08-01T13:11:47", "2024-08-01T21:11:47Z", "2024-08-01", "2024-08-02"
    )
    assert result["header_consistent"] is False
    assert result["api_consistent"] is False


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        diagnose_cache(tmp_path)


@pytest.fixture
def cached_pair(tmp_path: Path) -> Path:
    """Write one issuer/accession pair using the production cache-manifest shape."""
    accession = "0001209191-24-002111"
    api = {
        "accessionNumber": [accession],
        "form": ["4"],
        "filingDate": ["2024-01-23"],
        "primaryDocument": ["ownership.xml"],
        "acceptanceDateTime": ["2024-01-24T04:55:49Z"],
    }
    header = (
        f"ACCESSION NUMBER: {accession}\n"
        "CONFORMED SUBMISSION TYPE: 4\n"
        "<ACCEPTANCE-DATETIME>20240123185549\n"
        "FILED AS OF DATE: 20240123\n"
        "ISSUER:\n\tCENTRAL INDEX KEY: 909832\n"
        "</SEC-HEADER>\n"
    )
    entries = []
    for name, url, raw in (
        (
            "api.raw",
            "https://data.sec.gov/submissions/CIK0000909832.json",
            json.dumps({"filings": {"recent": api}}),
        ),
        (
            "header.raw",
            f"https://www.sec.gov/Archives/edgar/data/909832/000120919124002111/{accession}.txt",
            header,
        ),
    ):
        (tmp_path / name).write_text(raw)
        entries.append(
            {"url": url, "cache_file": name, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        )
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": entries}))
    return tmp_path


def test_hash_verified_cache_join(cached_pair: Path) -> None:
    result = diagnose_cache(cached_pair)
    assert result["pair_count"] == result["header_consistent_count"] == 1
    assert result["api_inconsistent_count"] == result["twice_offset_pattern_count"] == 1
    assert result["pairs"][0]["issuer_cik"] == "909832"
    assert result["pairs"][0]["api_sources"][0]["cache_file"] == "api.raw"


def test_corrupt_cache_fails_closed(cached_pair: Path) -> None:
    (cached_pair / "header.raw").write_text("corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        diagnose_cache(cached_pair)


def test_missing_api_pair_fails_closed(cached_pair: Path) -> None:
    manifest_path = cached_pair / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = manifest["files"][1:]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing or inconsistent cached API rows"):
        diagnose_cache(cached_pair)
