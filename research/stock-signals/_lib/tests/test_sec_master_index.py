"""Offline checks for SEC master-index listing-event metadata."""

import io
import zipfile

import pytest
import sec_master_index

pytestmark = pytest.mark.study

HEADER = "CIK|Company Name|Form Type|Date Filed|Filename\n-----\n"


def archive(rows):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("master.idx", "Description: SEC example\n" + HEADER + "\n".join(rows) + "\n")
    return target.getvalue()


def test_groups_duplicate_accessions_without_losing_indexed_parties():
    rows = [
        "1005409|DIME COMMUNITY BANCSHARES INC|25-NSE|2021-01-29|"
        "edgar/data/1005409/0001354457-21-000134.txt",
        "1354457|Nasdaq Stock Market LLC|25-NSE|2021-01-29|"
        "edgar/data/1354457/0001354457-21-000134.txt",
        "764401|INSTEEL INDUSTRIES INC|25|2021-03-15|edgar/data/764401/0001437749-21-006027.txt",
        "123|OTHER|10-K|2021-03-15|edgar/data/123/0000000123-21-000001.txt",
    ]
    result = sec_master_index.parse_listing_archive(archive(rows))
    assert len(result) == 2
    filing = next(r for r in result if r["accession"] == "0001354457-21-000134")
    assert filing["form"] == "25-NSE" and filing["filing_date"] == "2021-01-29"
    assert {p["cik"] for p in filing["parties"]} == {"0001005409", "0001354457"}
    assert all("ticker" not in party for party in filing["parties"])


@pytest.mark.parametrize(
    "rows",
    [
        ["123|NAME|25-NSE|2021-01-29|edgar/data/999/0000000123-21-000001.txt"],
        ["123|NAME|25-NSE|2021-02-30|edgar/data/123/0000000123-21-000001.txt"],
        ["123|NAME|25-NSE|2021-01-29|edgar/data/123/../0000000123-21-000001.txt"],
        ["123|NAME|25-NSE|2021-01-29|edgar/data/123/0000000123-21-000001.txt"] * 2,
        [
            "123|NAME|25-NSE|2021-01-29|edgar/data/123/0000000123-21-000001.txt",
            "456|OTHER|25|2021-01-29|edgar/data/456/0000000123-21-000001.txt",
        ],
        ["bad|row"],
    ],
)
def test_rejects_malformed_or_conflicting_index_rows(rows):
    with pytest.raises(ValueError):
        sec_master_index.parse_listing_archive(archive(rows))


def test_rejects_invalid_archive():
    with pytest.raises(ValueError):
        sec_master_index.parse_listing_archive(b"not a zip")


def test_preserves_utf8_party_names_without_mojibake():
    rows = ["123|CAFÉ INC|25|2021-03-15|edgar/data/123/0000000123-21-000001.txt"]
    assert (
        sec_master_index.parse_listing_archive(archive(rows))[0]["parties"][0]["name"] == "CAFÉ INC"
    )


def test_invalid_text_encoding_fails_closed():
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("master.idx", HEADER.encode() + b"123|\xff|25|2021-03-15|file.txt\n")
    with pytest.raises(ValueError, match="archive"):
        sec_master_index.parse_listing_archive(target.getvalue())
