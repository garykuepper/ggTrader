"""Check the SEC ticker-label diagnostic using invented filings."""

import json
from zipfile import ZipFile

import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "sec_ticker_coverage.py")
pytestmark = pytest.mark.study


def test_coverage_counts_labels_without_selecting_an_issuer(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps([{"symbol": s} for s in ["abc", "BRK.B", "XYZ"]]))
    archive = tmp_path / "filings.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(
            "sub.tsv",
            "adsh\tcik\tform\tfiled\n"
            "a\t1\t10-Q\t20210628\n"
            "b\t2\t8-K\t20210628\n"
            "c\t3\t10-Q\t20210629\n"
            "d\t4\t4\t20210601\n",
        )
        z.writestr(
            "txt.tsv",
            "adsh\ttag\tvalue\n"
            "a\tTradingSymbol\tABC\n"
            "a\tTradingSymbol\tBRK.B\n"
            "b\tTradingSymbol\tABC\n"
            "c\tTradingSymbol\tXYZ\n"
            "d\tTradingSymbol\tXYZ\n"
            "a\tOther\tXYZ\n",
        )
    result = module.coverage(archive, snapshot, "2021-06-29")
    assert result["counts"] == {
        "snapshot_distinct_normalized_symbols": 3,
        "qualifying_submissions": 2,
        "nonempty_symbol_facts": 3,
        "snapshot_matched_symbols": 2,
        "matched_single_cik": 1,
        "matched_multiple_ciks": 1,
        "snapshot_unmatched_symbols": 1,
        "issuers_with_multiple_ticker_labels": 1,
    }
    assert result["matched_multiple_cik_symbols"] == ["ABC"]


def test_rejects_invalid_comparison_date(tmp_path):
    with pytest.raises(ValueError, match="date"):
        module.coverage(tmp_path / "unused.zip", tmp_path / "unused.json", "20210629")


@pytest.mark.parametrize(
    "submissions",
    [
        "a\t1\t10-Q\t20210230\n",
        "a\t1\t10-Q\t20210601\na\t2\t10-Q\t20210602\n",
    ],
)
def test_rejects_malformed_or_duplicate_submissions(tmp_path, submissions):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('[{"symbol":"ABC"}]')
    archive = tmp_path / "filings.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("sub.tsv", "adsh\tcik\tform\tfiled\n" + submissions)
        z.writestr("txt.tsv", "adsh\ttag\tvalue\na\tTradingSymbol\tABC\n")
    with pytest.raises(ValueError, match="filing|accession"):
        module.coverage(archive, snapshot, "2021-06-29")


def test_two_archives_combine_labels_and_report_overlap(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('[{"symbol":"ABC"},{"symbol":"XYZ"},{"symbol":"NONE"}]')
    archives = []
    for n, ticker_rows in enumerate(
        ("a\tTradingSymbol\tABC\n", "b\tTradingSymbol\tABC\nb\tTradingSymbol\tXYZ\n")
    ):
        archive = tmp_path / f"{n}.zip"
        with ZipFile(archive, "w") as z:
            z.writestr("sub.tsv", f"adsh\tcik\tform\tfiled\n{'ab'[n]}\t{n + 1}\t10-Q\t20210601\n")
            z.writestr("txt.tsv", "adsh\ttag\tvalue\n" + ticker_rows)
        archives.append(archive)
    result = module.coverage(archives[0], snapshot, "2021-06-29", additional_archives=archives[1:])
    assert result["counts"]["snapshot_matched_symbols"] == 2
    assert result["counts"]["snapshot_unmatched_symbols"] == 1
    assert result["matched_multiple_cik_symbols"] == ["ABC"]
    assert result["per_archive_matched"] == [1, 2]
    assert result["matched_in_every_archive"] == 1
