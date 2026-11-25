"""Tests for streaming all-form name index lookups over cached SEC quarterly archives."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import sec_filings
import sec_name_index
from sec_client import SecClient
from sec_name_index import (
    MAX_ARCHIVE_BYTES,
    IndexedFiling,
    MissingArchive,
    cached_archives,
    filings_of,
    find_filers,
    parse_all_forms,
)

pytestmark = pytest.mark.study

DEFAULT_UA = "ggTrader Study contact@example.test"
INDEX_PREAMBLE = (
    "Description: Master Index of EDGAR Dissemination Feed\n"
    "Last Data Received: March 31, 2020\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)


def make_master_zip(
    rows: list[tuple[str, str, str, str]],
    *,
    preamble: str = INDEX_PREAMBLE,
    member_name: str = "master.idx",
) -> bytes:
    """Build a small valid master.zip containing ``(cik, name, form, filed)`` rows."""
    lines = []
    for idx, (cik, name, form, filed) in enumerate(rows):
        cik_num = int(cik) if cik.isdigit() else 1
        accession = f"{cik_num:010d}-20-{idx:06d}"
        lines.append(f"{cik}|{name}|{form}|{filed}|edgar/data/{cik_num}/{accession}.txt\n")
    return make_raw_zip(preamble + "".join(lines), member_name=member_name)


def make_raw_zip(text: str | bytes, *, member_name: str = "master.idx") -> bytes:
    """Build a zip archive holding one member with raw text or byte contents."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, text)
    return buffer.getvalue()


def seed_cache(cache_dir: Path, url: str, body: bytes) -> tuple[Path, Path]:
    """Write a verified ``SecClient`` cache entry directly on disk without using the network."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    body_path = cache_dir / f"{key}.body"
    meta_path = cache_dir / f"{key}.meta.json"
    body_path.write_bytes(body)
    meta = {
        "url": url,
        "sha256": hashlib.sha256(body).hexdigest(),
        "fetched_at": "2026-11-24T00:00:00+00:00",
        "status": 200,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body_path, meta_path


class CountingRefusingOpener:
    """Fake opener that counts any attempted network open and immediately fails the test."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"Unexpected network open for {request.full_url}")


def test_parse_all_forms_keeps_10k_8k_and_defm14a() -> None:
    archive = make_master_zip(
        [
            ("0000100001", "ACME CORP", "10-K", "2020-02-15"),
            ("100001", "ACME CORP", "8-K", "2020-03-01"),
            ("100001", "ACME CORP", "DEFM14A", "2020-03-20"),
        ]
    )

    filings = parse_all_forms(archive)

    assert filings == [
        IndexedFiling(
            cik=100001,
            name="ACME CORP",
            form="10-K",
            filed=date(2020, 2, 15),
            accession="0000100001-20-000000",
        ),
        IndexedFiling(
            cik=100001,
            name="ACME CORP",
            form="8-K",
            filed=date(2020, 3, 1),
            accession="0000100001-20-000001",
        ),
        IndexedFiling(
            cik=100001,
            name="ACME CORP",
            form="DEFM14A",
            filed=date(2020, 3, 20),
            accession="0000100001-20-000002",
        ),
    ]


def test_accession_is_read_from_filename() -> None:
    archive = make_raw_zip(
        INDEX_PREAMBLE
        + "1234|EXAMPLE CORP|8-K|2005-06-15|edgar/data/1234/0001104659-05-024772.txt\n"
        + "1234|EXAMPLE CORP|8-K|2005-06-15|edgar/data/1234/0001104659-05-024770.txt\n"
    )

    filings = parse_all_forms(archive)
    assert [f.accession for f in filings] == [
        "0001104659-05-024772",
        "0001104659-05-024770",
    ]
    assert filings[0] == IndexedFiling(
        cik=1234,
        name="EXAMPLE CORP",
        form="8-K",
        filed=date(2005, 6, 15),
        accession="0001104659-05-024772",
    )

    # filings_of breaks ties by accession as the fourth sort key: (filed, form, name, accession).
    sorted_filings = filings_of([("2005Q2", archive)], {1234})
    assert [f.accession for f in sorted_filings[1234]] == [
        "0001104659-05-024770",
        "0001104659-05-024772",
    ]


@pytest.mark.parametrize(
    "bad_filename",
    [
        "edgar/data/1234/0001104659-05-024772.htm",
        "edgar/data/1234/0001104659-05-024772",
        "edgar/data/1234/.txt",
        ".txt",
        "edgar/data/1234/   .txt",
        "",
    ],
)
def test_bad_filename_raises_with_line_number(bad_filename: str) -> None:
    valid_line = "1234|EXAMPLE CORP|8-K|2005-06-15|edgar/data/1234/0001104659-05-024772.txt\n"
    bad_line = f"1234|EXAMPLE CORP|8-K|2005-06-16|{bad_filename}\n"
    archive = make_raw_zip(INDEX_PREAMBLE + valid_line + bad_line)

    with pytest.raises(ValueError, match=r"line 6\b"):
        parse_all_forms(archive)


def test_find_filers_keeps_only_wanted_names() -> None:
    unrelated = [
        (str(200000 + i), f"UNRELATED CO {i} INC", "10-K", "2020-02-10") for i in range(10)
    ]
    wanted_row = ("0000999888", "Alpha Technologies, Inc.", "8-K", "2020-02-12")
    archive = make_master_zip([*unrelated[:5], wanted_row, *unrelated[5:]])

    def matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        if raw_name == "Alpha Technologies, Inc.":
            return {"PERMNO-12345": frozenset({"v1"})}
        return {}

    matched = find_filers([("2020Q1", archive)], matcher)

    assert matched == {999888: {"PERMNO-12345": frozenset({"v1"})}}


def test_matcher_prefix_and_labels() -> None:
    archive = make_master_zip(
        [
            ("111111", "ATHLON ENERGY INC", "10-K", "2020-02-10"),
            ("222222", "EXACT TARGET CORP", "8-K", "2020-02-11"),
            ("333333", "UNRELATED COMPANY", "10-Q", "2020-02-12"),
        ]
    )

    def matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        hits: dict[str, frozenset[str]] = {}
        if raw_name.startswith("ATHLON ENERGY IN"):
            hits["R1"] = frozenset({"v1.1"})
        if raw_name == "EXACT TARGET CORP":
            hits["R2"] = frozenset({"v1"})
        return hits

    matched = find_filers([("2020Q1", archive)], matcher)

    assert matched == {
        111111: {"R1": frozenset({"v1.1"})},
        222222: {"R2": frozenset({"v1"})},
    }


def test_labels_are_unioned_across_archives() -> None:
    archive_q1 = make_master_zip([("444444", "MERGED CO INC", "10-K", "2020-02-15")])
    archive_q2 = make_master_zip([("444444", "MERGED CO INC /DE/", "8-K", "2020-05-15")])

    def matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        if raw_name == "MERGED CO INC":
            return {"R1": frozenset({"v1"})}
        if raw_name == "MERGED CO INC /DE/":
            return {"R1": frozenset({"v1.1"})}
        return {}

    matched = find_filers([("2020Q1", archive_q1), ("2020Q2", archive_q2)], matcher)

    assert matched == {444444: {"R1": frozenset({"v1", "v1.1"})}}


def test_matcher_called_once_per_distinct_name_per_archive() -> None:
    archive_q1 = make_master_zip(
        [
            ("100001", "REPEATED CORP", "10-K", "2020-02-10"),
            ("100001", "REPEATED CORP", "8-K", "2020-02-11"),
            ("100002", "OTHER CORP", "10-Q", "2020-02-12"),
        ]
    )
    archive_q2 = make_master_zip(
        [
            ("100001", "REPEATED CORP", "10-Q", "2020-05-10"),
            ("100001", "REPEATED CORP", "8-K", "2020-05-11"),
        ]
    )
    calls: list[str] = []

    def counting_matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        calls.append(raw_name)
        if raw_name == "REPEATED CORP":
            return {"R1": frozenset({"v1"})}
        return {}

    matched = find_filers([("2020Q1", archive_q1), ("2020Q2", archive_q2)], counting_matcher)

    assert matched == {100001: {"R1": frozenset({"v1"})}}
    # Cache is cleared between archives, so REPEATED CORP is evaluated once per archive (2 total).
    assert calls == ["REPEATED CORP", "OTHER CORP", "REPEATED CORP"]


def test_no_match_returns_empty() -> None:
    archive = make_master_zip(
        [
            ("100001", "ACME CORP", "10-K", "2020-02-15"),
            ("100002", "BETA CORP", "8-K", "2020-03-01"),
        ]
    )

    matched = find_filers([("2020Q1", archive)], lambda _name: {})

    assert matched == {}


def test_filings_of_returns_all_form_types_sorted() -> None:
    archive_q2 = make_master_zip(
        [
            ("555555", "GAMMA CORP", "DEFM14A", "2020-05-10"),
            ("777777", "OTHER CORP", "10-K", "2020-05-01"),
            ("555555", "GAMMA CORP", "8-K", "2020-04-20"),
        ]
    )
    archive_q1 = make_master_zip(
        [
            ("555555", "GAMMA CORP", "10-K", "2020-02-28"),
            ("888888", "NOISE INC", "4", "2020-02-27"),
        ]
    )

    result = filings_of([("2020Q2", archive_q2), ("2020Q1", archive_q1)], {555555})

    assert result == {
        555555: [
            IndexedFiling(
                cik=555555,
                name="GAMMA CORP",
                form="10-K",
                filed=date(2020, 2, 28),
                accession="0000555555-20-000000",
            ),
            IndexedFiling(
                cik=555555,
                name="GAMMA CORP",
                form="8-K",
                filed=date(2020, 4, 20),
                accession="0000555555-20-000002",
            ),
            IndexedFiling(
                cik=555555,
                name="GAMMA CORP",
                form="DEFM14A",
                filed=date(2020, 5, 10),
                accession="0000555555-20-000000",
            ),
        ]
    }


def test_missing_archive_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opener = CountingRefusingOpener()
    client = SecClient(DEFAULT_UA, tmp_path / "cache", opener=opener)

    with pytest.raises(MissingArchive, match="2020Q2"):
        list(cached_archives(client, [(2020, 2)]))

    assert opener.calls == 0
    assert client.network_fetches == 0

    # Also prove a default-opener SecClient cannot trigger urllib when reading cached archives.
    default_opens = 0

    def forbid_default_open(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal default_opens
        default_opens += 1
        raise AssertionError("Default urllib opener must never be called by cached_archives")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbid_default_open)
    default_client = SecClient(DEFAULT_UA, tmp_path / "cache")
    with pytest.raises(MissingArchive, match="2020Q2"):
        list(cached_archives(default_client, [(2020, 2)]))
    assert default_opens == 0
    assert default_client.network_fetches == 0


def test_truncated_cached_body_raises_missing_archive_without_network(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = sec_filings.master_index_url(2020, 3)
    archive = make_master_zip([("100001", "ACME CORP", "10-K", "2020-08-01")])
    body_path, _ = seed_cache(cache_dir, url, archive)
    # Cut the cached body short after writing metadata so SHA-256 verification fails.
    body_path.write_bytes(archive[:20])

    opener = CountingRefusingOpener()
    client = SecClient(DEFAULT_UA, cache_dir, opener=opener)

    with pytest.raises(MissingArchive, match="2020Q3"):
        list(cached_archives(client, [(2020, 3)]))

    assert opener.calls == 0
    assert client.network_fetches == 0


@pytest.mark.parametrize(
    "case",
    ["oversized", "truncated_zip", "non_zip", "bad_header", "wrong_member"],
)
def test_invalid_archive_names_quarter_across_cache_and_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    valid = make_master_zip([("100001", "ACME CORP", "10-K", "2020-05-01")])
    if case == "oversized":
        # Shrink the ceiling below a valid zip so this proves the size cap rejects it directly.
        monkeypatch.setattr(sec_name_index, "MAX_ARCHIVE_BYTES", len(valid) - 1)
        body = valid
    elif case == "truncated_zip":
        body = valid[: len(valid) // 2]
    elif case == "non_zip":
        body = b"<!DOCTYPE html><html><body>Request Rate Threshold Exceeded</body></html>"
    elif case == "wrong_member":
        body = make_master_zip(
            [("100001", "ACME CORP", "10-K", "2020-05-01")], member_name="other.idx"
        )
    else:
        body = make_raw_zip("100001|ACME CORP|10-K|2020-05-01|edgar/data/100001/x.txt\n")

    cache_dir = tmp_path / "cache"
    seed_cache(cache_dir, sec_filings.master_index_url(2020, 2), body)
    opener = CountingRefusingOpener()
    client = SecClient(DEFAULT_UA, cache_dir, opener=opener)

    with pytest.raises(MissingArchive, match="2020Q2") as cache_exc:
        list(cached_archives(client, [(2020, 2)]))
    assert isinstance(cache_exc.value.__cause__, ValueError)
    if case == "oversized":
        assert "exceeds" in str(cache_exc.value)
    assert opener.calls == 0
    assert client.network_fetches == 0

    with pytest.raises(MissingArchive, match="2020Q2") as pass1_exc:
        find_filers([("2020Q2", body)], lambda _name: {})
    assert isinstance(pass1_exc.value.__cause__, ValueError)

    with pytest.raises(MissingArchive, match="2020Q2") as pass2_exc:
        filings_of([("2020Q2", body)], {100001})
    assert isinstance(pass2_exc.value.__cause__, ValueError)


def test_malformed_row_in_passes_raises_quarter_labelled_missing_archive() -> None:
    bad_row_archive = make_raw_zip(
        INDEX_PREAMBLE + "100001|ACME CORP|10-K|2020-05-01|edgar/data/100001/bad.htm\n"
    )

    with pytest.raises(MissingArchive, match="2020Q4") as pass1_exc:
        find_filers([("2020Q4", bad_row_archive)], lambda _name: {})
    assert isinstance(pass1_exc.value.__cause__, ValueError)

    with pytest.raises(MissingArchive, match="2020Q4") as pass2_exc:
        filings_of([("2020Q4", bad_row_archive)], {100001})
    assert isinstance(pass2_exc.value.__cause__, ValueError)


def test_matcher_value_error_is_not_masked_as_missing_archive() -> None:
    valid = make_master_zip([("100001", "ACME CORP", "10-K", "2020-05-01")])

    def broken_matcher(_raw_name: str) -> Mapping[str, frozenset[str]]:
        raise ValueError("caller bug")

    with pytest.raises(ValueError, match="caller bug"):
        find_filers([("2020Q2", valid)], broken_matcher)


def test_cached_archives_reads_verified_entries_in_order_without_network(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    body_q2 = make_master_zip([("100001", "ACME CORP", "10-Q", "2020-05-01")])
    body_q3 = make_master_zip([("100001", "ACME CORP", "8-K", "2020-08-01")])
    seed_cache(cache_dir, sec_filings.master_index_url(2020, 2), body_q2)
    seed_cache(cache_dir, sec_filings.master_index_url(2020, 3), body_q3)

    opener = CountingRefusingOpener()
    client = SecClient(DEFAULT_UA, cache_dir, opener=opener)

    got = list(cached_archives(client, [(2020, 2), (2020, 3)]))

    assert got == [("2020Q2", body_q2), ("2020Q3", body_q3)]
    assert opener.calls == 0
    assert client.network_fetches == 0


def test_oversized_or_truncated_archive_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    oversized = b"x" * (MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(oversized)

    valid = make_master_zip([("100001", "ACME CORP", "10-K", "2020-02-15")])
    truncated = valid[: len(valid) // 2]
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(truncated)

    no_header = make_raw_zip("100001|ACME CORP|10-K|2020-02-15|edgar/data/100001/x.txt\n")
    with pytest.raises(ValueError, match="header"):
        parse_all_forms(no_header)

    # Verify the uncompressed index ceiling refuses an archive whose master.idx exceeds the limit.
    monkeypatch.setattr(sec_name_index, "MAX_INDEX_BYTES", 64)
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(valid)


def test_wrong_header_is_not_parsed_as_data() -> None:
    html_page = b"<!DOCTYPE html><html><body>Request Rate Threshold Exceeded</body></html>"
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(html_page)

    zipped_html = make_raw_zip(html_page)
    with pytest.raises(ValueError, match="header"):
        parse_all_forms(zipped_html)


def test_empty_company_name_is_kept_not_rejected() -> None:
    archive = make_raw_zip(
        INDEX_PREAMBLE
        + "1593547||485BPOS|2016-02-26|edgar/data/1593547/0001135428-16-001124.txt\n"
        + "1593547|   |485BPOS|2016-02-27|edgar/data/1593547/0001135428-16-001125.txt\n"
        + "1593547|Advisors' Inner Circle Fund III|485BPOS|2016-02-25|"
        + "edgar/data/1593547/0001135428-16-001123.txt\n"
    )

    filings = parse_all_forms(archive)
    assert filings == [
        IndexedFiling(
            cik=1593547,
            name="",
            form="485BPOS",
            filed=date(2016, 2, 26),
            accession="0001135428-16-001124",
        ),
        IndexedFiling(
            cik=1593547,
            name="",
            form="485BPOS",
            filed=date(2016, 2, 27),
            accession="0001135428-16-001125",
        ),
        IndexedFiling(
            cik=1593547,
            name="Advisors' Inner Circle Fund III",
            form="485BPOS",
            filed=date(2016, 2, 25),
            accession="0001135428-16-001123",
        ),
    ]

    # Pass 2 still returns the empty-name filings under their CIK, sorted by date.
    by_cik = filings_of([("2016Q1", archive)], {1593547})
    assert [f.accession for f in by_cik[1593547]] == [
        "0001135428-16-001123",
        "0001135428-16-001124",
        "0001135428-16-001125",
    ]
    assert [f.name for f in by_cik[1593547]] == ["Advisors' Inner Circle Fund III", "", ""]

    # Pass 1 skips empty names so callers' matchers are never invoked with "".
    seen_by_matcher: list[str] = []

    def strict_matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        assert raw_name != ""
        seen_by_matcher.append(raw_name)
        if raw_name == "Advisors' Inner Circle Fund III":
            return {"R1": frozenset({"v1"})}
        return {}

    matched = find_filers([("2016Q1", archive)], strict_matcher)
    assert matched == {1593547: {"R1": frozenset({"v1"})}}
    assert seen_by_matcher == ["Advisors' Inner Circle Fund III"]


@pytest.mark.parametrize(
    ("bad_line", "expected_line_no"),
    [
        ("100001|ACME CORP|10-K|2020-02-15", 6),
        ("|ACME CORP|10-K|2020-02-15|edgar/data/1/x.txt", 6),
        ("NOTACIK|ACME CORP|10-K|2020-02-15|edgar/data/1/x.txt", 6),
        ("0|ACME CORP|10-K|2020-02-15|edgar/data/0/x.txt", 6),
        ("100001|ACME CORP||2020-02-15|edgar/data/100001/x.txt", 6),
        ("100001|ACME CORP| |2020-02-15|edgar/data/100001/x.txt", 6),
        ("100001|ACME CORP|10-K|2020-02-30|edgar/data/100001/x.txt", 6),
        ("100001|ACME CORP|10-K|2020-02-15|", 6),
    ],
)
def test_malformed_data_line_raises_with_line_number(bad_line: str, expected_line_no: int) -> None:
    # Preamble has 4 lines, followed by 1 valid data line (line 5), then the bad line (line 6).
    valid_line = "100001|ACME CORP|10-K|2020-02-14|edgar/data/100001/0000100001-20-000001.txt\n"
    archive = make_raw_zip(INDEX_PREAMBLE + valid_line + bad_line + "\n")

    with pytest.raises(ValueError, match=rf"line {expected_line_no}\b"):
        parse_all_forms(archive)


def test_two_pass_streaming_reads_one_archive_at_a_time() -> None:
    a1 = make_master_zip([("100001", "DELTA CORP", "10-K", "2020-02-15")])
    a2 = make_master_zip([("100001", "DELTA CORP", "8-K", "2020-05-01")])
    yielded_pass1: list[str] = []

    def gen(log: list[str]) -> Iterator[tuple[str, bytes]]:
        for label, body in (("2020Q1", a1), ("2020Q2", a2)):
            log.append(label)
            yield label, body

    def matcher(raw_name: str) -> Mapping[str, frozenset[str]]:
        if raw_name == "DELTA CORP":
            return {"PERMNO-99999": frozenset({"v1"})}
        return {}

    found = find_filers(gen(yielded_pass1), matcher)
    assert found == {100001: {"PERMNO-99999": frozenset({"v1"})}}
    assert yielded_pass1 == ["2020Q1", "2020Q2"]

    yielded_pass2: list[str] = []
    filings = filings_of(gen(yielded_pass2), set(found))
    assert [f.form for f in filings[100001]] == ["10-K", "8-K"]
    assert yielded_pass2 == ["2020Q1", "2020Q2"]


def test_invalid_quarter_raises_value_error(tmp_path: Path) -> None:
    client = SecClient(DEFAULT_UA, tmp_path / "cache", opener=CountingRefusingOpener())

    with pytest.raises(ValueError, match="quarter"):
        list(cached_archives(client, [(2020, 5)]))


def test_archive_structure_and_encoding_guards() -> None:
    wrong_member = make_master_zip(
        [("100001", "ACME CORP", "10-K", "2020-02-15")], member_name="other.idx"
    )
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(wrong_member)

    missing_sep = make_raw_zip(
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "100001|ACME CORP|10-K|2020-02-15|edgar/data/100001/0000100001-20-000001.txt\n"
    )
    with pytest.raises(ValueError, match="separator"):
        parse_all_forms(missing_sep)

    bad_utf8 = make_raw_zip(
        INDEX_PREAMBLE.encode("utf-8") + b"100001|\xff|10-K|2020-02-15|edgar/data/100001/x.txt\n"
    )
    with pytest.raises(ValueError, match="archive"):
        parse_all_forms(bad_utf8)
