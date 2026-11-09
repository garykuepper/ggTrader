"""Fast tests of ``ggTrader.utils.xlsx_min``, the standard-library ``.xlsx`` reader.

Workbooks are built in memory with ``zipfile`` (``tests/data/xlsx_builder.py``).
"""

from __future__ import annotations

import io
import zipfile

import pytest

from ggTrader.utils.xlsx_min import XlsxError, read_first_sheet
from tests.data.xlsx_builder import build_xlsx

ROWS = [["Name", "Ticker"], ["Apple", "AAPL", 12.5], [None, "MSFT"], [], ["x", None, "z"]]


@pytest.mark.parametrize("rich", [False, True])
def test_reader_returns_cell_text_with_blank_cells_and_blank_rows_in_place(rich):
    rows = read_first_sheet(build_xlsx(ROWS, rich_text=rich))
    assert rows == [["Name", "Ticker"], ["Apple", "AAPL", "12.5"], ["", "MSFT"], [], ["x", "", "z"]]


def test_reader_reads_inline_strings_and_rows_without_numbers():
    expected = [["Name", "Ticker"], ["Apple", "AAPL", "12.5"], ["", "MSFT"], [], ["x", "", "z"]]
    assert read_first_sheet(build_xlsx(ROWS, inline=True)) == expected
    assert read_first_sheet(build_xlsx([["a", "b"], ["c"]], skip_row_numbers=True)) == [
        ["a", "b"],
        ["c"],
    ]


def test_reader_falls_back_to_sheet1_without_a_workbook_part():
    assert read_first_sheet(build_xlsx([["Ticker"]], drop_workbook_part=True)) == [["Ticker"]]


def test_reader_takes_the_first_sheet_of_the_workbook():
    rows = read_first_sheet(build_xlsx([["Ticker", "AAA"]]))
    assert rows == [["Ticker", "AAA"]]  # not the second sheet, which holds only "Ticker"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"<html><body>Access Denied</body></html>", "not a zip"),
        (b"", "not a zip"),
        (build_xlsx([["Ticker"]])[:-60], "not a zip"),  # truncated: no central directory
        (build_xlsx([["Ticker"]], doctype=True), "DOCTYPE or entity"),
    ],
)
def test_reader_refuses_what_is_not_a_workbook(data, message):
    with pytest.raises(XlsxError, match=message):
        read_first_sheet(data)


def test_reader_refuses_a_zip_with_no_worksheet():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(XlsxError, match="no readable worksheet"):
        read_first_sheet(buffer.getvalue())


def test_reader_refuses_malformed_xml_and_a_bad_shared_string_pointer():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData><row>")
    with pytest.raises(XlsxError, match="not well-formed"):
        read_first_sheet(buffer.getvalue())
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>7</v></c></row></sheetData></worksheet>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    with pytest.raises(XlsxError, match="missing shared string"):
        read_first_sheet(buffer.getvalue())


def _sheet_zip(sheet_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet_bytes)
    return buffer.getvalue()


_MAIN = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def test_reader_refuses_one_huge_row_number_before_padding_to_it():
    sheet = f'<worksheet {_MAIN}><sheetData><row r="300000"/></sheetData></worksheet>'
    with pytest.raises(XlsxError, match="row cap"):
        read_first_sheet(_sheet_zip(sheet.encode()))


def test_reader_accepts_a_row_number_at_the_cap_and_refuses_one_past_it():
    def sheet_with(number: int) -> bytes:
        text = f'<worksheet {_MAIN}><sheetData><row r="{number}"/></sheetData></worksheet>'
        return _sheet_zip(text.encode())

    assert len(read_first_sheet(sheet_with(100_000))) == 100_000
    with pytest.raises(XlsxError, match="row cap"):
        read_first_sheet(sheet_with(100_001))


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be", "utf-32"])
def test_reader_refuses_a_utf16_part_that_declares_an_entity(codec):
    text = (
        '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
        f"<worksheet {_MAIN}><sheetData/></worksheet>"
    )
    with pytest.raises(XlsxError, match="UTF-8"):
        read_first_sheet(_sheet_zip(text.encode(codec)))


def test_reader_refuses_a_non_utf8_declaration_and_still_reads_a_bom_marked_utf8_part():
    declared = f'<?xml version="1.0" encoding="ISO-8859-1"?><worksheet {_MAIN}/>'
    with pytest.raises(XlsxError, match="non-UTF-8"):
        read_first_sheet(_sheet_zip(declared.encode()))
    ok = f'<?xml version="1.0" encoding="UTF-8"?><worksheet {_MAIN}><sheetData/></worksheet>'
    assert read_first_sheet(_sheet_zip(b"\xef\xbb\xbf" + ok.encode())) == []
