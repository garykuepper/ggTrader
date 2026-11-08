"""Build tiny ``.xlsx`` workbooks for tests with ``zipfile`` only (no spreadsheet dependency).

``build_xlsx`` writes one worksheet from rows of text/number cells; ``holdings_xlsx`` lays the rows
out like the SPDR SPY holdings download (title lines with an "As of" text, a blank row, the
``Name / Ticker / ...`` header, the holdings, cash and other lines, a disclaimer footer).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from xml.sax.saxutils import escape

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
Cell = str | int | float | None

HEADER = [
    "Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"
]  # fmt: skip


def _column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


def build_xlsx(
    rows: Sequence[Sequence[Cell]],
    *,
    inline: bool = False,
    rich_text: bool = False,
    drop_workbook_part: bool = False,
    doctype: bool = False,
    skip_row_numbers: bool = False,
) -> bytes:
    """A workbook whose first sheet holds ``rows`` (``None`` leaves a cell out).

    ``inline`` writes inline strings instead of shared strings; ``rich_text`` splits each shared
    string into two runs; ``drop_workbook_part`` removes the workbook and relationship parts (the
    reader must fall back to ``sheet1.xml``); ``doctype`` prepends an entity declaration to the
    sheet XML (which the reader must refuse); ``skip_row_numbers`` omits ``r=`` attributes.
    """
    strings: list[str] = []

    def string_index(text: str) -> int:
        if text not in strings:
            strings.append(text)
        return strings.index(text)

    sheet_rows = []
    for number, row in enumerate(rows, 1):
        cells = []
        for col, value in enumerate(row):
            if value is None:
                continue
            ref = f"{_column(col)}{number}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            elif inline:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}" t="s"><v>{string_index(value)}</v></c>')
        row_attr = "" if skip_row_numbers else f' r="{number}"'
        sheet_rows.append(f"<row{row_attr}>{''.join(cells)}</row>")
    prolog = '<?xml version="1.0" encoding="UTF-8"?>'
    if doctype:
        prolog += '<!DOCTYPE x [<!ENTITY a "aaaa">]>'
    sheet = (
        f'{prolog}<worksheet xmlns="{MAIN_NS}"><sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )

    def shared(text: str) -> str:
        if rich_text and len(text) > 1:
            half = len(text) // 2
            return (
                f"<si><r><t>{escape(text[:half])}</t></r>"
                f'<r><t xml:space="preserve">{escape(text[half:])}</t></r></si>'
            )
        return f"<si><t>{escape(text)}</t></si>"

    shared_xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{MAIN_NS}" count="{len(strings)}" '
        f'uniqueCount="{len(strings)}">{"".join(shared(t) for t in strings)}</sst>'
    )
    workbook = (
        f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{MAIN_NS}" '
        f'xmlns:r="{REL_NS}"><sheets><sheet name="holdings" sheetId="1" r:id="rId1"/>'
        '<sheet name="other" sheetId="2" r:id="rId2"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="worksheet" Target="worksheets/sheet2.xml"/>'
        '<Relationship Id="rId3" Type="sharedStrings" Target="sharedStrings.xml"/>'
        "</Relationships>"
    )
    other = (
        f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{MAIN_NS}"><sheetData>'
        '<row r="1"><c r="A1" t="inlineStr"><is><t>Ticker</t></is></c></row>'
        "</sheetData></worksheet>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        if not drop_workbook_part:
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
        archive.writestr("xl/worksheets/sheet2.xml", other)
        if not inline:
            archive.writestr("xl/sharedStrings.xml", shared_xml)
    return buffer.getvalue()


def holdings_rows(
    tickers: Sequence[str],
    as_of_text: str | None = "As of 05-Nov-2026",
    *,
    cash_lines: bool = True,
    footer: bool = True,
) -> list[list[Cell]]:
    """Rows laid out like the SPDR SPY download; ``as_of_text=None`` leaves the date out."""
    rows: list[list[Cell]] = [
        ["Fund Name:", "SPDR S&P 500 ETF Trust"],
        ["Ticker Symbol:", "SPY"],
        ["Holdings:", as_of_text] if as_of_text is not None else ["Holdings:"],
        [],
        list(HEADER),
    ]
    for i, ticker in enumerate(tickers):
        rows.append(
            [f"{ticker} Inc", ticker, f"ID{i:06d}", f"SD{i:05d}", 0.2, "Information Technology",
             1000 + i, "USD"]
        )  # fmt: skip
    if cash_lines:
        rows.append(["US DOLLAR", "USD", "-", "-", 0.01, "Cash", 5, "USD"])
        rows.append(["E-MINI S&P 500 FUTURE", "-", "-", "-", 0.0, "Other", 1, "USD"])
        rows.append(["CASH COLLATERAL", None, None, None, 0.0, "Cash and/or Derivatives", 1, "USD"])
    if footer:
        rows.append([])
        rows.append(["The information contained herein is for information purposes only."])
    return rows


def holdings_xlsx(
    tickers: Sequence[str], as_of_text: str | None = "As of 05-Nov-2026", **kwargs
) -> bytes:
    """A complete SPY-style workbook for ``tickers`` (raw tickers, as the fund writes them)."""
    build_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _BUILD_KEYS}
    return build_xlsx(holdings_rows(tickers, as_of_text, **kwargs), **build_kwargs)


_BUILD_KEYS = {"inline", "rich_text", "drop_workbook_part", "doctype", "skip_row_numbers"}
