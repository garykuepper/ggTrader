"""A minimal, standard-library reader for the first worksheet of an ``.xlsx`` workbook.

Only what a plain table download needs: shared strings, inline strings, text and numeric cells,
row and column placement. No styles, dates-as-numbers, formulas or merged cells; numbers come
back as the text stored in the file. It uses ``zipfile`` and ``xml.etree`` only, so the project
needs no spreadsheet dependency. Input is untrusted: a non-zip, a truncated archive, a missing
part, a DOCTYPE or entity declaration (the expansion-bomb vector of ``xml.etree``) or an
oversized part raises :class:`XlsxError` instead of being parsed.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from xml.etree import ElementTree

#: Longest uncompressed size accepted for any one part of the archive.
MAX_PART_BYTES = 50 * 1024 * 1024
#: Rows beyond this count are refused (a real holdings sheet has about 510).
MAX_ROWS = 100_000

_WORKBOOK = "xl/workbook.xml"
_WORKBOOK_RELS = "xl/_rels/workbook.xml.rels"
_SHARED_STRINGS = "xl/sharedStrings.xml"
_FALLBACK_SHEET = "xl/worksheets/sheet1.xml"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_CELL_REF = re.compile(r"^([A-Za-z]{1,3})(\d+)$")


class XlsxError(ValueError):
    """The bytes are not a readable ``.xlsx`` workbook (or not the structure this reader knows)."""


def _local(tag: str) -> str:
    """The tag name without its ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _read_part(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise XlsxError(f"the workbook has no part {name!r}") from None
    if info.file_size > MAX_PART_BYTES:
        raise XlsxError(f"part {name!r} is {info.file_size} bytes, over the {MAX_PART_BYTES} cap")
    try:
        with archive.open(info) as handle:
            data = handle.read(MAX_PART_BYTES + 1)
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError) as exc:
        raise XlsxError(f"part {name!r} cannot be read ({type(exc).__name__})") from None
    if len(data) > MAX_PART_BYTES:
        raise XlsxError(f"part {name!r} is over the {MAX_PART_BYTES} byte cap")
    return data


def _parse_xml(data: bytes, name: str) -> ElementTree.Element:
    head = data[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in data.lower():
        raise XlsxError(f"part {name!r} declares a DOCTYPE or entity, which is refused")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise XlsxError(f"part {name!r} is not well-formed XML ({exc})") from None


def _text_of(node: ElementTree.Element) -> str:
    """All ``<t>`` text under an ``<si>`` or ``<is>`` element (plain or rich text), joined."""
    return "".join(t.text or "" for t in node.iter() if _local(t.tag) == "t")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if _SHARED_STRINGS not in archive.namelist():
        return []
    root = _parse_xml(_read_part(archive, _SHARED_STRINGS), _SHARED_STRINGS)
    return [_text_of(si) for si in root if _local(si.tag) == "si"]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    """The part holding the first worksheet, from the workbook's relationships."""
    names = set(archive.namelist())
    if _WORKBOOK in names and _WORKBOOK_RELS in names:
        book = _parse_xml(_read_part(archive, _WORKBOOK), _WORKBOOK)
        rels = _parse_xml(_read_part(archive, _WORKBOOK_RELS), _WORKBOOK_RELS)
        targets = {r.get("Id"): r.get("Target") for r in rels if _local(r.tag) == "Relationship"}
        for sheet in book.iter():
            if _local(sheet.tag) == "sheet":
                target = targets.get(sheet.get(_REL_ID))
                if target:
                    # Targets are relative to xl/, or absolute from the archive root.
                    path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
                    path = posixpath.normpath(path)
                    if path in names:
                        return path
                break
    if _FALLBACK_SHEET in names:
        return _FALLBACK_SHEET
    raise XlsxError("the workbook has no readable worksheet")


def _column_index(letters: str) -> int:
    index = 0
    for ch in letters.upper():
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def read_first_sheet(data: bytes) -> list[list[str]]:
    """The first worksheet as rows of cell text (``""`` for an empty cell), top to bottom.

    Raises :class:`XlsxError` for anything that is not a well-formed workbook: HTML or text
    saved under an ``.xlsx`` name, a truncated archive, a missing worksheet part, malformed XML.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise XlsxError("the download is not a zip archive (not an .xlsx workbook)") from None
    with archive:
        strings = _shared_strings(archive)
        sheet_path = _first_sheet_path(archive)
        root = _parse_xml(_read_part(archive, sheet_path), sheet_path)
    rows: list[list[str]] = []
    for row in root.iter():
        if _local(row.tag) != "row":
            continue
        if len(rows) >= MAX_ROWS:
            raise XlsxError(f"the worksheet has more than {MAX_ROWS} rows")
        cells: dict[int, str] = {}
        next_col = 0
        for cell in row:
            if _local(cell.tag) != "c":
                continue
            match = _CELL_REF.match(cell.get("r") or "")
            col = _column_index(match.group(1)) if match else next_col
            next_col = col + 1
            cells[col] = _cell_text(cell, strings)
        width = max(cells, default=-1) + 1
        row_number = row.get("r")
        if row_number and row_number.isdigit():
            # A skipped row number means blank rows; keep positions stable for callers.
            while len(rows) < int(row_number) - 1:
                rows.append([])
        rows.append([cells.get(i, "") for i in range(width)])
    return rows


def _cell_text(cell: ElementTree.Element, strings: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        inline = next((c for c in cell if _local(c.tag) == "is"), None)
        return _text_of(inline) if inline is not None else ""
    value = next((c for c in cell if _local(c.tag) == "v"), None)
    text = (value.text or "") if value is not None else ""
    if kind == "s":
        try:
            return strings[int(text)]
        except (ValueError, IndexError):
            raise XlsxError(f"cell {cell.get('r')!r} points at a missing shared string") from None
    return text
