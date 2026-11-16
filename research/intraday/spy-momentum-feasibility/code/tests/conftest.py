"""Put this study's code on sys.path for its tests and provide synthetic issuer workbooks."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import pytest

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

HEADER = [
    "FUND NAME",
    "TICKER",
    "CUSIP",
    "EX-DATE",
    "RECORD DATE",
    "PAYABLE DATE",
    "DIVIDEND ($)",
    "SHORT TERM CAPITAL GAIN ($)",
    "LONG TERM CAPITAL GAIN ($)",
    "FREQUENCY",
]
COLUMNS = "ABCDEFGHIJ"


def issuer_row(
    ex: str,
    dividend: str,
    short: str = "",
    long: str = "",
    ticker: str = "SPY",
    cusip: str = "78462F103",
) -> list[str]:
    """One data row in the workbook's own text shape (dates as mm/dd/yyyy, amounts as text)."""
    return [
        "State Street SPDR S&P 500 ETF Trust",
        ticker,
        cusip,
        ex,
        ex,
        ex,
        dividend,
        short,
        long,
        "Quarterly",
    ]


def write_xlsx(path: Path, rows: list[list[str]], header: list[str] | None = None) -> bytes:
    """Write a minimal shared-string workbook shaped like the issuer's and return its bytes."""
    table = [HEADER if header is None else header, *rows]
    strings: list[str] = []
    sheet_rows = []
    for number, row in enumerate(table, start=1):
        cells = []
        for column, text in zip(COLUMNS, row, strict=False):
            if text not in strings:
                strings.append(text)
            cells.append(f'<c r="{column}{number}" t="s"><v>{strings.index(text)}</v></c>')
        sheet_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    shared = "".join(f"<si><t>{escape(text)}</t></si>" for text in strings)
    parts = {
        "xl/sharedStrings.xml": f'<?xml version="1.0"?><sst xmlns="{ns}">{shared}</sst>',
        "xl/worksheets/sheet1.xml": (
            f'<?xml version="1.0"?><worksheet xmlns="{ns}"><sheetData>'
            f"{''.join(sheet_rows)}</sheetData></worksheet>"
        ),
    }
    with zipfile.ZipFile(path, "w") as book:
        for name, body in parts.items():
            book.writestr(name, body)
    return path.read_bytes()


@pytest.fixture
def issuer_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> SimpleNamespace:
    """``install(rows, pin=True)`` points the study at a synthetic issuer workbook.

    With ``pin=False`` the frozen hash is left alone, so the file counts as tampered.
    """
    import intraday_momentum as core

    def install(rows: list[list[str]], *, pin: bool = True, **kwargs: object) -> Path:
        path = tmp_path_factory.mktemp("issuer") / "issuer.xlsx"
        raw = write_xlsx(path, rows, **kwargs)
        monkeypatch.setattr(core, "ISSUER_WORKBOOK_PATH", path)
        if pin:
            monkeypatch.setattr(core, "ISSUER_WORKBOOK_SHA256", hashlib.sha256(raw).hexdigest())
        return path

    return SimpleNamespace(install=install, row=issuer_row, write=write_xlsx)
