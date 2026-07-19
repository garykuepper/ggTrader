"""House Periodic Transaction Reports (STOCK Act insider disclosures):
free, structured discovery + text-extractable PDFs, via the House Clerk's
public annual bulk index. Scoped to the House only (v1) -- the Senate's
efdsearch.senate.gov requires a stateful CSRF-protected session rather
than a simple bulk index, a meaningfully different engineering lift not
undertaken here.

PTR PDFs are genuinely digitally-generated (not scanned images) even for
older filings, so no OCR is needed -- but the extracted text is messy
(irregular whitespace, owner codes like "SP"/"JT" glued onto asset names
with no separator, dates sometimes glued together with no separator).
parse_ptr_text anchors on the most reliably-structured part of each
transaction line (ticker in parens, transaction code, two dates, a dollar
amount range) rather than trying to parse the noisy asset-name/owner
prefix.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import Callable, Iterable, List

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

YEAR_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
USER_AGENT = "ggTrader research contact@example.com"

_COLUMNS = [
    "symbol",
    "last_name",
    "first_name",
    "state_dst",
    "doc_id",
    "filing_date",
    "transaction_type",
    "transaction_date",
    "notification_date",
    "amount_low",
    "amount_high",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS house_ptr_transactions (
    symbol text NOT NULL,
    last_name text,
    first_name text,
    state_dst text,
    doc_id text NOT NULL,
    filing_date date,
    transaction_type text,
    transaction_date date NOT NULL,
    notification_date date,
    amount_low double precision,
    amount_high double precision,
    PRIMARY KEY (doc_id, symbol, transaction_date, transaction_type, amount_low)
)
"""

#: Anchors on the reliably-structured part of a transaction line: ticker in
#: parens, an optional bracketed asset-type code ("[ST]"), the single-letter
#: transaction code (P/S/E, optionally followed by a "(partial)"-style
#: qualifier), two MM/DD/YYYY dates (sometimes glued together with no
#: separator), and a "$X,XXX - $Y,YYY" amount range.
_TX_PATTERN = re.compile(
    r"\(([A-Za-z]{1,6})\)\s*(?:\[[A-Z]+\]\s*)?([PSE])(?:\s*\([^)]*\))?\s*"
    r"(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}/\d{1,2}/\d{4})\s*\$?([\d,]+)\s*-\s*\$?([\d,]+)"
)

HttpFetch = Callable[[str], bytes]


def _default_http_fetch(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_year_index(year: int, http_fetch: HttpFetch = _default_http_fetch) -> List[dict]:
    """Every PTR (Periodic Transaction Report, FilingType "P") filed in a
    given year, from the House Clerk's public bulk annual index."""
    raw = http_fetch(YEAR_INDEX_URL.format(year=year))
    zf = zipfile.ZipFile(io.BytesIO(raw))
    with zf.open(f"{year}FD.txt") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
        return [
            {
                "last": row["Last"],
                "first": row["First"],
                "state_dst": row["StateDst"],
                "filing_date": row["FilingDate"],
                "doc_id": row["DocID"],
                "year": year,
            }
            for row in reader
            if row["FilingType"] == "P"
        ]


def parse_ptr_text(text_content: str) -> List[dict]:
    """Every transaction line in one PTR's extracted text."""
    rows: List[dict] = []
    for m in _TX_PATTERN.finditer(text_content):
        symbol, tx_type, tx_date, notif_date, low, high = m.groups()
        rows.append(
            {
                "symbol": symbol.upper(),
                "transaction_type": tx_type,
                "transaction_date": pd.Timestamp(tx_date),
                "notification_date": pd.Timestamp(notif_date),
                "amount_low": float(low.replace(",", "")),
                "amount_high": float(high.replace(",", "")),
            }
        )
    return rows


def parse_ptr_pdf(pdf_bytes: bytes) -> List[dict]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_content = " ".join(p.extract_text() for p in reader.pages).replace("\x00", "")
    return parse_ptr_text(text_content)


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def cache_filing(
    year: int,
    doc_id: str,
    last: str,
    first: str,
    state_dst: str,
    filing_date: str,
    http_fetch: HttpFetch = _default_http_fetch,
) -> int:
    """Fetch one PTR's PDF, parse its transactions, and upsert into the DB
    cache. Returns rows written."""
    ensure_schema()
    pdf_bytes = http_fetch(PTR_PDF_URL.format(year=year, doc_id=doc_id))
    rows = parse_ptr_pdf(pdf_bytes)
    if not rows:
        return 0
    with get_engine().begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "INSERT INTO house_ptr_transactions (symbol, last_name, first_name, "
                    "state_dst, doc_id, filing_date, transaction_type, transaction_date, "
                    "notification_date, amount_low, amount_high) "
                    "VALUES (:symbol, :last_name, :first_name, :state_dst, :doc_id, "
                    ":filing_date, :transaction_type, :transaction_date, :notification_date, "
                    ":amount_low, :amount_high) "
                    "ON CONFLICT (doc_id, symbol, transaction_date, transaction_type, "
                    "amount_low) DO UPDATE SET "
                    "notification_date = EXCLUDED.notification_date, "
                    "amount_high = EXCLUDED.amount_high"
                ),
                {
                    **r,
                    "last_name": last,
                    "first_name": first,
                    "state_dst": state_dst,
                    "doc_id": doc_id,
                    "filing_date": filing_date,
                },
            )
    return len(rows)


def load_house_ptr_transactions(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Load cached House PTR transactions for symbols within [start, end]
    (by transaction_date)."""
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, last_name, first_name, state_dst, doc_id, filing_date, "
                "transaction_type, transaction_date, notification_date, amount_low, "
                "amount_high FROM house_ptr_transactions "
                "WHERE symbol = ANY(:syms) AND transaction_date BETWEEN :start AND :end "
                "ORDER BY transaction_date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
