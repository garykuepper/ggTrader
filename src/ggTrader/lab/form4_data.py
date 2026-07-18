"""SEC EDGAR Form 4 (insider transaction) data: free, structured, no API
key, via SEC's data.sec.gov submissions API + the raw ownership-document
XML each Form 4 filing carries. Full historical depth back to when a
company started filing electronically (typically the 1990s-2000s).

Free-data-only cut for candidate #1 (insider cluster-buying): this module
fetches and caches every Form 4 transaction, unfiltered -- filtering to
open-market purchases (transaction code "P"), excluding 10b5-1 plans, and
clustering 3+ distinct insiders within a compressed window all happen in
the strategy layer, not here, so the cache stays reusable.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable, Iterable, List

import pandas as pd
from defusedxml import ElementTree as ET
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{filename}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"

#: SEC requires a descriptive User-Agent identifying the requester on every
#: request or it 403s -- see https://www.sec.gov/os/webmaster-faq#developers
USER_AGENT = "ggTrader research contact@example.com"

_COLUMNS = [
    "symbol",
    "issuer_cik",
    "insider_cik",
    "insider_name",
    "accession_number",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "acquired_disposed_code",
    "is_10b5_1_plan",
    "filing_date",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS form4_transactions (
    symbol text NOT NULL,
    issuer_cik integer NOT NULL,
    insider_cik integer NOT NULL,
    insider_name text,
    accession_number text NOT NULL,
    transaction_date date NOT NULL,
    transaction_code text,
    shares double precision,
    price_per_share double precision,
    acquired_disposed_code text,
    is_10b5_1_plan boolean,
    filing_date date,
    PRIMARY KEY (accession_number, insider_cik, transaction_date, transaction_code, shares)
)
"""

HttpFetch = Callable[[str], str]


def _default_http_fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def load_ticker_cik_map(http_fetch: HttpFetch = _default_http_fetch) -> dict[str, int]:
    """symbol -> CIK, from SEC's bulk company_tickers.json (one free file,
    no pagination, covers every SEC-registered issuer)."""
    raw = json.loads(http_fetch(COMPANY_TICKERS_URL))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def _xml_filename(primary_document: str) -> str:
    """The submissions API's primaryDocument is the XSLT-rendered viewer
    path (e.g. 'xslF345X06/form4.xml'); the raw XML this module parses
    sits at the accession root under the same trailing filename."""
    return primary_document.rsplit("/", 1)[-1]


def _filings_from_submissions_json(raw: dict) -> List[dict]:
    forms = raw["form"]
    dates = raw["filingDate"]
    accessions = raw["accessionNumber"]
    docs = raw["primaryDocument"]
    return [
        {
            "form": forms[i],
            "filingDate": dates[i],
            "accessionNumber": accessions[i],
            "xml_filename": _xml_filename(docs[i]),
        }
        for i in range(len(forms))
        if forms[i] == "4"
    ]


def list_form4_filings(cik: int, http_fetch: HttpFetch = _default_http_fetch) -> List[dict]:
    """Every Form 4 filing for a CIK (as issuer), across pagination --
    the submissions JSON's 'recent' block holds only the newest ~1000
    filings; older history lives in separate paginated files listed under
    filings.files.
    """
    raw = json.loads(http_fetch(SUBMISSIONS_URL.format(cik=cik)))
    filings = _filings_from_submissions_json(raw["filings"]["recent"])
    for page in raw["filings"].get("files", []):
        older = json.loads(http_fetch(SUBMISSIONS_FILE_URL.format(filename=page["name"])))
        filings.extend(_filings_from_submissions_json(older))
    return filings


def _find(elem: Element | None, path: str) -> str | None:
    if elem is None:
        return None
    found = elem.find(path)
    return found.text if found is not None else None


def parse_form4_transactions(xml_text: str) -> List[dict]:
    """Every non-derivative transaction in one Form 4 filing, unfiltered by
    transaction code -- open-market-purchase filtering happens downstream.
    """
    root = ET.fromstring(xml_text)
    issuer_cik_raw = _find(root, "issuer/issuerCik")
    if issuer_cik_raw is None:
        return []
    issuer_cik = int(issuer_cik_raw)
    symbol = _find(root, "issuer/issuerTradingSymbol") or ""
    insider_cik_raw = _find(root, "reportingOwner/reportingOwnerId/rptOwnerCik")
    insider_cik = int(insider_cik_raw) if insider_cik_raw else None
    insider_name = _find(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    is_10b5_1 = (_find(root, "aff10b5One") or "false").strip().lower() == "true"

    rows: List[dict] = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        date_str = _find(tx, "transactionDate/value")
        code = _find(tx, "transactionCoding/transactionCode")
        shares_str = _find(tx, "transactionAmounts/transactionShares/value")
        price_str = _find(tx, "transactionAmounts/transactionPricePerShare/value")
        acq_disp = _find(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        rows.append(
            {
                "symbol": symbol,
                "issuer_cik": issuer_cik,
                "insider_cik": insider_cik,
                "insider_name": insider_name,
                "transaction_date": pd.Timestamp(date_str) if date_str else None,
                "transaction_code": code,
                "shares": float(shares_str) if shares_str else float("nan"),
                "price_per_share": float(price_str) if price_str else float("nan"),
                "acquired_disposed_code": acq_disp,
                "is_10b5_1_plan": is_10b5_1,
            }
        )
    return rows


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def cache_filing(
    cik: int,
    accession_number: str,
    xml_filename: str,
    filing_date: str,
    http_fetch: HttpFetch = _default_http_fetch,
) -> int:
    """Fetch one Form 4 filing's raw XML, parse, and upsert its
    transactions into the DB cache. Returns rows written."""
    ensure_schema()
    accession_nodash = accession_number.replace("-", "")
    url = ARCHIVE_URL.format(cik=cik, accession_nodash=accession_nodash, filename=xml_filename)
    xml_text = http_fetch(url)
    rows = parse_form4_transactions(xml_text)
    if not rows:
        return 0
    with get_engine().begin() as conn:
        for r in rows:
            if r["transaction_date"] is None or r["insider_cik"] is None:
                continue
            conn.execute(
                text(
                    "INSERT INTO form4_transactions (symbol, issuer_cik, insider_cik, "
                    "insider_name, accession_number, transaction_date, transaction_code, "
                    "shares, price_per_share, acquired_disposed_code, is_10b5_1_plan, "
                    "filing_date) "
                    "VALUES (:symbol, :issuer_cik, :insider_cik, :insider_name, "
                    ":accession_number, :transaction_date, :transaction_code, :shares, "
                    ":price_per_share, :acquired_disposed_code, :is_10b5_1_plan, "
                    ":filing_date) "
                    "ON CONFLICT (accession_number, insider_cik, transaction_date, "
                    "transaction_code, shares) DO UPDATE SET "
                    "price_per_share = EXCLUDED.price_per_share, "
                    "acquired_disposed_code = EXCLUDED.acquired_disposed_code, "
                    "is_10b5_1_plan = EXCLUDED.is_10b5_1_plan"
                ),
                {**r, "accession_number": accession_number, "filing_date": filing_date},
            )
    return len(rows)


def load_form4_transactions(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Load cached Form 4 transactions for symbols within [start, end]
    (by transaction_date)."""
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, issuer_cik, insider_cik, insider_name, accession_number, "
                "transaction_date, transaction_code, shares, price_per_share, "
                "acquired_disposed_code, is_10b5_1_plan, filing_date "
                "FROM form4_transactions "
                "WHERE symbol = ANY(:syms) AND transaction_date BETWEEN :start AND :end "
                "ORDER BY transaction_date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
