"""Tests for the SEC EDGAR Form 4 (insider transaction) data loader."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ggTrader.lab.form4_data import (
    list_form4_filings,
    load_ticker_cik_map,
    parse_form4_transactions,
)


class TestLoadTickerCikMap:
    def test_parses_ticker_to_cik(self):
        raw = json.dumps(
            {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
            }
        )
        m = load_ticker_cik_map(http_fetch=lambda url: raw)
        assert m == {"AAPL": 320193, "MSFT": 789019}


class TestListForm4Filings:
    def _submissions(self, forms_dates_docs, files=None):
        return json.dumps(
            {
                "filings": {
                    "recent": {
                        "form": [f for f, _, _ in forms_dates_docs],
                        "filingDate": [d for _, d, _ in forms_dates_docs],
                        "accessionNumber": [f"000-{i}" for i in range(len(forms_dates_docs))],
                        "primaryDocument": [doc for _, _, doc in forms_dates_docs],
                    },
                    "files": files or [],
                }
            }
        )

    def test_filters_to_form_4_only(self):
        raw = self._submissions(
            [
                ("4", "2020-01-01", "xslF345X06/form4.xml"),
                ("8-K", "2020-01-02", "8k.htm"),
                ("4", "2020-01-03", "xslF345X06/form4.xml"),
            ]
        )
        filings = list_form4_filings(320193, http_fetch=lambda url: raw)
        assert len(filings) == 2
        assert all(f["form"] == "4" for f in filings)

    def test_derives_raw_xml_filename_from_primary_document(self):
        raw = self._submissions([("4", "2020-01-01", "xslF345X06/form4.xml")])
        filings = list_form4_filings(320193, http_fetch=lambda url: raw)
        assert filings[0]["xml_filename"] == "form4.xml"

    def test_follows_pagination_to_older_filings(self):
        recent = self._submissions(
            [("4", "2020-06-01", "xslF345X06/form4.xml")],
            files=[{"name": "CIK0000320193-submissions-001.json"}],
        )
        older = json.dumps(
            {
                "form": ["4"],
                "filingDate": ["2016-01-01"],
                "accessionNumber": ["000-old"],
                "primaryDocument": ["xslF345X03/edgar.xml"],
            }
        )

        def fake_http(url):
            if "submissions-001" in url:
                return older
            return recent

        filings = list_form4_filings(320193, http_fetch=fake_http)
        dates = sorted(f["filingDate"] for f in filings)
        assert dates == ["2016-01-01", "2020-06-01"]


def _form4_xml(
    transactions: list[dict],
    aff10b5one: str = "false",
    owner_cik: str = "0001780525",
    owner_name: str = "Newstead Jennifer",
) -> str:
    tx_xml = ""
    for t in transactions:
        price_xml = (
            f"<value>{t['price']}</value>"
            if t.get("price") is not None
            else '<footnoteId id="F1"/>'
        )
        tx_xml += f"""
        <nonDerivativeTransaction>
            <transactionDate><value>{t["date"]}</value></transactionDate>
            <transactionCoding><transactionCode>{t["code"]}</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>{t["shares"]}</value></transactionShares>
                <transactionPricePerShare>{price_xml}</transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>{t["acq_disp"]}</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>"""
    return f"""<?xml version="1.0"?>
<ownershipDocument>
    <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerCik>{owner_cik}</rptOwnerCik><rptOwnerName>{owner_name}</rptOwnerName></reportingOwnerId>
    </reportingOwner>
    <aff10b5One>{aff10b5one}</aff10b5One>
    <nonDerivativeTable>{tx_xml}</nonDerivativeTable>
</ownershipDocument>"""


class TestParseForm4Transactions:
    def test_extracts_open_market_purchase(self):
        xml = _form4_xml(
            [
                {
                    "date": "2020-01-15",
                    "code": "P",
                    "shares": "100",
                    "price": "50.25",
                    "acq_disp": "A",
                }
            ]
        )
        rows = parse_form4_transactions(xml)
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "AAPL"
        assert r["issuer_cik"] == 320193
        assert r["insider_cik"] == 1780525
        assert r["insider_name"] == "Newstead Jennifer"
        assert r["transaction_date"] == pd.Timestamp("2020-01-15")
        assert r["transaction_code"] == "P"
        assert r["shares"] == pytest.approx(100.0)
        assert r["price_per_share"] == pytest.approx(50.25)
        assert r["acquired_disposed_code"] == "A"
        assert r["is_10b5_1_plan"] is False

    def test_flags_10b5_1_plan(self):
        xml = _form4_xml(
            [
                {
                    "date": "2020-01-15",
                    "code": "P",
                    "shares": "100",
                    "price": "50.25",
                    "acq_disp": "A",
                }
            ],
            aff10b5one="true",
        )
        rows = parse_form4_transactions(xml)
        assert rows[0]["is_10b5_1_plan"] is True

    def test_includes_multiple_transaction_codes_unfiltered(self):
        """Parsing extracts everything -- filtering to code 'P' happens in
        the strategy/query layer, not here, so the cache stays reusable for
        other purposes (e.g. a future insider-selling study)."""
        xml = _form4_xml(
            [
                {
                    "date": "2020-01-15",
                    "code": "P",
                    "shares": "100",
                    "price": "50.0",
                    "acq_disp": "A",
                },
                {
                    "date": "2020-01-15",
                    "code": "M",
                    "shares": "200",
                    "price": None,
                    "acq_disp": "A",
                },
            ]
        )
        rows = parse_form4_transactions(xml)
        assert {r["transaction_code"] for r in rows} == {"P", "M"}

    def test_missing_price_footnote_becomes_nan(self):
        xml = _form4_xml(
            [{"date": "2020-01-15", "code": "M", "shares": "200", "price": None, "acq_disp": "A"}]
        )
        rows = parse_form4_transactions(xml)
        assert pd.isna(rows[0]["price_per_share"])

    def test_no_nondrivative_table_returns_empty(self):
        xml = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
    <reportingOwner><reportingOwnerId><rptOwnerCik>1</rptOwnerCik><rptOwnerName>X</rptOwnerName></reportingOwnerId></reportingOwner>
    <aff10b5One>false</aff10b5One>
</ownershipDocument>"""
        rows = parse_form4_transactions(xml)
        assert rows == []


@pytest.mark.integration
def test_cache_filing_and_load_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.form4_data import cache_filing, ensure_schema, load_form4_transactions
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_F4"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM form4_transactions WHERE symbol = :s"), {"s": marker})

    xml = _form4_xml(
        [{"date": "2020-01-15", "code": "P", "shares": "100", "price": "50.25", "acq_disp": "A"}]
    ).replace("AAPL", marker)
    n = cache_filing(
        320193, "0001780525-20-000001", "form4.xml", "2020-01-17", http_fetch=lambda url: xml
    )
    assert n == 1

    df = load_form4_transactions([marker], "2020-01-01", "2020-12-31")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == marker
    assert df.iloc[0]["transaction_code"] == "P"
    assert df.iloc[0]["shares"] == pytest.approx(100.0)

    # Re-caching the same filing upserts, not duplicates.
    cache_filing(
        320193, "0001780525-20-000001", "form4.xml", "2020-01-17", http_fetch=lambda url: xml
    )
    df2 = load_form4_transactions([marker], "2020-01-01", "2020-12-31")
    assert len(df2) == 1

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM form4_transactions WHERE symbol = :s"), {"s": marker})
