"""Tests for the House Periodic Transaction Report (STOCK Act) data loader."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from ggTrader.lab.house_ptr_data import fetch_year_index


def _year_index_zip(rows: list[dict]) -> bytes:
    header = "Prefix\tLast\tFirst\tSuffix\tFilingType\tStateDst\tYear\tFilingDate\tDocID\n"
    body = "".join(
        f"{r.get('prefix', 'Hon.')}\t{r['last']}\t{r['first']}\t\t{r['filing_type']}\t"
        f"{r['state_dst']}\t{r['year']}\t{r['filing_date']}\t{r['doc_id']}\n"
        for r in rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("2016FD.txt", header + body)
    return buf.getvalue()


class TestFetchYearIndex:
    def test_filters_to_ptr_filings_only(self):
        raw = _year_index_zip(
            [
                {
                    "last": "Smith",
                    "first": "Jane",
                    "filing_type": "P",
                    "state_dst": "CA01",
                    "year": 2016,
                    "filing_date": "1/15/2016",
                    "doc_id": "20004419",
                },
                {
                    "last": "Smith",
                    "first": "Jane",
                    "filing_type": "A",
                    "state_dst": "CA01",
                    "year": 2016,
                    "filing_date": "5/15/2016",
                    "doc_id": "20005000",
                },
            ]
        )
        filings = fetch_year_index(2016, http_fetch=lambda url: raw)
        assert len(filings) == 1
        assert filings[0]["doc_id"] == "20004419"
        assert filings[0]["last"] == "Smith"
        assert filings[0]["filing_date"] == "1/15/2016"

    def test_empty_index_returns_empty_list(self):
        raw = _year_index_zip([])
        filings = fetch_year_index(2016, http_fetch=lambda url: raw)
        assert filings == []


class TestParsePtrPdf:
    """parse_ptr_pdf just wraps PDF-bytes -> text extraction around
    parse_ptr_text, which holds all the actual parsing logic and is what
    these tests exercise directly -- pypdf can't easily synthesize
    arbitrary extractable text for a fake PDF fixture."""

    def test_extracts_a_sale(self):
        from ggTrader.lab.house_ptr_data import parse_ptr_text

        text = (
            "SP Apple Inc. - Common Stock (AAPL)\n[ST]\n"
            "S (partial) 10/22/202510/22/2025$100,001 -\n$250,000"
        )
        rows = parse_ptr_text(text)
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "AAPL"
        assert r["transaction_type"] == "S"
        assert r["transaction_date"] == pd.Timestamp("2025-10-22")
        assert r["notification_date"] == pd.Timestamp("2025-10-22")
        assert r["amount_low"] == pytest.approx(100001.0)
        assert r["amount_high"] == pytest.approx(250000.0)

    def test_extracts_a_purchase(self):
        from ggTrader.lab.house_ptr_data import parse_ptr_text

        text = "Some Bank (MSB)P 11/22/2016 11/22/2016 $1,001 - $15,000"
        rows = parse_ptr_text(text)
        assert rows[0]["transaction_type"] == "P"
        assert rows[0]["symbol"] == "MSB"

    def test_uppercases_font_rendered_lowercase_ticker(self):
        """Small-caps font rendering sometimes yields mixed-case tickers
        (e.g. 'aCMP' for 'ACMP') -- real tickers are always uppercase."""
        from ggTrader.lab.house_ptr_data import parse_ptr_text

        text = "Access Midstream Partners (aCMP)E 02/2/2015 02/2/2015 $1,001 - $15,000"
        rows = parse_ptr_text(text)
        assert rows[0]["symbol"] == "ACMP"

    def test_multiple_transactions_in_one_filing(self):
        from ggTrader.lab.house_ptr_data import parse_ptr_text

        text = (
            "3M Company (MMM)S 03/29/2016 03/29/2016 $15,001 - $50,000 "
            "Apache Corporation (APA)S 03/29/2016 03/29/2016 $15,001 - $50,000"
        )
        rows = parse_ptr_text(text)
        assert {r["symbol"] for r in rows} == {"MMM", "APA"}

    def test_no_transactions_returns_empty_list(self):
        from ggTrader.lab.house_ptr_data import parse_ptr_text

        assert parse_ptr_text("no transactions here") == []


@pytest.mark.integration
def test_cache_filing_and_load_roundtrip(monkeypatch):
    from sqlalchemy import text

    import ggTrader.lab.house_ptr_data as mod
    from ggTrader.lab.persist import get_engine

    mod.ensure_schema()
    marker = "ZZTEST_HOUSE"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM house_ptr_transactions WHERE symbol = :s"), {"s": marker})

    canned = [
        {
            "symbol": marker,
            "transaction_type": "P",
            "transaction_date": pd.Timestamp("2020-01-15"),
            "notification_date": pd.Timestamp("2020-01-20"),
            "amount_low": 1001.0,
            "amount_high": 15000.0,
        }
    ]
    monkeypatch.setattr(mod, "parse_ptr_pdf", lambda pdf_bytes: canned)

    n = mod.cache_filing(
        2020, "20099999", "Smith", "Jane", "CA01", "2020-01-20", http_fetch=lambda url: b""
    )
    assert n == 1

    df = mod.load_house_ptr_transactions([marker], "2020-01-01", "2020-12-31")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == marker
    assert df.iloc[0]["transaction_type"] == "P"

    # Re-caching the same filing upserts, not duplicates.
    mod.cache_filing(
        2020, "20099999", "Smith", "Jane", "CA01", "2020-01-20", http_fetch=lambda url: b""
    )
    df2 = mod.load_house_ptr_transactions([marker], "2020-01-01", "2020-12-31")
    assert len(df2) == 1

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM house_ptr_transactions WHERE symbol = :s"), {"s": marker})
