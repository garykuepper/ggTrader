"""Synthetic tests for a metadata-only N-CEN cross-check."""

import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "ncen_common_name_audit.py")
pytestmark = pytest.mark.study


@pytest.fixture
def inputs():
    screener = [
        {"symbol": "AIF", "name": "A Fund Common Stock"},
        {"symbol": "OTHER", "name": "Other"},
    ]
    submissions = pd.DataFrame(
        [{"ACCESSION_NUMBER": "a", "CIK": "0000000123", "FILING_DATE": "25-FEB-2021"}]
    )
    registrants = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": "a",
                "CIK": "0000000123",
                "REGISTRANT_NAME": "A Fund",
                "INVESTMENT_COMPANY_TYPE": "N-2",
            }
        ]
    )
    funds = pd.DataFrame([{"ACCESSION_NUMBER": "a", "FUND_ID": "fund"}])
    securities = pd.DataFrame(
        [
            {
                "FUND_ID": "fund",
                "TICKER_SYMBOL": "AIF",
                "DESCRIPTION": "Preferred stock",
                "SECURITY_CLASS_TITLE": "Preferred",
                "EXCHANGE": "XNYS",
            }
        ]
    )
    return screener, submissions, registrants, funds, securities


def test_reports_exact_match_without_treating_it_as_an_exclusion(inputs):
    result = module.compare(*inputs, cutoff="2021-06-29")
    assert result["screener_common_name_labels"] == 1
    assert result["matched_unique_labels"] == 1
    assert result["matches"][0]["DESCRIPTION"] == "Preferred stock"
    assert result["matches"][0]["TICKER_SYMBOL"] == "AIF"


def test_later_filing_and_non_n2_are_not_included(inputs):
    inputs[1].loc[0, "FILING_DATE"] = "29-JUN-2021"
    assert module.compare(*inputs, cutoff="2021-06-29")["matched_unique_labels"] == 0
    inputs[1].loc[0, "FILING_DATE"] = "25-FEB-2021"
    inputs[2].loc[0, "INVESTMENT_COMPANY_TYPE"] = "N-1A"
    assert module.compare(*inputs, cutoff="2021-06-29")["matched_unique_labels"] == 0


def test_rejects_duplicate_mapping_or_mismatched_issuer(inputs):
    inputs[2].loc[0, "CIK"] = "0000000999"
    with pytest.raises(ValueError, match="CIK"):
        module.compare(*inputs, cutoff="2021-06-29")
    inputs[2].loc[0, "CIK"] = "0000000123"
    duplicate = pd.concat([inputs[3], inputs[3]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        module.compare(inputs[0], inputs[1], inputs[2], duplicate, inputs[4], cutoff="2021-06-29")
