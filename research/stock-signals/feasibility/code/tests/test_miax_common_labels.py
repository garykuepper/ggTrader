"""Test conservative extraction of common-stock labels from MIAX text."""

import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "miax_common_labels.py")
pytestmark = pytest.mark.study


def test_extracts_rows_and_not_wrapped_descriptions():
    page = (
        "MIAX Pearl Equities as of 6/24/21\n"
        "Symbol    Security Name       Security Class    Class Category Inst Type\n"
        "PMM  Trust Shares             CommonStock       Equity\n"
        "DFP  Very long name CommonStock                  Equity\n"
        "     wrapped name CommonStock                    Equity\n"
        "BIL  Fund                    ExchangeTradedFund ETF\n"
    )
    assert module.extract(page) == {"PMM", "DFP"}


def test_missing_header_and_duplicate_labels_fail():
    with pytest.raises(ValueError, match="header"):
        module.extract("PMM Trust CommonStock Equity")
    page = "MIAX Pearl Equities as of 6/24/21\nPMM Trust CommonStock Equity\n"
    with pytest.raises(ValueError, match="duplicate"):
        module.extract(page + page)
