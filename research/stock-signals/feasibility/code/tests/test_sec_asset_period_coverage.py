"""Check annual-period presence selection using invented filing metadata."""

from datetime import date
from zipfile import ZipFile

import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "sec_asset_period_coverage.py")
pytestmark = pytest.mark.study


def test_uses_only_prior_filed_four_month_eligible_periods():
    reports = [
        ("1", "old", date(2019, 12, 31), date(2020, 2, 1)),
        ("1", "new", date(2020, 12, 31), date(2021, 2, 1)),
        ("1", "too_recent", date(2021, 3, 31), date(2021, 5, 1)),
        ("2", "same_day", date(2020, 12, 31), date(2021, 6, 29)),
    ]
    facts = {
        "old": {date(2018, 12, 31), date(2019, 12, 31)},
        "new": {date(2019, 12, 31), date(2020, 12, 31)},
    }
    result = module.summarize(reports, facts, date(2021, 6, 29))
    assert result["ciks_with_eligible_10k_in_archives"] == 1
    assert result["ciks_with_three_asset_periods"] == 1


def test_refuses_ambiguous_prior_period_and_counts_non_december():
    reports = [
        ("1", "a", date(2019, 12, 25), date(2020, 2, 1)),
        ("1", "b", date(2019, 12, 31), date(2020, 2, 1)),
        ("1", "c", date(2020, 12, 31), date(2021, 2, 1)),
        ("2", "d", date(2019, 9, 28), date(2019, 11, 1)),
        ("2", "e", date(2020, 9, 26), date(2020, 11, 1)),
    ]
    facts = {
        "d": {date(2018, 9, 29), date(2019, 9, 28)},
        "e": {date(2019, 9, 28), date(2020, 9, 26)},
    }
    result = module.summarize(reports, facts, date(2021, 6, 29))
    assert result["ciks_with_ambiguous_prior_period"] == 1
    assert result["ciks_with_three_asset_periods"] == 1
    assert result["three_period_ciks_with_non_dec31_latest"] == 1


def test_archive_scan_excludes_non_us_gaap_facts(tmp_path):
    archive = tmp_path / "sec.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(
            "sub.tsv",
            "adsh\tcik\tform\tperiod\tfiled\n"
            "old\t1\t10-K\t20191231\t20200201\n"
            "new\t1\t10-K\t20201231\t20210201\n",
        )
        header = "adsh\ttag\tversion\tuom\tqtrs\tdimn\tcoreg\tvalue\tddate\n"
        template = "{acc}\tAssets\t{taxonomy}\tUSD\t0\t0\t\t100\t{end}\n"
        z.writestr(
            "num.tsv",
            header
            + template.format(acc="old", taxonomy="ifrs/2020", end="20181231")
            + template.format(acc="old", taxonomy="us-gaap/2020", end="20191231")
            + template.format(acc="new", taxonomy="us-gaap/2021", end="20191231")
            + template.format(acc="new", taxonomy="us-gaap/2021", end="20201231"),
        )
    result = module.scan([archive], date(2021, 6, 29))
    assert result["ciks_with_current_and_prior_year_assets"] == 1
    assert result["ciks_with_three_asset_periods"] == 0
