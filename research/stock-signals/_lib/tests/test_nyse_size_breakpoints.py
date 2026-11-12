"""Check NYSE size-breakpoint parsing on invented monthly data."""

import nyse_size_breakpoints
import pytest

pytestmark = pytest.mark.study


def row(month, count=10):
    return ",".join([month, str(count), *(str(n) for n in range(1, 21))])


def test_returns_median_in_millions_at_month_end():
    parsed = nyse_size_breakpoints.parse("preamble\n\n" + row("202401") + "\n" + row("202402"))
    assert parsed.index.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-02-29"]
    assert parsed["median_usd_millions"].tolist() == [10.0, 10.0]
    assert parsed["count"].tolist() == [10, 10]


@pytest.mark.parametrize("text", [row("202401") + "\n" + row("202403"), row("202401", 0)])
def test_rejects_missing_month_or_invalid_count(text):
    with pytest.raises(ValueError):
        nyse_size_breakpoints.parse(text)


@pytest.mark.parametrize("value", ["-99.99", "nan", "0"])
def test_rejects_invalid_median(value):
    fields = row("202401").split(",")
    fields[11] = value
    with pytest.raises(ValueError):
        nyse_size_breakpoints.parse(",".join(fields))


def test_rejects_malformed_final_month_even_after_valid_data():
    with pytest.raises(ValueError):
        nyse_size_breakpoints.parse(row("202401") + "\n" + row("2024O2"))
