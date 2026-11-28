"""Check the registration form includes the required fixed terms."""

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study


def test_template_requires_pinned_dates_and_comparison() -> None:
    template = (
        (study_dir(__file__) / "2026-MM-DD-forward-check-registration-TEMPLATE.md")
        .read_text()
        .lower()
    )
    for term in (
        "eval_start",
        "eval_end",
        "family_size",
        "cost",
        "tax",
        "spy",
        "same dates",
        "only data dated after the registration commit",
    ):
        assert term in template
