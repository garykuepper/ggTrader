"""Specify standalone taxable-lot accounting with hand-computed, offline examples."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from ggTrader.lab.tax_lots import (
    InsufficientCash,
    LedgerEvent,
    Realization,
    TaxLotBook,
    TaxParams,
    verify_events,
)

pytestmark = pytest.mark.study


@pytest.mark.parametrize(
    ("replacement_date", "replacement_basis", "replacement_gain"),
    [
        ("2025-02-21", 900.0, 100.0),  # Sale + 20 days.
        ("2025-03-03", 900.0, 100.0),  # Sale + 30 days: inclusive.
        ("2025-03-04", 700.0, 300.0),  # Sale + 31 days: excluded.
    ],
)
def test_wash_sale_forward_window_disallows_loss(
    replacement_date: str, replacement_basis: float, replacement_gain: float
) -> None:
    """A later same-symbol purchase carries the matched loss into its basis."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    original_id = book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    loss = book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2025-02-01"))
    assert loss == [
        Realization(original_id, "XYZ", 10.0, 800.0, 1_000.0, -200.0, "ST", "provisional", 0.0)
    ]
    before_repurchase = deepcopy(book.events)

    replacement_id = book.buy("XYZ", 10.0, 70.0, pd.Timestamp(replacement_date))
    assert book.events[: len(before_repurchase)] == before_repurchase
    if replacement_basis == 900.0:
        # $200 loss * 10/10 matched; $700 purchase + $200 wash = $900 basis.
        assert len(book.events) >= len(before_repurchase) + 2  # Buy plus adjustment.
    realized = book.sell("XYZ", 10.0, 100.0, pd.Timestamp("2025-06-01"))
    assert realized == [
        Realization(
            replacement_id,
            "XYZ",
            10.0,
            1_000.0,
            replacement_basis,
            replacement_gain,
            "ST",
            "final",
            0.0,
        )
    ]
    # Inside: $100 taxable; outside: -$200 + $300 = $100; both owe $100 * 24%.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(24.0)


def test_wash_sale_backward_window_partial_quantity() -> None:
    """Only the separate 40-share lot replaces a sale out of the 100-share lot."""
    book = TaxLotBook(TaxParams(), cash=20_000.0)
    replacement_id = book.buy("XYZ", 40.0, 80.0, pd.Timestamp("2025-01-01"))
    sold_id = book.buy("XYZ", 100.0, 100.0, pd.Timestamp("2025-01-15"))
    loss = book.sell("XYZ", 60.0, 90.0, pd.Timestamp("2025-01-31"))
    # Loss = 60 * ($90 - $100) = -$600; disallowed = $600 * 40/60 = $400.
    # The 40 unsold shares of the selling lot are excluded, not replacements.
    assert loss == [
        Realization(sold_id, "XYZ", 60.0, 5_400.0, 6_000.0, -600.0, "ST", "provisional", 400.0)
    ]
    realized = book.sell("XYZ", 80.0, 110.0, pd.Timestamp("2025-04-01"))
    # HIFO: residual 40 * $100; replacement 40 * ($80 + $400/40) = $3,600.
    assert realized == [
        Realization(sold_id, "XYZ", 40.0, 4_400.0, 4_000.0, 400.0, "ST", "final", 0.0),
        Realization(replacement_id, "XYZ", 40.0, 4_400.0, 3_600.0, 800.0, "ST", "final", 0.0),
    ]
    # (-$600 + $400 disallowed + $400 + $800) * 24% = $240.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(240.0)


def test_wash_sale_excludes_lots_being_sold() -> None:
    """A partially sold recent lot cannot replace itself; another symbol cannot either."""
    book = TaxLotBook(TaxParams(), cash=20_000.0)
    lot_id = book.buy("XYZ", 100.0, 100.0, pd.Timestamp("2025-05-01"))
    book.buy("OTHER", 100.0, 10.0, pd.Timestamp("2025-05-02"))
    loss = book.sell("XYZ", 60.0, 90.0, pd.Timestamp("2025-05-20"))
    # 60 * ($90 - $100) = -$600, with zero eligible same-symbol replacements.
    assert loss == [
        Realization(lot_id, "XYZ", 60.0, 5_400.0, 6_000.0, -600.0, "ST", "provisional", 0.0)
    ]
    # Residual basis stays 40 * $100 = $4,000; gain $400 cannot absorb all -$600.
    assert book.sell("XYZ", 40.0, 110.0, pd.Timestamp("2025-07-01")) == [
        Realization(lot_id, "XYZ", 40.0, 4_400.0, 4_000.0, 400.0, "ST", "final", 0.0)
    ]
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("sale_date", "term", "tax"),
    [("2025-01-21", "ST", 24.0), ("2025-01-22", "LT", 15.0)],
)
def test_wash_sale_adjusts_holding_period(sale_date: str, term: str, tax: float) -> None:
    """Tack elapsed holding days, rather than copying the old acquisition date."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2024-01-01"))
    book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2024-10-27"))
    replacement_id = book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2024-11-16"))
    # Jan 1 -> Oct 27 is 300 days; Nov 16 - 300 days = Jan 21, 2024.
    # Jan 21, 2025 is exactly one year (ST); Jan 22 is strictly later (LT).
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(0.0)
    assert book.sell("XYZ", 10.0, 100.0, pd.Timestamp(sale_date)) == [
        Realization(replacement_id, "XYZ", 10.0, 1_000.0, 900.0, 100.0, term, "final", 0.0)
    ]
    # $1,000 proceeds - ($700 purchase + $200 wash) = $100 gain; tax $24/$15.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(tax)


def test_wash_sale_loss_provisional_until_window_closes() -> None:
    """A pending loss is recorded provisionally; day-31 buys cannot reopen it."""
    book = TaxLotBook(TaxParams(), cash=3_000.0)
    book.buy("GAIN", 1.0, 100.0, pd.Timestamp("2024-01-02"))
    book.sell("GAIN", 1.0, 400.0, pd.Timestamp("2024-06-03"))
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2024-11-01"))
    loss = book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2024-12-20"))[0]
    assert loss.status == "provisional"
    assert loss.gain == -200.0
    assert loss.disallowed == 0.0
    # $300 ST gain - $200 provisional loss = $100; $100 * 24% = $24.
    assert book.year_end(2024, pd.Timestamp("2025-01-19")) == pytest.approx(24.0)
    assert book.year_end(2024, pd.Timestamp("2025-01-20")) == pytest.approx(24.0)
    closed_events = deepcopy(book.events)
    replacement_id = book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2025-01-20"))
    assert book.events[: len(closed_events)] == closed_events
    assert book.amended_tax(2024) == pytest.approx(0.0)
    # Dec 20 + 30 days = Jan 19; Jan 20 is outside, so basis remains $700.
    assert book.sell("XYZ", 10.0, 100.0, pd.Timestamp("2025-05-01")) == [
        Realization(replacement_id, "XYZ", 10.0, 1_000.0, 700.0, 300.0, "ST", "final", 0.0)
    ]
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(72.0)
    # There is no public as-of realization accessor: the frozen sale-time object
    # remains provisional; finality above is checked through the closed window.
    assert loss.status == "provisional"


def test_late_december_loss_january_repurchase_amends_prior_year() -> None:
    """A January wash appends a correction to the already assessed December year."""
    book = TaxLotBook(TaxParams(), cash=3_000.0)
    book.buy("GAIN", 1.0, 100.0, pd.Timestamp("2024-01-02"))
    book.sell("GAIN", 1.0, 400.0, pd.Timestamp("2024-06-03"))
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2024-11-01"))
    loss = book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2024-12-20"))[0]
    # Initial tax: ($300 gain - $200 loss) * 24% = $24.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(24.0)
    assert book.amended_tax(2024) == pytest.approx(0.0)
    assessed_events = deepcopy(book.events)
    replacement_id = book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2025-01-10"))
    assert book.events[: len(assessed_events)] == assessed_events
    assert len(book.events) >= len(assessed_events) + 2  # Buy and later wash correction.
    assert any(
        event.date == pd.Timestamp("2025-01-10") for event in book.events[len(assessed_events) :]
    )
    assert loss.gain == -200.0
    assert loss.disallowed == 0.0
    assert loss.status == "provisional"
    # Revised tax = $300 * 24% = $72; amendment = $72 - $24 = $48.
    assert book.amended_tax(2024) == pytest.approx(48.0)
    assert book.year_end(2024, pd.Timestamp("2025-01-21")) == pytest.approx(72.0)
    assert book.sell("XYZ", 10.0, 100.0, pd.Timestamp("2025-05-01")) == [
        Realization(replacement_id, "XYZ", 10.0, 1_000.0, 900.0, 100.0, "ST", "final", 0.0)
    ]
    # The $200 is deferred into January's lot, not a second loss or new-year tax.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(24.0)
    assert book.verify_chain()


def test_hifo_tie_break_is_deterministic() -> None:
    """Sort by basis descending, acquisition ascending, then lot id ascending."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    low_lt_id = book.buy("XYZ", 1.0, 100.0, pd.Timestamp("2023-01-01"))
    early_id = book.buy("XYZ", 1.0, 110.0, pd.Timestamp("2024-01-01"))
    tied_ids = [book.buy("XYZ", 1.0, 110.0, pd.Timestamp("2024-06-01")) for _ in range(2)]
    high_id = book.buy("XYZ", 1.0, 120.0, pd.Timestamp("2024-07-01"))
    assert len({low_lt_id, early_id, high_id, *tied_ids}) == 5
    assert all(isinstance(lot_id, str) for lot_id in [low_lt_id, early_id, high_id, *tied_ids])
    realized = book.sell("XYZ", 5.0, 130.0, pd.Timestamp("2024-08-01"))
    assert [item.lot_id for item in realized] == [high_id, early_id, *sorted(tied_ids), low_lt_id]
    assert [item.basis for item in realized] == [120.0, 110.0, 110.0, 110.0, 100.0]
    assert [item.gain for item in realized] == [10.0, 20.0, 20.0, 20.0, 30.0]
    assert [item.term for item in realized] == ["ST", "ST", "ST", "ST", "LT"]
    # $650 proceeds - $550 basis = $100 gain: $70 ST * 24% + $30 LT * 15% = $21.30.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(21.3)


@pytest.mark.parametrize(
    ("st_sale_price", "lt_sale_price", "tax"),
    [(500.0, 300.0, 39.0), (300.0, 400.0, 15.0), (500.0, 150.0, 12.0)],
)
def test_carryforward_preserves_character(
    st_sale_price: float, lt_sale_price: float, tax: float
) -> None:
    """Keep ST/LT losses distinct and cross-net only opposite-sign net buckets."""
    book = TaxLotBook(TaxParams(), cash=10_000.0)
    book.buy("LT_LOSS", 1.0, 200.0, pd.Timestamp("2023-01-02"))
    book.buy("LT_GAIN", 1.0, 100.0, pd.Timestamp("2023-01-02"))
    book.buy("ST_LOSS", 1.0, 400.0, pd.Timestamp("2024-01-02"))
    book.sell("LT_LOSS", 1.0, 100.0, pd.Timestamp("2024-06-03"))
    book.sell("ST_LOSS", 1.0, 100.0, pd.Timestamp("2024-06-03"))
    # 2024 carries -$300 ST and -$100 LT; repeating assessment cannot duplicate them.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(0.0)
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(0.0)
    book.buy("ST_GAIN", 1.0, 100.0, pd.Timestamp("2025-01-02"))
    book.sell("ST_GAIN", 1.0, st_sale_price, pd.Timestamp("2025-06-03"))
    book.sell("LT_GAIN", 1.0, lt_sale_price, pd.Timestamp("2025-06-03"))
    # Gains $400 ST/$200 LT -> net $100/$100 -> $24 + $15 = $39.
    # Gains $200 ST/$300 LT -> net -$100/$200 -> $100 LT * 15% = $15.
    # Gains $400 ST/$50 LT -> net $100/-$50 -> $50 ST * 24% = $12.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(tax)
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(tax)


def test_no_3000_ordinary_income_credit() -> None:
    """Capital losses cannot shelter ordinary income or add terminal wealth."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    book.sell("XYZ", 10.0, 90.0, pd.Timestamp("2025-05-01"))
    book.interest(50.0, pd.Timestamp("2025-06-01"))
    # Capital loss -$100 carries unused; $50 interest * 24% = $12, not zero.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(12.0)
    # $1,000 - $100 trading loss + $50 interest - $12 tax = $938; CF adds $0.
    assert book.terminal_liquidate({}, pd.Timestamp("2025-12-31")) == pytest.approx(938.0)


@pytest.mark.parametrize(
    ("wash", "sale_date", "term", "first_basis", "second_basis", "tax"),
    [
        (False, "2024-01-01", "ST", 400.0, 600.0, 48.0),
        (False, "2024-01-02", "LT", 400.0, 600.0, 30.0),
        (True, "2024-01-21", "ST", 360.0, 540.0, 72.0),
        (True, "2024-01-22", "LT", 360.0, 540.0, 45.0),
    ],
)
def test_split_preserves_basis_and_holding_period(
    wash: bool,
    sale_date: str,
    term: str,
    first_basis: float,
    second_basis: float,
    tax: float,
) -> None:
    """A 2:1 split also preserves basis and tacked dates on a washed lot."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    lot_id = book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2023-01-01"))
    if wash:
        book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2023-10-28"))
        lot_id = book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2023-11-17"))
    book.split("XYZ", 2.0, pd.Timestamp("2023-12-01"))
    assert book.year_end(2023, pd.Timestamp("2023-12-31")) == pytest.approx(0.0)
    first = book.sell("XYZ", 8.0, 60.0, pd.Timestamp(sale_date))[0]
    second = book.sell("XYZ", 12.0, 60.0, pd.Timestamp(sale_date))[0]
    # Bare: 10 * $100 -> 20 * $50; washed: ($700 + $200) -> 20 * $45.
    # Wash holding: Nov 17 - 300 days = Jan 21, 2023; a split changes neither date.
    assert first.lot_id == second.lot_id == lot_id
    assert (first.qty, first.proceeds, first.basis, first.term) == (8.0, 480.0, first_basis, term)
    assert (second.qty, second.proceeds, second.basis, second.term) == (
        12.0,
        720.0,
        second_basis,
        term,
    )
    # $1,200 proceeds - $1,000/$900 basis = $200/$300 gain, at 24% or 15%.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(tax)


def test_rename_keeps_lot_dates() -> None:
    """A ticker change moves all lots without resetting basis or acquisition."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    old_id = book.buy("OLD", 5.0, 100.0, pd.Timestamp("2023-01-01"))
    new_id = book.buy("OLD", 5.0, 120.0, pd.Timestamp("2024-01-01"))
    book.rename("OLD", "NEW", pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError):
        book.sell("OLD", 1.0, 150.0, pd.Timestamp("2024-01-03"))
    # HIFO sells 5 * $120 first (ST gain $150), then 1 * $100 (LT gain $50).
    assert book.sell("NEW", 6.0, 150.0, pd.Timestamp("2024-01-03")) == [
        Realization(new_id, "NEW", 5.0, 750.0, 600.0, 150.0, "ST", "final", 0.0),
        Realization(old_id, "NEW", 1.0, 150.0, 100.0, 50.0, "LT", "final", 0.0),
    ]
    assert book.sell("NEW", 4.0, 150.0, pd.Timestamp("2024-01-04")) == [
        Realization(old_id, "NEW", 4.0, 600.0, 400.0, 200.0, "LT", "final", 0.0)
    ]
    # $150 ST * 24% + ($50 + $200) LT * 15% = $36 + $37.50 = $73.50.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(73.5)


@pytest.mark.parametrize(
    ("qualified", "tax", "final_cash"), [(True, 15.0, 1_085.0), (False, 24.0, 1_076.0)]
)
def test_qualified_vs_unqualified_dividend_rates(
    qualified: bool, tax: float, final_cash: float
) -> None:
    """Distribution classification, not the young lot's age, sets its tax rate."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 10.0, 10.0, pd.Timestamp("2025-01-02"))
    # 10 shares * $10/share = $100 cash dividend; qualified $15 tax, else $24.
    assert book.dividend("XYZ", 10.0, pd.Timestamp("2025-03-01"), qualified) == 100.0
    book.sell("XYZ", 10.0, 10.0, pd.Timestamp("2025-04-01"))
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(tax)
    # $1,000 + $100 dividend - $15/$24 = $1,085/$1,076.
    assert book.terminal_liquidate({}, pd.Timestamp("2025-12-31")) == pytest.approx(final_cash)


def test_interest_taxed_as_ordinary() -> None:
    """Interest credits cash and is taxed at the ordinary/ST rate."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.interest(100.0, pd.Timestamp("2025-06-30"))
    # $100 * 24% = $24 tax; $1,000 + $100 - $24 = $1,076 final cash.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(24.0)
    assert book.terminal_liquidate({}, pd.Timestamp("2025-12-31")) == pytest.approx(1_076.0)


def test_pay_tax_raises_insufficient_cash_not_same_day_sale() -> None:
    """Tax payment never sells positions; an explicit later fill can fund it."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2024-01-02"))
    book.sell("XYZ", 10.0, 200.0, pd.Timestamp("2024-06-01"))
    replacement_id = book.buy("XYZ", 10.0, 200.0, pd.Timestamp("2024-06-02"))
    # $1,000 ST gain * 24% = $240 due; reinvesting all $2,000 leaves cash $0.
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(240.0)
    before_payment = deepcopy(book.events)
    with pytest.raises(InsufficientCash):
        book.pay_tax(240.0, pd.Timestamp("2024-12-31"))
    assert book.events == before_payment
    assert book.liquidation_value({"XYZ": 200.0}, pd.Timestamp("2024-12-31")) == pytest.approx(
        1_760.0
    )
    # Explicit next-session sale: 2 * $200 = $400 cash, no gain; pay $240 -> $160.
    assert book.sell("XYZ", 2.0, 200.0, pd.Timestamp("2025-01-02")) == [
        Realization(replacement_id, "XYZ", 2.0, 400.0, 400.0, 0.0, "ST", "final", 0.0)
    ]
    book.pay_tax(240.0, pd.Timestamp("2025-01-02"))
    # $160 cash + 8 * $200 shares = $1,760; assessed tax cannot be charged twice.
    assert book.terminal_liquidate({"XYZ": 200.0}, pd.Timestamp("2025-01-02")) == pytest.approx(
        1_760.0
    )


def test_oversell_raises_value_error() -> None:
    """An invalid sale leaves the original quantity, cash and ledger intact."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    lot_id = book.buy("XYZ", 5.0, 100.0, pd.Timestamp("2025-01-02"))
    before_sale = deepcopy(book.events)
    with pytest.raises(ValueError):
        book.sell("XYZ", 6.0, 110.0, pd.Timestamp("2025-06-01"))
    assert book.events == before_sale
    assert book.sell("XYZ", 5.0, 110.0, pd.Timestamp("2025-06-01")) == [
        Realization(lot_id, "XYZ", 5.0, 550.0, 500.0, 50.0, "ST", "final", 0.0)
    ]
    # $1,000 + $50 gain - ($50 * 24%) = $1,038; failed sale contributes nothing.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(12.0)
    assert book.terminal_liquidate({}, pd.Timestamp("2025-12-31")) == pytest.approx(1_038.0)


def test_spy_benchmark_uses_identical_engine() -> None:
    """Identical candidate and benchmark fills use identical accounting and cash."""
    params = TaxParams(st_rate=0.24, lt_rate=0.15)
    candidate = TaxLotBook(params, cash=1_000.0)
    benchmark = TaxLotBook(params, cash=1_000.0)
    final_values = []
    for book in (candidate, benchmark):
        original_id = book.buy("SPY", 10.0, 10.0, pd.Timestamp("2023-01-01"))
        assert book.dividend("SPY", 1.0, pd.Timestamp("2023-06-01"), True) == 10.0
        book.buy("SPY", 1.0, 10.0, pd.Timestamp("2023-06-02"))  # Explicit reinvestment.
        # $10 qualified dividend * 15% = $1.50, paid from each book's cash.
        assert book.year_end(2023, pd.Timestamp("2023-12-31")) == pytest.approx(1.5)
        book.pay_tax(1.5, pd.Timestamp("2023-12-31"))
        book.interest(20.0, pd.Timestamp("2024-01-01"))
        assert book.sell("SPY", 2.0, 20.0, pd.Timestamp("2024-01-02")) == [
            Realization(original_id, "SPY", 2.0, 40.0, 20.0, 20.0, "LT", "final", 0.0)
        ]
        # Gross cash after all fills = $1,000 + $10 dividend + $20 interest
        # + $110 total gains - $1.50 paid tax = $1,138.50.
        # Remaining tax = $100 LT * 15% + $10 ST * 24% + $20 * 24% = $22.20.
        assert book.liquidation_value({"SPY": 20.0}, pd.Timestamp("2024-01-02")) == pytest.approx(
            1_116.3
        )
        final_values.append(book.terminal_liquidate({"SPY": 20.0}, pd.Timestamp("2024-01-02")))
    assert final_values == pytest.approx([1_116.3, 1_116.3])
    assert final_values[0] == final_values[1]  # Same params/trades -> same benchmark wealth.


def test_liquidation_value_includes_unrealized_lot_tax_and_does_not_mutate() -> None:
    """Hypothetical wealth deducts accrued and unrealized tax without a fill."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("LONG", 5.0, 100.0, pd.Timestamp("2023-01-01"))
    book.buy("SHORT", 2.0, 50.0, pd.Timestamp("2024-01-01"))
    book.sell("SHORT", 1.0, 60.0, pd.Timestamp("2024-06-01"))
    book.interest(50.0, pd.Timestamp("2024-06-02"))
    date = pd.Timestamp("2024-12-31")
    prices = {"LONG": 120.0, "SHORT": 70.0}
    # Cash = $1,000 - $500 - $100 + $60 + $50 = $510; market value = $670.
    # Accrued tax ($10 ST + $50 interest) * 24% = $14.40.
    # Unrealized tax = $100 LT * 15% + $20 ST * 24% = $19.80.
    # $510 + $670 - $14.40 - $19.80 = $1,145.80.
    assert book.year_end(2024, date) == pytest.approx(14.4)
    before_valuation = deepcopy(book.events)
    assert book.liquidation_value(prices, date) == pytest.approx(1_145.8)
    assert book.liquidation_value(prices, date) == pytest.approx(1_145.8)
    assert book.events == before_valuation
    assert book.year_end(2024, date) == pytest.approx(14.4)
    # A real liquidation must still find every original share and the same net tax.
    assert book.terminal_liquidate(prices, date) == pytest.approx(1_145.8)
    assert book.verify_chain()


def test_terminal_liquidation_after_tax_cagr_matches_hand_calculation() -> None:
    """A two-calendar-year endpoint pays the LT tax before calculating CAGR."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2023-01-01"))
    # 10 * $144 = $1,440; gain $440 * 15% = $66; final cash $1,374.
    final_cash = book.terminal_liquidate({"XYZ": 144.0}, pd.Timestamp("2025-01-01"))
    assert final_cash == pytest.approx(1_374.0)
    # Two-year CAGR = sqrt($1,374 / $1,000) - 1 = 0.17217746096740827.
    cagr = (final_cash / 1_000.0) ** 0.5 - 1.0
    assert cagr == pytest.approx(0.17217746096740827)
    assert book.liquidation_value({}, pd.Timestamp("2025-01-01")) == pytest.approx(1_374.0)
    assert book.verify_chain()


def test_ledger_append_only_correction_preserves_original_event() -> None:
    """Two forward replacements append corrections without rewriting sale history."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    original_loss = book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2025-03-01"))[0]
    original_events = deepcopy(book.events)
    assert isinstance(original_events, tuple)
    first_id = book.buy("XYZ", 4.0, 70.0, pd.Timestamp("2025-03-11"))
    assert book.events[: len(original_events)] == original_events
    assert len(book.events) >= len(original_events) + 2
    first_adjusted_events = deepcopy(book.events)
    second_id = book.buy("XYZ", 6.0, 60.0, pd.Timestamp("2025-03-21"))
    assert book.events[: len(first_adjusted_events)] == first_adjusted_events
    assert len(book.events) >= len(first_adjusted_events) + 2
    # $200 loss: 4/10 -> $80 added to $280 basis; remaining 6/10 -> $120 + $360.
    assert book.sell("XYZ", 10.0, 100.0, pd.Timestamp("2025-06-01")) == [
        Realization(first_id, "XYZ", 4.0, 400.0, 360.0, 40.0, "ST", "final", 0.0),
        Realization(second_id, "XYZ", 6.0, 600.0, 480.0, 120.0, "ST", "final", 0.0),
    ]
    assert original_loss.gain == -200.0
    assert original_loss.disallowed == 0.0
    assert original_loss.status == "provisional"
    with pytest.raises(FrozenInstanceError):
        setattr(original_loss, "disallowed", 200.0)
    # Fully deferred loss; ($40 + $120) * 24% = $38.40, not a negative tax.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(38.4)
    assert book.events[: len(original_events)] == original_events
    assert verify_events(book.events)
    assert book.verify_chain()


def test_ledger_chain_detects_tampering() -> None:
    """Payload replacement invalidates the copied SHA-256 chain, not the real one."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 1.0, 100.0, pd.Timestamp("2025-01-01"))
    book.interest(10.0, pd.Timestamp("2025-01-02"))
    book.sell("XYZ", 1.0, 110.0, pd.Timestamp("2025-03-01"))
    original_events = deepcopy(book.events)
    assert isinstance(original_events, tuple)
    assert len(original_events) >= 3
    for event in original_events:
        assert isinstance(event, LedgerEvent)
        assert len(event.hash) == 64  # SHA-256 is 256 bits / 4 bits per hex digit.
        assert int(event.hash, 16) >= 0
    for previous, following in zip(original_events, original_events[1:]):
        assert following.seq == previous.seq + 1
        assert following.prev_hash == previous.hash
    with pytest.raises(FrozenInstanceError):
        setattr(book.events[0], "seq", -1)
    tampered_event = replace(original_events[0], payload={"tampered": True})
    tampered_events = (tampered_event, *original_events[1:])
    assert verify_events(original_events)
    assert not verify_events(tampered_events)
    assert book.events == original_events
    assert book.verify_chain()
    # $10 gain + $10 interest = $20 ordinary/ST * 24% = $4.80.
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(4.8)


def test_loss_status_becomes_final_after_window() -> None:
    """Day 30 remains provisional; day 31 finalizes the unchanged sale record."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    date = pd.Timestamp("2025-03-01")
    loss = book.sell("XYZ", 10.0, 80.0, date)[0]
    before = deepcopy(book.events)
    assert loss.date == date
    assert book.loss_status(loss, date + pd.Timedelta(days=30)) == "provisional"
    assert book.loss_status(loss, date + pd.Timedelta(days=31)) == "final"
    assert loss.status == "provisional"
    assert book.events == before


def test_wash_vs_non_wash_year_end_discriminates() -> None:
    """An inside-window repurchase raises tax from $24 to $72 before resale."""
    taxes = []
    for replacement_date in ("2025-03-31", "2025-04-01"):
        book = TaxLotBook(TaxParams(), cash=2_000.0)
        book.buy("GAIN", 1.0, 100.0, pd.Timestamp("2025-01-01"))
        book.sell("GAIN", 1.0, 400.0, pd.Timestamp("2025-02-01"))
        book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-02-01"))
        book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2025-03-01"))
        book.buy("XYZ", 10.0, 70.0, pd.Timestamp(replacement_date))
        taxes.append(book.year_end(2025, pd.Timestamp("2025-12-31")))
    # March 31 is +30 days: $300 taxable. April 1 allows the $200 loss.
    assert taxes == pytest.approx([72.0, 24.0])


def test_year_end_is_idempotent_for_assessed_year() -> None:
    """Repeated assessment returns the same tax without another liability or event."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.interest(100.0, pd.Timestamp("2024-06-01"))
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(24.0)
    assessed = deepcopy(book.events)
    assert book.year_end(2024, pd.Timestamp("2025-01-20")) == pytest.approx(24.0)
    assert book.events == assessed
    assert book.unpaid_tax == pytest.approx(24.0)
    assert book.cash == pytest.approx(1_100.0)  # Assessment is not payment.
    assert book.terminal_liquidate({}, pd.Timestamp("2025-01-20")) == pytest.approx(1_076.0)


def test_pay_tax_is_credited_against_unpaid_assessment_globally() -> None:
    """One later-year payment credits multiple assessed years, without reassessment."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.interest(100.0, pd.Timestamp("2023-06-01"))
    assert book.year_end(2023, pd.Timestamp("2023-12-31")) == pytest.approx(24.0)
    book.interest(200.0, pd.Timestamp("2024-06-01"))
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(48.0)
    book.pay_tax(30.0, pd.Timestamp("2025-01-02"))
    assert book.unpaid_tax == pytest.approx(42.0)
    # Gross wealth $1,300; $30 already paid; terminal pays only $42 more.
    assert book.liquidation_value({}, pd.Timestamp("2025-01-02")) == pytest.approx(1_228.0)
    assert book.terminal_liquidate({}, pd.Timestamp("2025-01-02")) == pytest.approx(1_228.0)
    assert book.terminal_liquidate({}, pd.Timestamp("2025-01-02")) == pytest.approx(1_228.0)


def test_liquidation_value_deducts_assessed_unpaid_once() -> None:
    """Partial payments and repeated marks never double-count assessed tax."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 5.0, 100.0, pd.Timestamp("2024-01-01"))
    book.interest(100.0, pd.Timestamp("2024-06-01"))
    date = pd.Timestamp("2024-12-31")
    assert book.year_end(2024, date) == pytest.approx(24.0)
    book.pay_tax(10.0, date)
    before = deepcopy(book.events)
    # $590 cash + $600 shares - $14 unpaid - $100 unrealized ST * 24%.
    assert book.liquidation_value({"XYZ": 120.0}, date) == pytest.approx(1_152.0)
    assert book.liquidation_value({"XYZ": 120.0}, date) == pytest.approx(1_152.0)
    assert book.events == before
    assert book.cash == pytest.approx(590.0)
    assert book.unpaid_tax == pytest.approx(14.0)
    assert book.terminal_liquidate({"XYZ": 120.0}, date) == pytest.approx(1_152.0)


def test_paid_original_assessment_leaves_only_wash_amendment_unpaid() -> None:
    """A January adjustment adds $48 without rewriting or recharging December's $24."""
    book = TaxLotBook(TaxParams(), cash=3_000.0)
    book.buy("GAIN", 1.0, 100.0, pd.Timestamp("2024-01-02"))
    book.sell("GAIN", 1.0, 400.0, pd.Timestamp("2024-06-03"))
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2024-11-01"))
    book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2024-12-20"))
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(24.0)
    book.pay_tax(24.0, pd.Timestamp("2024-12-31"))
    before = deepcopy(book.events)
    book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2025-01-10"))
    assert book.events[: len(before)] == before
    assert book.amended_tax(2024) == pytest.approx(48.0)
    assert book.unpaid_tax == pytest.approx(48.0)
    assert book.year_end(2024, pd.Timestamp("2025-01-21")) == pytest.approx(72.0)
    corrections = [event for event in book.events[len(before) :] if "ref_seq" in event.payload]
    assert len(corrections) == 2  # Loss correction and assessment correction.
    assert all(book.events[event.payload["ref_seq"] - 1] in before for event in corrections)
    # Replacement sold at adjusted basis: gross wealth $3,300; total tax $72.
    assert book.terminal_liquidate({"XYZ": 90.0}, pd.Timestamp("2025-06-01")) == pytest.approx(
        3_228.0
    )
    assert book.verify_chain()


def test_lot_ids_remain_lexically_ordered_across_digit_boundary() -> None:
    """Equal-date, equal-basis HIFO ties follow zero-padded creation ids."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    date = pd.Timestamp("2025-01-01")
    ids = [book.buy("XYZ", 1.0, 10.0, date) for _ in range(12)]
    assert ids == [f"L{number:06d}" for number in range(1, 13)]
    assert sorted(ids) == ids
    assert [item.lot_id for item in book.sell("XYZ", 12.0, 20.0, date)] == ids


def test_partial_forward_match_remains_provisional_until_fully_matched() -> None:
    """As-of finality reflects partial, complete and same-date forward corrections."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    date = pd.Timestamp("2025-03-01")
    loss = book.sell("XYZ", 10.0, 80.0, date)[0]
    book.buy("XYZ", 4.0, 70.0, date)
    assert book.loss_status(loss, date) == "provisional"
    book.buy("XYZ", 6.0, 70.0, date + pd.Timedelta(days=20))
    assert book.loss_status(loss, date) == "provisional"
    assert book.loss_status(loss, date + pd.Timedelta(days=20)) == "final"
    assert loss.disallowed == 0.0
    assert loss.status == "provisional"


def test_partial_replacement_keeps_unmatched_shares_basis_and_dates() -> None:
    """A larger replacement purchase contains distinct washed and unwashed slices."""
    book = TaxLotBook(TaxParams(), cash=3_000.0)
    book.buy("XYZ", 4.0, 100.0, pd.Timestamp("2023-01-01"))
    book.sell("XYZ", 4.0, 80.0, pd.Timestamp("2023-10-28"))
    replacement = book.buy("XYZ", 10.0, 70.0, pd.Timestamp("2023-11-17"))
    # Four shares get $80 basis adjustment and 300 holding days; six do not.
    sales = book.sell("XYZ", 10.0, 100.0, pd.Timestamp("2024-01-22"))
    assert [(item.lot_id, item.qty, item.basis, item.gain, item.term) for item in sales] == [
        (replacement, 4.0, 360.0, 40.0, "LT"),
        (replacement, 6.0, 420.0, 180.0, "ST"),
    ]
    assert book.year_end(2024, pd.Timestamp("2024-12-31")) == pytest.approx(49.2)


def test_replacement_shares_cannot_disallow_two_overlapping_losses() -> None:
    """A replacement share's matching capacity is consumed by the first loss."""
    book = TaxLotBook(TaxParams(), cash=3_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    book.sell("XYZ", 5.0, 80.0, pd.Timestamp("2025-03-01"))
    book.sell("XYZ", 5.0, 80.0, pd.Timestamp("2025-03-02"))
    book.buy("XYZ", 5.0, 70.0, pd.Timestamp("2025-03-03"))
    # $200 losses, only $100 disallowed; a $300 gain leaves $200 taxable.
    book.buy("GAIN", 1.0, 100.0, pd.Timestamp("2025-04-01"))
    book.sell("GAIN", 1.0, 400.0, pd.Timestamp("2025-06-01"))
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(48.0)


def test_split_scales_open_wash_matching_quantities() -> None:
    """A split between sale and repurchase scales share matches, not loss dollars."""
    book = TaxLotBook(TaxParams(), cash=2_000.0)
    book.buy("XYZ", 10.0, 100.0, pd.Timestamp("2025-01-01"))
    book.sell("XYZ", 10.0, 80.0, pd.Timestamp("2025-03-01"))
    book.split("XYZ", 2.0, pd.Timestamp("2025-03-10"))
    lot_id = book.buy("XYZ", 20.0, 35.0, pd.Timestamp("2025-03-11"))
    sale = book.sell("XYZ", 20.0, 50.0, pd.Timestamp("2025-06-01"))[0]
    assert (sale.lot_id, sale.basis, sale.gain) == (lot_id, 900.0, 100.0)
    assert book.year_end(2025, pd.Timestamp("2025-12-31")) == pytest.approx(24.0)


def test_ledger_payload_is_immutable_and_hash_is_independently_reproducible() -> None:
    """Frozen events protect their payloads and use the documented canonical JSON."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.buy("XYZ", 1.0, 100.0, pd.Timestamp("2025-01-01"))
    event = book.events[0]
    with pytest.raises(TypeError):
        event.payload["qty"] = 2.0
    encoded = json.dumps(
        {
            "seq": event.seq,
            "kind": event.kind,
            "date": event.date.isoformat(),
            "payload": event.payload,
            "prev_hash": event.prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert event.hash == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert not verify_events((replace(event, seq=2),))
    assert book.verify_chain()


def test_pay_tax_rejects_overpayment_without_mutation() -> None:
    """Payments may settle assessed liabilities, not create an external tax credit."""
    book = TaxLotBook(TaxParams(), cash=1_000.0)
    book.interest(100.0, pd.Timestamp("2025-06-01"))
    date = pd.Timestamp("2025-12-31")
    assert book.year_end(2025, date) == pytest.approx(24.0)
    before = deepcopy(book.events)
    with pytest.raises(ValueError, match="exceeds assessed unpaid tax"):
        book.pay_tax(25.0, date)
    assert book.cash == pytest.approx(1_100.0)
    assert book.unpaid_tax == pytest.approx(24.0)
    assert book.events == before


def test_tax_params_are_frozen() -> None:
    """The primary marginal rates cannot change after constructing a book."""
    params = TaxParams()
    assert (params.st_rate, params.lt_rate) == (0.24, 0.15)
    with pytest.raises(FrozenInstanceError):
        setattr(params, "st_rate", 0.30)


def test_fractional_sells_tolerate_binary_float_roundoff() -> None:
    """Selling 0.1 then 0.2 of a 0.3-share lot must drain it exactly, not raise."""
    book = TaxLotBook(TaxParams(), cash=1.0)
    book.buy("X", 0.3, 1.0, pd.Timestamp("2025-01-01"))
    book.sell("X", 0.1, 1.0, pd.Timestamp("2025-03-01"))
    book.sell("X", 0.2, 1.0, pd.Timestamp("2025-03-02"))
    assert book.liquidation_value({"X": 1.0}, pd.Timestamp("2025-03-02")) == pytest.approx(1.0)


def test_fractional_buy_tolerates_cost_roundoff_but_not_real_overspend() -> None:
    """0.1 shares at 3.0 costs 0.30000000000000004; cash of 0.3 must fund it."""
    book = TaxLotBook(TaxParams(), cash=0.3)
    book.buy("X", 0.1, 3.0, pd.Timestamp("2025-01-01"))
    assert book.cash == pytest.approx(0.0, abs=1e-12)
    poor = TaxLotBook(TaxParams(), cash=0.3)
    with pytest.raises(InsufficientCash):
        poor.buy("X", 0.2, 3.0, pd.Timestamp("2025-01-01"))  # costs 0.6: a real overspend


def test_tiny_positive_residual_position_is_not_deleted() -> None:
    """A 1e-11-share lot worth $10 keeps its $9 remainder after a $1 partial sale."""
    book = TaxLotBook(TaxParams(), cash=100.0)
    book.buy("X", 1e-11, 1e12, pd.Timestamp("2025-01-01"))
    book.sell("X", 1e-12, 1e12, pd.Timestamp("2025-03-01"))
    # cash 100 - 10 + 1 = 91; remaining 9e-12 shares x 1e12 = 9; no gain so no tax
    assert book.liquidation_value({"X": 1e12}, pd.Timestamp("2025-03-01")) == pytest.approx(100.0)


def test_fractional_forward_matches_finalize_as_of_last_match() -> None:
    book = TaxLotBook(TaxParams(), cash=10.0)
    book.buy("XYZ", 0.3, 1.0, pd.Timestamp("2025-01-01"))
    loss = book.sell("XYZ", 0.3, 0.7, pd.Timestamp("2025-03-01"))[0]
    for day in range(2, 12):
        book.buy("XYZ", 0.03, 1.0, pd.Timestamp("2025-03-01") + pd.Timedelta(days=day - 1))
    assert book.loss_status(loss, pd.Timestamp("2025-03-10")) == "provisional"
    assert book.loss_status(loss, pd.Timestamp("2025-03-11")) == "final"
    adjustments = [e for e in book.events if e.kind == "wash_adjustment"]
    assert sum(e.payload["disallowed"] for e in adjustments) == pytest.approx(0.09)
    assert loss.disallowed == 0.0 and loss.status == "provisional"
    assert book.verify_chain()


def test_tiny_loss_matches_large_replacement_lot() -> None:
    book = TaxLotBook(TaxParams(), cash=2_000_001.0)
    book.buy("XYZ", 0.000001, 100.0, pd.Timestamp("2025-01-01"))
    loss = book.sell("XYZ", 0.000001, 80.0, pd.Timestamp("2025-03-01"))[0]
    book.buy("XYZ", 2_000_000.0, 1.0, pd.Timestamp("2025-03-02"))
    adjustments = [e for e in book.events if e.kind == "wash_adjustment"]
    assert len(adjustments) == 1
    assert adjustments[0].payload["qty"] == pytest.approx(0.000001)
    assert adjustments[0].payload["disallowed"] == pytest.approx(0.00002)
    assert book.loss_status(loss, pd.Timestamp("2025-03-02")) == "final"
    sales = book.sell("XYZ", 2_000_000.0, 2.0, pd.Timestamp("2025-04-15"))
    assert len(sales) == 2
    assert sales[0].qty == pytest.approx(0.000001, rel=0, abs=1e-15)
    assert sales[0].basis == pytest.approx(0.000021, rel=0, abs=1e-15)
    assert sales[1].qty == pytest.approx(1_999_999.999999, rel=0, abs=1e-9)
    assert sales[1].basis == pytest.approx(1_999_999.999999, rel=0, abs=1e-9)
    assert book.verify_chain()


def test_loss_status_as_of_split_before_forward_matches() -> None:
    book = TaxLotBook(TaxParams(), cash=1000.0)
    book.buy("XYZ", 4.0, 10.0, pd.Timestamp("2025-01-01"))
    loss = book.sell("XYZ", 4.0, 8.0, pd.Timestamp("2025-03-01"))[0]
    book.split("XYZ", 2.0, pd.Timestamp("2025-03-02"))
    book.buy("XYZ", 4.0, 4.0, pd.Timestamp("2025-03-03"))
    book.buy("XYZ", 4.0, 4.0, pd.Timestamp("2025-03-04"))
    assert book.loss_status(loss, pd.Timestamp("2025-03-02")) == "provisional"
    assert book.loss_status(loss, pd.Timestamp("2025-03-03")) == "provisional"
    assert book.loss_status(loss, pd.Timestamp("2025-03-04")) == "final"
    assert book.verify_chain()
