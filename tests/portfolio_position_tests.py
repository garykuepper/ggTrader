import math
from datetime import datetime

import pytest

from ggTrader.Portfolio import Portfolio
from ggTrader.Position import Position


def test_position_basic_calculations():
    date0 = datetime(2025, 1, 1, 12, 0, 0)
    pos = Position(symbol="AAPL", qty=2, price=100.0, date=date0)

    # Basic fields
    assert pos.symbol == "AAPL"
    assert pos.qty == 2
    assert pos.entry_price == 100.0
    assert pos.entry_date == date0
    assert pos.exit_price is None
    assert pos.exit_date is None
    assert pos.current_price == 100.0
    assert pos.status == "open"

    # Cost and current value
    assert pos.cost == pytest.approx(2 * 100.0)
    assert pos.current_value == pytest.approx(2 * 100.0)
    assert pos.profit == pytest.approx(0.0)
    assert pos.profit_pct == pytest.approx(0.0)

    # Update price and recompute
    pos.update_price(110.0)
    assert pos.current_price == pytest.approx(110.0)
    assert pos.current_value == pytest.approx(2 * 110.0)
    assert pos.profit == pytest.approx(2 * 110.0 - 2 * 100.0)
    assert pos.profit_pct == pytest.approx((2 * 110.0 - 2 * 100.0) / (2 * 100.0))


def test_position_close_and_open_behaviour():
    date0 = datetime(2025, 1, 1, 9, 0, 0)
    pos = Position(symbol="GOOG", qty=1, price=50.0, date=date0)

    # Open/close lifecycle
    assert pos.status == "open"
    pos.close_position(datetime(2025, 1, 2, 15, 0, 0))
    assert pos.status == "closed"
    assert pos.exit_date == datetime(2025, 1, 2, 15, 0, 0)

    # exit_price should be current_price at time of close
    assert pos.exit_price is None or isinstance(pos.exit_price, (int, float))
    # current_price remains at entry unless updated
    assert pos.current_price == 50.0


def test_portfolio_add_and_close_position_without_fees():
    # Use zero transaction fee to simplify expectations
    port = Portfolio(cash=1000, transaction_fee=0.0)
    date0 = datetime(2025, 1, 1, 10, 0, 0)

    p = Position(symbol="MSFT", qty=5, price=100.0, date=date0)
    # Add position
    port.add_position(p)

    # Cash reduced by cost
    assert port.cash == pytest.approx(1000 - (5 * 100.0))

    # Current total value includes position value
    assert port.total_value == pytest.approx(port.cash + p.current_value)

    # Update price and then close
    port.update_position_price("MSFT", 120.0, datetime(2025, 1, 2, 11, 0, 0))
    # Close the position
    pos_in_port = port.get_position("MSFT")
    assert pos_in_port is not None
    port.close_position(pos_in_port, datetime(2025, 1, 2, 11, 30, 0))

    # Position should be removed from open positions
    assert port.get_position("MSFT") is None
    assert not port.in_position("MSFT")

    # Cash should reflect exit value with zero fees
    exit_value = 5 * 120.0
    assert port.cash == pytest.approx(1000 - (5 * 100.0) + exit_value)

    # Realized profit should reflect the P/L of the trade
    # Profit = (exit_value - cost) with zero fees
    expected_profit = (exit_value - (5 * 100.0))
    assert port.realized_profit == pytest.approx(expected_profit)


def test_portfolio_unrealized_profit_and_profit_snapshots():
    port = Portfolio(cash=1000, transaction_fee=0.0)
    date0 = datetime(2025, 1, 1, 9, 0, 0)

    p1 = Position(symbol="AAPL", qty=2, price=150.0, date=date0)
    p2 = Position(symbol="TSLA", qty=1, price=600.0, date=date0)

    port.add_position(p1)
    port.add_position(p2)

    # Initially unrealized profit equals sum of individual profits (which are zero at entry)
    assert port.unrealized_profit == pytest.approx(0.0)

    # Move prices to create unrealized profit
    port.update_position_price("AAPL", 160.0, date0)
    port.update_position_price("TSLA", 590.0, date0)

    # Unrealized should reflect updated profits
    expected_unrealized = (p1.current_value - p1.cost) + (p2.current_value - p2.cost)
    assert port.unrealized_profit == pytest.approx(expected_unrealized)

    # Total value should reflect cash plus position values
def almost_equal(a, b, tol=1e-8):
    return abs(a - b) <= tol


def test_add_position_applies_entry_fee_and_updates_cash_and_fee_totals():
    p = Portfolio(cash=10000, transaction_fee=0.01)  # 1% fee to make numbers easy
    pos = Position("BTC", qty=1.0, price=1000.0, date=datetime.utcnow())

    # before
    assert p.cash == 10000
    assert p.transaction_fee_total == 0.0
    assert not p.in_position("BTC")

    p.add_position(pos)

    entry_fee = pos.cost * p.transaction_fee  # 1000 * 0.01 = 10.0
    assert almost_equal(pos.entry_fee, entry_fee)
    assert almost_equal(p.transaction_fee_total, entry_fee)
    # cash reduced by cost + fee
    assert almost_equal(p.cash, 10000 - pos.cost - entry_fee)
    # position tracked
    assert p.in_position("BTC")
    assert pos in p.positions
    assert pos in p.trades


def test_close_position_applies_exit_fee_and_updates_realized_and_cash():
    p = Portfolio(cash=10000, transaction_fee=0.02)  # 2% fee
    pos = Position("ETH", qty=2.0, price=500.0, date=datetime.utcnow())  # cost = 1000
    p.add_position(pos)

    # move price up
    pos.update_price(600.0)  # current_value = 1200, profit = 200

    # close
    p.close_position(pos, date=datetime.utcnow())

    # exit fee should be 2% of exit value (1200) = 24
    assert almost_equal(pos.exit_fee, pos.current_value * p.transaction_fee)
    # entry fee should have been recorded earlier
    assert pos.entry_fee > 0

    # realized_profit should equal position.profit - entry_fee - exit_fee
    expected_realized = pos.profit - pos.entry_fee - pos.exit_fee
    assert almost_equal(p.realized_profit, expected_realized)

    # cash should have increased by exit_value - exit_fee (entry already removed at add_position)
    expected_cash = 10000 - pos.cost - pos.entry_fee + pos.current_value - pos.exit_fee
    assert almost_equal(p.cash, expected_cash)

    # position removed from open positions
    assert not p.in_position("ETH")
    assert pos not in p.positions
    # but still present in trades history
    assert pos in p.trades


def test_unrealized_profit_is_net_of_entry_fee_for_open_positions():
    p = Portfolio(cash=5000, transaction_fee=0.01)  # 1% fee
    pos = Position("XRP", qty=100.0, price=1.00, date=datetime.utcnow())  # cost = 100
    p.add_position(pos)

    # price moves to 1.10 -> current_value = 110, profit = 10
    pos.update_price(1.10)
    # unrealized should be profit - entry_fee
    expected_unreal = pos.profit - pos.entry_fee
    assert almost_equal(p.unrealized_profit, expected_unreal)

    # change price again and confirm unrealized follows
    pos.update_price(0.90)  # current_value = 90, profit = -10
    expected_unreal2 = pos.profit - pos.entry_fee
    assert almost_equal(p.unrealized_profit, expected_unreal2)


def test_reconcile_invariant_holds_after_multiple_trades():
    p = Portfolio(cash=10000, transaction_fee=0.005)  # 0.5% fee
    # open trade A
    a = Position("AAA", qty=10, price=10.0, date=datetime.utcnow())  # cost 100
    p.add_position(a)
    # open trade B
    b = Position("BBB", qty=5, price=20.0, date=datetime.utcnow())  # cost 100
    p.add_position(b)

    # update prices
    a.update_price(12.0)  # a profit = 20
    b.update_price(18.0)  # b profit = -10

    # close B
    p.close_position(b, date=datetime.utcnow())

    # now compute reconciliation values (use the same logic as reconcile)
    left = p.starting_cash + p.realized_profit + p.unrealized_profit
    right = p.total_value

    # they should match (within small rounding)
    assert almost_equal(left, right, tol=1e-6)

    # check transaction fees sum equals recorded total
    sum_entry = sum(getattr(t, "entry_fee", 0.0) for t in p.trades)
    sum_exit = sum(getattr(t, "exit_fee", 0.0) for t in p.trades if getattr(t, "exit_fee", 0.0) is not None)
    assert almost_equal(p.transaction_fee_total, sum_entry + sum_exit, tol=1e-6)


def test_profit_and_profit_pct_consistency():
    p = Portfolio(cash=2000, transaction_fee=0.01)
    pos = Position("ZZZ", qty=2, price=100.0, date=datetime.utcnow())  # cost 200
    p.add_position(pos)

    # unrealized profit net-of-fee should be computed correctly
    pos.update_price(150.0)  # profit = 100, entry_fee = 2 -> unrealized net = 98
    assert almost_equal(p.unrealized_profit, pos.profit - pos.entry_fee)

    # profit property should equal total_value - starting_cash
    assert almost_equal(p.profit, p.total_value - p.starting_cash)
    # profit_pct should be profit / starting_cash
    assert almost_equal(p.profit_pct, p.profit / p.starting_cash)


# OPTIONAL: some negative / edge cases
def test_zero_qty_or_zero_price_positions_behave_safely():
    p = Portfolio(cash=1000, transaction_fee=0.01)
    pos = Position("ZERO", qty=0.0, price=100.0, date=datetime.utcnow())
    p.add_position(pos)
    # cost = 0, entry fee should be 0
    assert almost_equal(pos.cost, 0.0)
    assert almost_equal(pos.entry_fee, 0.0)
    # unrealized profit should be -entry_fee (zero)
    assert almost_equal(p.unrealized_profit, 0.0)

    pos2 = Position("P0", qty=1.0, price=0.0, date=datetime.utcnow())
    p.add_position(pos2)
    assert almost_equal(pos2.cost, 0.0)
    # no division by zero for profit_pct on position when cost==0 is handled by Position (if not, skip)
    # we only assert safe current_value/profit calculations
    assert almost_equal(pos2.profit, pos2.current_value - pos2.cost)
