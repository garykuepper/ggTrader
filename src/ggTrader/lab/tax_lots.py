"""Account for standalone taxable lots with immutable, hash-chained audit events."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import pandas as pd

WASH_WINDOW = pd.Timedelta(days=30)
GENESIS_HASH = "0" * 64
REL_TOL = 1e-12  # relative float tolerance for share quantities and funding checks
PAYMENT_TOLERANCE = 1e-10  # absolute dollars


def _exhausted(value: float, reference: float) -> bool:
    """Return True when ``value`` is zero or float dust relative to ``reference``."""
    return value <= REL_TOL * abs(reference)


@dataclass(frozen=True)
class TaxParams:
    """Specify the registered standalone tax rates."""

    st_rate: float = 0.24
    lt_rate: float = 0.15

    def __post_init__(self) -> None:
        """Reject invalid marginal rates."""
        for rate in (self.st_rate, self.lt_rate):
            if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError("Tax rates must be finite and between zero and one")


@dataclass(frozen=True)
class Realization:
    """Preserve a sale-time realization; later washes are separate adjustments."""

    lot_id: str
    symbol: str
    qty: float
    proceeds: float
    basis: float
    gain: float
    term: str
    status: str
    disallowed: float
    # Legacy nine-field comparisons describe economics, not the sale timestamp.
    date: pd.Timestamp = field(default=pd.NaT, compare=False)


class _FrozenPayload(dict):
    """Keep JSON-compatible event contents immutable, including nested values."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        """Reject mutation through the standard mapping interface."""
        raise TypeError("Ledger payloads are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenPayload:
        """Share immutable payloads when copying an event or a book."""
        return self


def _freeze(value: Any) -> Any:
    """Convert an event value to immutable JSON-compatible contents."""
    if isinstance(value, Mapping):
        return _FrozenPayload({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class LedgerEvent:
    """Record one immutable entry in the book's SHA-256 chain."""

    seq: int
    kind: str
    date: pd.Timestamp
    payload: Mapping[str, Any]
    prev_hash: str
    hash: str

    def __post_init__(self) -> None:
        """Freeze nested payloads without depending on caller-owned mappings."""
        object.__setattr__(self, "payload", _freeze(self.payload))


def _event_hash(
    seq: int, kind: str, date: pd.Timestamp, payload: Mapping[str, Any], prev_hash: str
) -> str:
    """Hash deterministic JSON containing every event field except its own hash."""
    encoded = json.dumps(
        {
            "seq": seq,
            "kind": kind,
            "date": date.isoformat(),
            "payload": _freeze(payload),
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_events(events: Iterable[LedgerEvent]) -> bool:
    """Check sequence, linkage and contents starting at the genesis hash."""
    previous = GENESIS_HASH
    try:
        for seq, event in enumerate(events, start=1):
            if event.seq != seq or event.prev_hash != previous:
                return False
            if event.hash != _event_hash(
                event.seq, event.kind, event.date, event.payload, event.prev_hash
            ):
                return False
            previous = event.hash
    except (AttributeError, TypeError, ValueError):
        return False
    return True


class InsufficientCash(ValueError):
    """Indicate that a requested cash debit needs a separately funded fill."""


@dataclass
class _Lot:
    """Track a homogeneous slice of a purchase, including wash matching capacity."""

    lot_id: str
    symbol: str
    qty: float
    per_share: float
    acquired: pd.Timestamp
    purchased: pd.Timestamp
    replacement_used: bool = False


@dataclass
class _Sale:
    """Track mutable wash facts separately from the original realization."""

    realization: Realization
    symbol: str
    held: pd.Timedelta
    ref_seq: int
    disallowed: float = 0.0
    matched_qty: float = 0.0
    split_factor: float = 1.0

    @property
    def remaining(self) -> float:
        """Return unmatched sale quantity in current corporate-action units."""
        return (self.realization.qty - self.matched_qty) * self.split_factor


@dataclass(frozen=True)
class _TaxResult:
    """Keep annual tax components and character-preserving carryforwards."""

    st_tax: float
    lt_tax: float
    dividend_tax: float
    interest_tax: float
    cf_st: float
    cf_lt: float

    @property
    def total(self) -> float:
        """Return tax due without assigning value to unused losses."""
        return self.st_tax + self.lt_tax + self.dividend_tax + self.interest_tax


def _nonnegative(value: float, name: str) -> float:
    """Validate finite, nonnegative monetary and quantity inputs."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _date(value: pd.Timestamp) -> pd.Timestamp:
    """Require a valid transaction timestamp."""
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("Transaction date must not be NaT")
    return date


def _term(acquired: pd.Timestamp, date: pd.Timestamp) -> str:
    """Classify a holding as LT only strictly after its calendar anniversary."""
    return "LT" if date > acquired + pd.DateOffset(years=1) else "ST"


class TaxLotBook:
    """Maintain cash, HIFO lots, wash adjustments and annual standalone taxes.

    Fills and income must be supplied in economic order. Execution timing, costs,
    distribution classification and outside-account activity belong to the caller.
    """

    def __init__(self, params: TaxParams, cash: float = 0.0) -> None:
        """Initialize an unfunded or explicitly funded standalone book."""
        self.params = params
        self.cash = _nonnegative(cash, "Cash")
        self._lots: list[_Lot] = []
        self._sales: list[_Sale] = []
        self._income: list[tuple[pd.Timestamp, str, float]] = []
        self._events: list[LedgerEvent] = []
        self._next_lot = 1
        self._original_assessments: dict[int, float] = {}
        self._assessments: dict[int, float] = {}
        self._assessment_seqs: dict[int, int] = {}
        self._paid = 0.0

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """Expose the append-only history without exposing its mutable container."""
        return tuple(self._events)

    @property
    def unpaid_tax(self) -> float:
        """Return globally assessed tax less all book-funded payments."""
        return max(0.0, sum(self._assessments.values()) - self._paid)

    @property
    def cf_st(self) -> float:
        """Return the ST loss carryforward at the last assessed year, as a loss."""
        return self._taxes()[max(self._assessments)].cf_st if self._assessments else 0.0

    @property
    def cf_lt(self) -> float:
        """Return the LT loss carryforward at the last assessed year, as a loss."""
        return self._taxes()[max(self._assessments)].cf_lt if self._assessments else 0.0

    def _append(self, kind: str, date: pd.Timestamp, payload: Mapping[str, Any]) -> int:
        """Append a self-contained event and return its sequence number."""
        seq = len(self._events) + 1
        previous = self._events[-1].hash if self._events else GENESIS_HASH
        digest = _event_hash(seq, kind, date, payload, previous)
        self._events.append(LedgerEvent(seq, kind, date, payload, previous, digest))
        return seq

    def verify_chain(self) -> bool:
        """Verify every event from the start of this book's ledger."""
        return verify_events(self.events)

    def buy(self, symbol: str, qty: float, price: float, date: pd.Timestamp) -> str:
        """Debit a fill and match remaining forward-window losses oldest first."""
        qty, price, date = self._fill_inputs(qty, price, date)
        if not symbol:
            raise ValueError("Symbol must be nonempty")
        cost = qty * price
        if cost > self.cash:
            if not math.isclose(cost, self.cash, rel_tol=REL_TOL, abs_tol=PAYMENT_TOLERANCE * 1e-2):
                raise InsufficientCash("Purchase exceeds book cash")
            cost = self.cash  # roundoff only: snap to the available cash
        lot_id = f"L{self._next_lot:06d}"
        self._next_lot += 1
        lot = _Lot(lot_id, symbol, qty, price, date, date)
        self._lots.append(lot)
        self.cash -= cost
        self._append("buy", date, {"lot_id": lot_id, "symbol": symbol, "qty": qty, "price": price})
        for sale in self._sales:
            if (
                sale.symbol == symbol
                and sale.realization.gain < 0.0
                and pd.Timedelta(0) <= date - sale.realization.date <= WASH_WINDOW
                and not _exhausted(sale.remaining, sale.realization.qty * sale.split_factor)
            ):
                for replacement in list(self._lots):
                    if replacement.lot_id == lot_id and not replacement.replacement_used:
                        self._match(sale, replacement, date)
        self._refresh_assessments(date)
        return lot_id

    @staticmethod
    def _fill_inputs(
        qty: float, price: float, date: pd.Timestamp
    ) -> tuple[float, float, pd.Timestamp]:
        """Validate a fill before any book mutation."""
        qty = _nonnegative(qty, "Quantity")
        if qty == 0.0:
            raise ValueError("Quantity must be positive")
        return qty, _nonnegative(price, "Price"), _date(date)

    def _match(self, sale: _Sale, lot: _Lot, date: pd.Timestamp) -> None:
        """Assign one replacement slice exactly once and append its correction."""
        matched = min(sale.remaining, lot.qty)
        if _exhausted(matched, lot.qty):
            return
        if matched < lot.qty:
            remainder = replace(lot, qty=lot.qty - matched)
            self._lots.append(remainder)
            lot.qty = matched
        sold_units = matched / sale.split_factor
        disallowed = -sale.realization.gain * sold_units / sale.realization.qty
        lot.per_share += disallowed / matched
        lot.acquired -= sale.held
        lot.replacement_used = True
        sale.matched_qty += sold_units
        sale.disallowed += disallowed
        self._append(
            "wash_adjustment",
            date,
            {
                "ref_seq": sale.ref_seq,
                "sold_lot_id": sale.realization.lot_id,
                "replacement_lot_id": lot.lot_id,
                "qty": matched,
                "disallowed": disallowed,
                "acquired": lot.acquired,
                "basis_per_share": lot.per_share,
            },
        )

    def sell(self, symbol: str, qty: float, price: float, date: pd.Timestamp) -> list[Realization]:
        """Sell HIFO slices, excluding every participating lot from replacements."""
        qty, price, date = self._fill_inputs(qty, price, date)
        lots = sorted(
            (lot for lot in self._lots if lot.symbol == symbol),
            key=lambda lot: (-lot.per_share, lot.acquired, lot.lot_id),
        )
        held = sum(lot.qty for lot in lots)
        if not lots or (qty > held and not math.isclose(qty, held, rel_tol=REL_TOL)):
            raise ValueError(f"Cannot sell {qty} shares of {symbol}: insufficient holdings")
        qty = min(qty, held)  # snap float roundoff to the exact holdings
        selected: list[tuple[_Lot, float]] = []
        remaining = qty
        for lot in lots:
            used = min(lot.qty, remaining)
            selected.append((lot, used))
            remaining -= used
            if _exhausted(remaining, qty):
                break
        excluded = {lot.lot_id for lot, _ in selected}
        realized = []
        # Selection is frozen before any replacement basis changes.
        for lot, used in selected:
            basis = used * lot.per_share
            proceeds = used * price
            gain = proceeds - basis
            realization = Realization(
                lot.lot_id,
                symbol,
                used,
                proceeds,
                basis,
                gain,
                _term(lot.acquired, date),
                "provisional" if gain < 0.0 else "final",
                0.0,
                date,
            )
            seq = self._append("sell", date, asdict(realization))
            sale = _Sale(realization, symbol, date - lot.acquired, seq)
            self._sales.append(sale)
            lot.qty -= used
            self.cash += proceeds
            if gain < 0.0:
                replacements = sorted(
                    (
                        candidate
                        for candidate in self._lots
                        if candidate.symbol == symbol
                        and candidate.lot_id not in excluded
                        and not candidate.replacement_used
                        and candidate.qty > 0.0
                        and pd.Timedelta(0) <= date - candidate.purchased <= WASH_WINDOW
                    ),
                    key=lambda candidate: (candidate.purchased, candidate.lot_id),
                )
                for candidate in replacements:
                    self._match(sale, candidate, date)
                exhausted = _exhausted(sale.remaining, sale.realization.qty * sale.split_factor)
                status = "final" if exhausted else "provisional"
                realization = replace(realization, disallowed=sale.disallowed, status=status)
                sale.realization = realization
            realized.append(realization)
        self._lots = [lot for lot in self._lots if lot.qty > 0.0]
        self._refresh_assessments(date)
        return realized

    def loss_status(self, realization: Realization, asof: pd.Timestamp) -> str:
        """Report current finality without rewriting the frozen sale-time object."""
        asof = _date(asof)
        if realization.gain >= 0.0 or asof - realization.date > WASH_WINDOW:
            return "final"
        disallowed = realization.disallowed
        for sale in self._sales:
            if sale.realization is realization:
                disallowed = sum(
                    float(event.payload["disallowed"])
                    for event in self._events
                    if event.kind == "wash_adjustment"
                    and event.payload["ref_seq"] == sale.ref_seq
                    and event.date <= asof
                )
                break
        return "final" if disallowed >= -realization.gain else "provisional"

    def split(self, symbol: str, ratio: float, date: pd.Timestamp) -> None:
        """Scale holdings and open replacement quantities without changing basis."""
        ratio = _nonnegative(ratio, "Split ratio")
        date = _date(date)
        if ratio == 0.0:
            raise ValueError("Split ratio must be positive")
        for lot in self._lots:
            if lot.symbol == symbol:
                lot.qty *= ratio
                lot.per_share /= ratio
        for sale in self._sales:
            if sale.symbol == symbol and date - sale.realization.date <= WASH_WINDOW:
                sale.split_factor *= ratio
        self._append("split", date, {"symbol": symbol, "ratio": ratio})

    def rename(self, old: str, new: str, date: pd.Timestamp) -> None:
        """Move symbol identity, including open washes, without resetting dates."""
        date = _date(date)
        if not new:
            raise ValueError("Replacement symbol must be nonempty")
        for lot in self._lots:
            if lot.symbol == old:
                lot.symbol = new
        for sale in self._sales:
            if sale.symbol == old:
                sale.symbol = new
        self._append("rename", date, {"old": old, "new": new})

    def dividend(self, symbol: str, per_share: float, date: pd.Timestamp, qualified: bool) -> float:
        """Credit the per-share distribution using the caller's classification."""
        per_share = _nonnegative(per_share, "Dividend per share")
        date = _date(date)
        amount = per_share * sum(lot.qty for lot in self._lots if lot.symbol == symbol)
        kind = "qualified_dividend" if qualified else "unqualified_dividend"
        self._income.append((date, kind, amount))
        self.cash += amount
        self._append(kind, date, {"symbol": symbol, "per_share": per_share, "amount": amount})
        self._refresh_assessments(date)
        return amount

    def interest(self, amount: float, date: pd.Timestamp) -> None:
        """Credit cash interest, taxable separately from capital losses."""
        amount, date = _nonnegative(amount, "Interest"), _date(date)
        self._income.append((date, "interest", amount))
        self.cash += amount
        self._append("interest", date, {"amount": amount})
        self._refresh_assessments(date)

    def _taxes(
        self, extra: Mapping[int, tuple[float, float]] | None = None
    ) -> dict[int, _TaxResult]:
        """Recompute annual netting in order so amendments also update carryforwards."""
        buckets: dict[int, dict[str, float]] = {}
        for sale in self._sales:
            item = sale.realization
            annual = buckets.setdefault(item.date.year, {})
            annual[item.term] = annual.get(item.term, 0.0) + item.gain + sale.disallowed
        for date, kind, amount in self._income:
            annual = buckets.setdefault(date.year, {})
            annual[kind] = annual.get(kind, 0.0) + amount
        for year, (st, lt) in (extra or {}).items():
            annual = buckets.setdefault(year, {})
            annual["ST"] = annual.get("ST", 0.0) + st
            annual["LT"] = annual.get("LT", 0.0) + lt
        results = {}
        cf_st = cf_lt = 0.0
        for year in sorted(buckets.keys() | self._assessments.keys()):
            annual = buckets.get(year, {})
            st, lt = annual.get("ST", 0.0) + cf_st, annual.get("LT", 0.0) + cf_lt
            if st < 0.0 < lt:
                offset = min(-st, lt)
                st, lt = st + offset, lt - offset
            elif lt < 0.0 < st:
                offset = min(-lt, st)
                st, lt = st - offset, lt + offset
            cf_st, cf_lt = min(st, 0.0), min(lt, 0.0)
            results[year] = _TaxResult(
                max(st, 0.0) * self.params.st_rate,
                max(lt, 0.0) * self.params.lt_rate,
                annual.get("qualified_dividend", 0.0) * self.params.lt_rate
                + annual.get("unqualified_dividend", 0.0) * self.params.st_rate,
                annual.get("interest", 0.0) * self.params.st_rate,
                cf_st,
                cf_lt,
            )
        return results

    def _refresh_assessments(self, date: pd.Timestamp) -> None:
        """Append corrections when new facts change an already assessed year."""
        results = self._taxes()
        for year, previous in list(self._assessments.items()):
            current = results[year].total
            if current != previous:
                self._assessments[year] = current
                self._append(
                    "tax_adjustment",
                    date,
                    {
                        "ref_seq": self._assessment_seqs[year],
                        "year": year,
                        "amount": current - previous,
                        "revised_tax": current,
                        **asdict(results[year]),
                    },
                )

    def year_end(self, year: int, asof: pd.Timestamp) -> float:
        """Assess annual tax once, returning the revised total after any corrections."""
        asof = _date(asof)
        if year > asof.year:
            raise ValueError("Cannot assess a future tax year")
        if year not in self._assessments:
            # Include an empty year so its incoming carryforward is still recorded.
            result = self._taxes({year: (0.0, 0.0)})[year]
            self._assessments[year] = result.total
            self._original_assessments[year] = result.total
            self._assessment_seqs[year] = self._append(
                "tax_assessment", asof, {"year": year, "amount": result.total, **asdict(result)}
            )
        return self._assessments[year]

    def amended_tax(self, year: int) -> float:
        """Return the cumulative revision relative to the original year assessment."""
        return self._assessments.get(year, 0.0) - self._original_assessments.get(year, 0.0)

    def pay_tax(self, amount: float, date: pd.Timestamp) -> None:
        """Pay a global assessed balance using only existing book cash."""
        amount, date = _nonnegative(amount, "Tax payment"), _date(date)
        if amount > self.cash:
            raise InsufficientCash("Tax payment exceeds book cash; provide a separate sale fill")
        if amount > self.unpaid_tax + PAYMENT_TOLERANCE:
            raise ValueError("Tax payment exceeds assessed unpaid tax")
        self.cash -= amount
        self._paid += amount
        self._append("tax_payment", date, {"amount": amount})

    def liquidation_value(self, prices: Mapping[str, float], date: pd.Timestamp) -> float:
        """Value a hypothetical full liquidation net of accrued and unrealized tax."""
        date = _date(date)
        st = lt = market_value = 0.0
        for lot in self._lots:
            value = lot.qty * _nonnegative(prices[lot.symbol], f"Price for {lot.symbol}")
            gain = value - lot.qty * lot.per_share
            market_value += value
            if _term(lot.acquired, date) == "ST":
                st += gain
            else:
                lt += gain
        taxes = self._taxes({date.year: (st, lt)})
        unpaid = max(0.0, sum(result.total for result in taxes.values()) - self._paid)
        return self.cash + market_value - unpaid

    def terminal_liquidate(self, prices: Mapping[str, float], date: pd.Timestamp) -> float:
        """Sell every holding, assess outstanding years and pay remaining tax once."""
        date = _date(date)
        symbols = sorted({lot.symbol for lot in self._lots})
        # Validate every price before executing any terminal fill.
        fills = {symbol: _nonnegative(prices[symbol], f"Price for {symbol}") for symbol in symbols}
        for symbol in symbols:
            qty = sum(lot.qty for lot in self._lots if lot.symbol == symbol)
            self.sell(symbol, qty, fills[symbol], date)
        for year in sorted(self._taxes({date.year: (0.0, 0.0)})):
            self.year_end(year, date)
        if self.unpaid_tax > 0.0:
            self.pay_tax(self.unpaid_tax, date)
        return self.cash
