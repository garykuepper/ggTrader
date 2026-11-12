"""Calculate listed share-class value from caller-verified as-traded inputs."""

import math
from datetime import date
from numbers import Real

from filing_vintages import _utc


def calculate(rows: list[dict], expected_classes: set[str], decision_at: object) -> float:
    """Sum listed classes using contemporaneous-basis prices and available share counts.

    The caller must verify the issuer, complete listed-class set, USD units, price
    basis, timestamps and suitable freshness. A label is not proof. Unlisted classes
    are excluded, so the result is not necessarily the paper's firm market equity.
    Historical split-adjusted prices must be independently reconciled before use.
    An as-traded price also needs a share count on the same split basis: callers must
    account for splits and class changes between the share date and the price date.
    The cutoff is inclusive; callers must allow for publication and execution latency.
    """
    if not isinstance(rows, list) or not isinstance(expected_classes, set) or not expected_classes:
        raise ValueError("Invalid expected classes")
    if not all(isinstance(x, str) and x and x == x.strip() for x in expected_classes):
        raise ValueError("Invalid expected classes")
    decision = _utc(decision_at)
    values = []
    found = set()
    price_times = set()
    for row in rows:
        try:
            symbol = row["security"]
            if symbol not in expected_classes or symbol in found:
                raise ValueError("Missing, unexpected or duplicate classes")
            if row["price_basis"] != "as_traded":
                raise ValueError("Price basis must be as_traded")
            price_at = _utc(row["price_at"])
            available_at = _utc(row["shares_available_at"])
            shares_asof = date.fromisoformat(row["shares_asof"])
            if shares_asof.isoformat() != row["shares_asof"]:
                raise ValueError("Invalid share date")
            if available_at.tz_convert("America/New_York").date() < shares_asof:
                raise ValueError("Availability before share date")
            if available_at > decision or price_at > decision:
                raise ValueError("Inputs not yet available")
            if shares_asof > price_at.tz_convert("America/New_York").date():
                raise ValueError("Share date follows price date")
            shares, price = row["shares"], row["price"]
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
                for value in (shares, price)
            ):
                raise ValueError("Invalid price or share count")
            value = float(shares) * float(price)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid market value")
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid market-value input") from exc
        found.add(symbol)
        price_times.add(price_at)
        values.append(value)
    if found != expected_classes:
        raise ValueError("Missing expected classes")
    if len(price_times) != 1:
        raise ValueError("Prices must share a timestamp")
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("Invalid market value") from exc
    if not math.isfinite(total):
        raise ValueError("Invalid market value")
    return total
