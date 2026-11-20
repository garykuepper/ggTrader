"""Late-bound collaborators used by a paper-trading run."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from ggTrader.paper.models import UUID, OrderStatus, SignalBundle, SubmittedOrder


@dataclass(frozen=True)
class RunHooks:
    """External functions resolved when called."""

    today_et: Callable[[], date]
    now_pt: Callable[[], datetime]
    utcnow: Callable[[], datetime]
    new_uuid: Callable[[], UUID]
    clock_time: Callable[[], float]
    sleep: Callable[[float], None]
    generate_signals: Callable[..., SignalBundle]
    membership_source: Callable[[str | None], object]
    refresh_tape: Callable[[], list[str]]
    find_split_applied: Callable[[dict, list, dict, date], tuple[set[str], set[str]]]


@dataclass(frozen=True)
class TraderSeams:
    """Private trader methods resolved on their instance when called."""

    submit_order: Callable[[str, str, float, str | None], str]
    poll_orders: Callable[[list[SubmittedOrder]], dict[str, OrderStatus]]
    missed_exits: Callable[[dict, dict[str, str], set[str]], list[str]]
    split_corrections: Callable[[dict, date, date], dict[str, float]]
    accrue_dividends: Callable[[dict], dict]
