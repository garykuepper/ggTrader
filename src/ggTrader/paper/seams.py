"""Late-bound collaborators used by a paper-trading run."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from ggTrader.paper.models import UUID, SignalBundle


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
