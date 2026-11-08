"""An in-memory ``MembershipStore`` for tests: the DB source without psycopg2 or a server."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from ggTrader.data.membership import MembershipUnavailable, StoredVintage
from ggTrader.data.universe_algebra import (
    History,
    collapse,
    content_sha256,
    history_stats,
    raw_sha256,
)

BASE_TIME = datetime(2026, 6, 9, tzinfo=timezone.utc)


class FakeMembershipStore:
    """Keeps whole vintages in memory and serves them the way ``PgMembershipStore`` does."""

    def __init__(self) -> None:
        self.vintages: dict[int, StoredVintage] = {}
        self.read_calls: list[int | None] = []

    def add(
        self,
        history: History,
        *,
        retrieved_at: datetime = BASE_TIME,
        verified_at: datetime | None = None,
        index_name: str = "sp500",
    ) -> int:
        """Store ``history`` as the next vintage and return its load id."""
        load_id = max(self.vintages, default=0) + 1
        intervals = collapse(history)
        stats = history_stats(history)
        load = {
            "load_id": load_id,
            "index_name": index_name,
            "retrieved_at": retrieved_at,
            "raw_sha256": raw_sha256(f"fake-raw-{load_id}".encode()),
            "content_sha256": content_sha256(history),
            "n_intervals": len(intervals),
            **stats,
        }
        dates = {day: len(tickers) for day, tickers in history.items()}
        self.vintages[load_id] = StoredVintage(load, dates, intervals, verified_at or retrieved_at)
        return load_id

    def read(self, load_id: int | None, index_name: str) -> StoredVintage:
        """The vintage ``load_id`` (the newest when None), else ``MembershipUnavailable``."""
        self.read_calls.append(load_id)
        wanted = max(self.vintages, default=None) if load_id is None else load_id
        if wanted is None or wanted not in self.vintages:
            raise MembershipUnavailable(f"no load {wanted} for index {index_name!r}")
        return self.vintages[wanted]


class FailingStore:
    """A store whose database cannot be reached."""

    def read(self, load_id: int | None, index_name: str) -> StoredVintage:
        """Always raises, as an unreachable database would."""
        raise MembershipUnavailable("cannot read membership from the database (test)")


def synthetic_history(seed: int = 0, n_dates: int = 30, size: int = 485) -> History:
    """A valid random history (symbols leave and rejoin) for unit tests."""
    import random

    rng = random.Random(seed)
    pool = [f"T{n:04d}" for n in range(size + 60)]
    members = set(rng.sample(pool, size))
    out: History = {}
    day = date(2000, 1, 3)
    for _ in range(n_dates):
        for old in rng.sample(sorted(members), 3):
            members.discard(old)
        for new in rng.sample(sorted(set(pool) - members), 3):
            members.add(new)
        out[day] = tuple(sorted(members))
        day += timedelta(days=rng.randint(1, 40))
    return out
