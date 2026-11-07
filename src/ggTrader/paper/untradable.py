"""Symbols the trader must never try to trade (OD17): configured set, broker status, 422s.

AVB merged into EQR (2026-08-17) and the broker marks it inactive, so every order for it is
rejected with HTTP 422 ``asset AVB is not active`` while the paper account still holds 4.05
shares. The rebuilt trader treats such a symbol as *untradable*: it never takes a slot, is
never sold or bought, and one alert names it. Three sources feed one set:

1. ``UNTRADABLE_SYMBOLS`` (comma separated, environment): the owner's explicit list.
2. The broker's asset status (``tradable: false``), looked up for held names and buy
   candidates; a lookup failure degrades to the configured set only.
3. A rejected order whose error says the asset is not active or tradable: the order is skipped
   and alerted, never raised, so one dead ticker cannot abort a run that has other orders.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

_log = logging.getLogger(__name__)

ENV_UNTRADABLE = "UNTRADABLE_SYMBOLS"
#: Lower-case fragments of the broker's rejection text for an asset it will not trade.
INACTIVE_MARKERS: tuple[str, ...] = (
    "is not active",
    "not active",
    "is not tradable",
    "not tradable",
    "not tradeable",
    "cannot be traded",
)


def configured_untradable(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """Symbols the owner listed in ``UNTRADABLE_SYMBOLS`` (upper-cased, blanks ignored)."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_UNTRADABLE, "")
    return frozenset(s.strip().upper() for s in raw.split(",") if s.strip())


def is_untradable_error(exc: BaseException) -> bool:
    """True when an order error means "the broker will not trade this asset".

    Matches the broker's rejection wording (alpaca-py ``APIError`` carries ``status_code``; its
    text embeds the JSON body, e.g. ``{"code":42210000,"message":"asset AVB is not active"}``).
    A 422 for any other reason (bad quantity, wrong time in force) is **not** matched, so a
    genuine order bug is still reported as an error.
    """
    text = str(exc).lower()
    if not any(marker in text for marker in INACTIVE_MARKERS):
        return False
    status: Any = getattr(exc, "status_code", None)
    return status in (None, 403, 422) or "422" in text or "unprocessable" in text


def resolve_untradable(
    broker: Any, symbols: Iterable[str], configured: Iterable[str] = ()
) -> set[str]:
    """Configured symbols among ``symbols`` plus those the broker marks ``tradable: false``.

    Never raises: a broker lookup failure is logged and only the configured set is used.
    """
    wanted = sorted(set(symbols))
    fixed = set(configured)
    out = {s for s in wanted if s in fixed}
    rest = [s for s in wanted if s not in out]
    if rest:
        try:
            out |= set(broker.get_untradable(rest))
        except Exception as exc:  # noqa: BLE001 - fail soft: never block a run on a lookup
            _log.warning("untradable lookup failed, using the configured set only: %s", exc)
    return out
