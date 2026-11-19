"""The paper trader's run-time settings, read once per run into one frozen record.

The environment variable names stay written in one place each (`cash_sweep`, `catastrophe_stop`,
`untradable`); `TraderConfig.from_env` only calls those accessors, so this module reads no
environment of its own. A malformed float is stored with the `ValueError` its accessor raised and
re-raised by the matching method where the trader used to parse it, so each setting keeps its old
fail time (the reserve ends the run, the other two sweep floats fail soft inside the sweep-buy
``try``) although every value is now read at the top of ``run()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ggTrader.paper import cash_sweep, catastrophe_stop
from ggTrader.paper.untradable import configured_untradable

_RESERVE = "reserve_pct"
_MIN_CLIP = "min_clip_usd"
_BUY_TRIGGER = "buy_trigger_pct"


@dataclass(frozen=True)
class TraderConfig:
    """One run's settings. The risk limits stay in `RiskConfig`.

    Attributes:
        dry_run: log intended orders instead of submitting them.
        read_only: write nothing to the ledger.
        stop_enabled: the catastrophe stop is on.
        stop_threshold: the stop's loss floor; ``None`` when the stop is off or the value is
            invalid.
        stop_threshold_error: the error the threshold accessor raised, else ``None``.
        sweep_enabled: the cash sweep is on.
        sweep_symbol: the sweep's ETF; ``None`` when the sweep is off.
        untradable: symbols the owner listed as untradable.
        sweep_reserve_pct: cash buffer fraction; ``None`` when the sweep is off or the value is
            invalid.
        sweep_min_clip_usd: smallest sweep buy; same ``None`` rule.
        sweep_buy_trigger_pct: cash fraction that triggers a sweep buy; same ``None`` rule.
        sweep_parse_errors: the error each invalid sweep float raised, keyed by accessor name
            (``reserve_pct``, ``min_clip_usd``, ``buy_trigger_pct``); empty when all parsed.
    """

    dry_run: bool
    read_only: bool
    stop_enabled: bool
    stop_threshold: float | None
    stop_threshold_error: ValueError | None
    sweep_enabled: bool
    sweep_symbol: str | None
    untradable: frozenset[str]
    sweep_reserve_pct: float | None
    sweep_min_clip_usd: float | None
    sweep_buy_trigger_pct: float | None
    sweep_parse_errors: Mapping[str, ValueError]

    @classmethod
    def from_env(cls, *, dry_run: bool, read_only: bool) -> TraderConfig:
        """Read every run-time setting; never raises for a malformed number.

        The stop threshold is read only when the stop is on, and the sweep symbol and its three
        floats only when the sweep is on: nothing reads them otherwise, so a bad value there is
        harmless, as before.
        """
        stop_enabled = catastrophe_stop.catastrophe_stop_enabled()
        stop_threshold: float | None = None
        stop_error: ValueError | None = None
        if stop_enabled:
            try:
                stop_threshold = catastrophe_stop.catastrophe_stop_pct()
            except ValueError as exc:
                stop_error = exc

        sweep_enabled = cash_sweep.sweep_enabled()
        sweep_symbol: str | None = None
        floats: dict[str, float | None] = {_RESERVE: None, _MIN_CLIP: None, _BUY_TRIGGER: None}
        errors: dict[str, ValueError] = {}
        if sweep_enabled:
            sweep_symbol = cash_sweep.sweep_symbol()
            accessors: dict[str, Callable[[], float]] = {
                _RESERVE: cash_sweep.reserve_pct,
                _MIN_CLIP: cash_sweep.min_clip_usd,
                _BUY_TRIGGER: cash_sweep.buy_trigger_pct,
            }
            for name, accessor in accessors.items():
                try:
                    floats[name] = accessor()
                except ValueError as exc:
                    errors[name] = exc

        return cls(
            dry_run=dry_run,
            read_only=read_only,
            stop_enabled=stop_enabled,
            stop_threshold=stop_threshold,
            stop_threshold_error=stop_error,
            sweep_enabled=sweep_enabled,
            sweep_symbol=sweep_symbol,
            untradable=configured_untradable(),
            sweep_reserve_pct=floats[_RESERVE],
            sweep_min_clip_usd=floats[_MIN_CLIP],
            sweep_buy_trigger_pct=floats[_BUY_TRIGGER],
            sweep_parse_errors=MappingProxyType(errors),
        )

    def _sweep_float(self, name: str, value: float | None) -> float:
        error = self.sweep_parse_errors.get(name)
        if error is not None:
            raise error
        if value is None:
            raise RuntimeError(f"{name} was not read: the cash sweep is off")
        return value

    def reserve_pct(self) -> float:
        """The sweep's cash-buffer fraction; re-raises the stored parse error."""
        return self._sweep_float(_RESERVE, self.sweep_reserve_pct)

    def min_clip_usd(self) -> float:
        """The smallest sweep buy; re-raises the stored parse error."""
        return self._sweep_float(_MIN_CLIP, self.sweep_min_clip_usd)

    def buy_trigger_pct(self) -> float:
        """The cash fraction that triggers a sweep buy; re-raises the stored parse error."""
        return self._sweep_float(_BUY_TRIGGER, self.sweep_buy_trigger_pct)
