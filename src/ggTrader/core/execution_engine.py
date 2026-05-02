"""Live execution engine for transition from optimization to real-world trading."""

from __future__ import annotations

import html
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from ggTrader.utils.result_db_manager import ResultDBManager

from ggTrader.core.regime_filtering import (
    _compute_altcoin_index_mask,
    _compute_btc_correlations,
    _compute_btc_regime_mask,
)
from ggTrader.core.trade_tracker import TradeTracker
from ggTrader.data.live.cached_loader import CachedExchangeLoader
from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.utils.notifier import build_notifiers_from_env
from ggTrader.indicators.strategies import (
    AtrTrailingExit,
    BollingerMeanReversionEntry,
    DonchianBreakoutEntry,
    EmaCrossEntry,
    FixedStopTakeProfit,
    KeltnerBreakoutEntry,
    MacdCrossEntry,
    PsarAdxEntry,
    RsiReversalEntry,
    StochRsiReversalEntry,
    SupertrendFlipEntry,
    TrailingStopExit,
)


def _safe_extract_filled_amount(order: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the actual filled amount from a CCXT order response.

    Prefers ``order["filled"]`` (the canonical CCXT field for cumulative
    filled quantity). Falls back to summing ``order["trades"]`` amounts if
    ``filled`` is missing. Returns ``None`` if no usable value is present
    so the caller can decide whether to fall back to a stored estimate.

    Used by ``_record_exit`` to ensure ``position_closes.csv`` reflects the
    real exchange fill quantity rather than the bot's pre-trade expectation
    (Bug D in the 2026-04-09 audit).
    """
    if not order:
        return None
    filled = order.get("filled")
    if filled and float(filled) > 0:
        return float(filled)
    trades = order.get("trades") or []
    if trades:
        total = sum(float(t.get("amount") or 0) for t in trades)
        if total > 0:
            return total
    return None


def _safe_extract_fill_price(order: Optional[Dict[str, Any]]) -> Optional[float]:
    """Robustly extract a meaningful fill price from a CCXT order response.

    Tries multiple fields in order of preference. Returns None if none are
    valid (caller should treat None as "unknown" and refuse to record bogus
    data rather than fall back to 0).
    """
    if not order:
        return None
    # Prefer cost/filled — CCXT's standardized 'cost' is the total quote-currency
    # value of all fills, so cost/filled is the true VWAP. Kraken via CCXT has
    # been observed to return stale/partial 'average' values on multi-fill orders
    # (see PENGU 2026-04-02: average=0.01365 but real VWAP=0.006225), so we no
    # longer trust 'average' as the primary source.
    cost = order.get("cost")
    filled = order.get("filled")
    if cost and filled and float(cost) > 0 and float(filled) > 0:
        return float(cost) / float(filled)
    avg = order.get("average")
    if avg and float(avg) > 0:
        return float(avg)
    # Compute weighted average from trade fills if available
    trades = order.get("trades") or []
    if trades:
        total_amt = 0.0
        total_val = 0.0
        for t in trades:
            amt = float(t.get("amount") or 0)
            px = float(t.get("price") or 0)
            if amt > 0 and px > 0:
                total_amt += amt
                total_val += amt * px
        if total_amt > 0:
            return total_val / total_amt
    # Fall back to limit price for filled limit orders (e.g. OCO take-profit)
    price = order.get("price")
    if price and float(price) > 0 and order.get("status") in ("closed", "filled"):
        return float(price)
    return None

def _format_hold_time(entry_time: Optional[str]) -> str:
    """Render ISO8601-UTC entry_time as a compact hold-duration string.

    ``1d 4h``, ``3h 22m``, ``14m`` — chooses the largest two units that fit.
    Returns ``"?"`` if entry_time is missing or unparseable.
    """
    if not entry_time:
        return "?"
    try:
        t0 = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - t0
        total_min = int(delta.total_seconds() // 60)
    except Exception:
        return "?"
    if total_min < 0:
        return "?"
    days, rem_min = divmod(total_min, 1440)
    hours, minutes = divmod(rem_min, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_entry_alert(
    symbol: str,
    strategy: str,
    exit_name: str,
    fill_price: float,
    amount: float,
    cost_usd: float,
    stop_pct: float,
    open_positions: int,
    universe_size: int,
) -> str:
    """HTML-formatted Telegram alert for a successful entry fill."""
    base = symbol.split("-")[0]
    sym_e = html.escape(symbol)
    strat_e = html.escape(strategy or "?")
    exit_e = html.escape(exit_name or "?")
    return (
        f"🟢 <b>BUY {sym_e}</b> @ <code>${fill_price:,.6g}</code>\n"
        f"strategy: <code>{strat_e}</code> / <code>{exit_e}</code>\n"
        f"size: <code>{amount:,.6g} {html.escape(base)}</code> "
        f"(~${cost_usd:,.2f})\n"
        f"stop: -{stop_pct:.2f}%\n"
        f"open positions: {open_positions}/{universe_size}"
    )


def _format_exit_alert(
    symbol: str,
    strategy: str,
    exit_reason: str,
    entry_price: Optional[float],
    exit_price: float,
    amount: float,
    entry_fee: float,
    exit_fee: float,
    entry_time: Optional[str],
    open_positions: int,
    universe_size: int,
) -> str:
    """HTML-formatted Telegram alert for an exit fill."""
    sym_e = html.escape(symbol)
    strat_e = html.escape(strategy or "?")
    reason_e = html.escape(exit_reason or "?")
    hold_str = _format_hold_time(entry_time)

    if entry_price and entry_price > 0:
        pnl_usd = (exit_price - entry_price) * amount - (entry_fee + exit_fee)
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        win = pnl_usd >= 0
        marker = "✅" if win else "❌"
        sign = "+" if pnl_usd >= 0 else "-"
        pnl_line = (
            f"PnL: <code>{sign}${abs(pnl_usd):,.2f}</code> "
            f"({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%) {marker}"
        )
        entry_line = (
            f"entry: <code>${entry_price:,.6g}</code> (held {hold_str})"
        )
    else:
        marker = "⚠️"
        pnl_line = "PnL: <code>?</code> (no stored entry price)"
        entry_line = f"entry: <code>?</code> (held {hold_str})"

    return (
        f"🔴 <b>SELL {sym_e}</b> @ <code>${exit_price:,.6g}</code>  {marker}\n"
        f"{entry_line}\n"
        f"{pnl_line}\n"
        f"reason: <code>{reason_e}</code>\n"
        f"strategy: <code>{strat_e}</code>\n"
        f"open positions: {open_positions}/{universe_size}"
    )


_BTC_SYMBOL = "BTC-USD"

# Minimum USD value to consider a balance a real position. Exchange sells often
# leave sub-penny dust that should not be tracked or reconciled.
_DUST_THRESHOLD_USD = 1.00

# Interim polling interval (seconds) — how often we check whether TSL/OCO
# orders triggered between 4h candle reconciliations. 5 min = 48 polls per
# 4h window, well within Kraken rate limits.
POLL_INTERVAL_SEC = 300


def setup_live_logger() -> logging.Logger:
    """Configures a thread-safe logger writing to both Console and logs/live_trader.log"""
    logger = logging.getLogger("ggTraderLive")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (10MB max size, keeps 5 backups)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = RotatingFileHandler(log_dir / "live_trader.log", maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class ExecutionEngine:
    """Live trading bot orchestrator using optimized WFO parameters."""

    STRATEGY_MAP = {
        "psar_adx": PsarAdxEntry,
        "ema_cross": EmaCrossEntry,
        "rsi_reversal": RsiReversalEntry,
        "macd_cross": MacdCrossEntry,
        "bbands_mean_reversion": BollingerMeanReversionEntry,
        "donchian_breakout": DonchianBreakoutEntry,
        "supertrend_flip": SupertrendFlipEntry,
        "stoch_rsi_reversal": StochRsiReversalEntry,
        "keltner_breakout": KeltnerBreakoutEntry,
    }

    EXIT_MAP = {
        "atr_trailing": AtrTrailingExit,
        "fixed_sl_tp": FixedStopTakeProfit,
        "trailing_stop": TrailingStopExit,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        results_path: Optional[str] = None,
        db_manager: Optional[ResultDBManager] = None,
        run_id: str = "LIVE",
    ):
        """Initialize engine.

        If results_path is provided, it loads per-coin optimized parameters.
        Otherwise, it uses the global params in config.
        """
        self.config = config
        self.results_path = results_path
        self.db_manager = db_manager
        self.run_id = run_id
        self.logger = setup_live_logger()
        self.exchange_id = config.get("EXCHANGE", "kraken")

        # Initialize CCXT exchange with caching
        self.loader = CachedExchangeLoader(exchange_id=self.exchange_id)
        self.exchange = self.loader.exchange

        # API Keys from env if not provided in config
        if not self.exchange.apiKey:
            self.exchange.apiKey = os.getenv("KRAKEN_KEY")
            self.exchange.secret = os.getenv("KRAKEN_SECRET")

        self.state = "INITIALIZED"
        self.interval = config.get("INTERVAL", "4h")
        self.symbols = config.get("SYMBOLS", [])
        self.per_coin_params = {}
        self.active_positions = {}
        self.persistence_path = config.get("PERSISTENCE_PATH", "data/active_positions.json")
        self._reopt_done_month: int | None = None
        self._reopt_flag_path = Path("data/last_reopt_month.txt")
        self._load_reopt_flag()
        self.tracker = TradeTracker(
            data_dir=config.get("TRACKER_DATA_DIR", "data/live"),
            db_manager=self.db_manager,
            run_id=self.run_id,
        )

        # Real-time fill alerts. Silently no-ops if no channels are configured
        # (TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID or DISCORD_WEBHOOK_URL).
        self.notifiers = build_notifiers_from_env()

        if results_path:
            self.load_optimized_params(results_path)
            # Override symbols if symbols not provided in config
            if not self.symbols:
                self.symbols = list(self.per_coin_params.keys())

        self.load_state()
        self._reconcile_positions()

    def _notify(self, message: str) -> None:
        """Fan-out a fill alert to every configured notifier.

        Exceptions from individual backends are swallowed so a Telegram outage
        or 429 can never crash the trade loop. Logs at debug level.
        """
        if not self.notifiers:
            return
        for n in self.notifiers:
            try:
                n.send_text(message)
            except Exception as e:
                self.logger.debug(f"  [Notify] {n.name} send failed: {e!r}")

    # ------------------------------------------------------------------
    # Parameter and state loading
    # ------------------------------------------------------------------

    def load_optimized_params(self, json_path: str) -> None:
        """Load per-coin parameters from WFO run_results.json.

        Applies risk-control gates from config:
        - SYMBOL_BLACKLIST: manual ban list (e.g. illiquid or misbehaving coins)
        - MIN_ROBUSTNESS_SCORE: drops coins whose WFO robustness is too low
        - MAX_COINS_PER_STRATEGY: limits how many coins share the same entry strategy
        """
        if not os.path.exists(json_path):
            self.logger.warning(f"Results file {json_path} not found.")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        sp = data.get("strategy_parameters", {})
        if "per_coin" in sp:
            per_coin = sp["per_coin"]
        elif "per_coin_results" in data:
            per_coin = data["per_coin_results"]
        elif "per_coin_results" in data.get("configuration", {}):
            per_coin = data["configuration"]["per_coin_results"]
        else:
            per_coin = {}

        min_rob = float(self.config.get("MIN_ROBUSTNESS_SCORE", 0.0))
        max_per_strategy = self.config.get("MAX_COINS_PER_STRATEGY", None)
        # Normalise blacklist to a set for O(1) lookup. Accept symbols with or
        # without the -USD suffix to be forgiving.
        raw_blacklist = self.config.get("SYMBOL_BLACKLIST", []) or []
        blacklist = {s.upper() for s in raw_blacklist}
        blacklist |= {f"{s.upper()}-USD" for s in raw_blacklist if "-" not in s}

        strategy_counts: Dict[str, int] = defaultdict(int)
        dropped_blacklist: List[str] = []
        dropped_low_rob: List[str] = []
        dropped_strategy_cap: List[str] = []

        for symbol, result in per_coin.items():
            best_params = result.get("best_params", {})
            strategy = result.get("best_strategy")
            exit_strategy = result.get("best_exit")
            robustness = float(result.get("robustness_score", 0.0))

            # Gate 0: manual blacklist (highest priority — runs before any other gate)
            if symbol.upper() in blacklist:
                dropped_blacklist.append(symbol)
                continue

            # Gate 1: minimum robustness
            if min_rob > 0 and robustness < min_rob:
                dropped_low_rob.append(f"{symbol}({robustness:.3f})")
                continue

            # Gate 2: max coins per entry strategy
            if max_per_strategy is not None and strategy_counts[strategy] >= max_per_strategy:
                dropped_strategy_cap.append(f"{symbol}({strategy})")
                continue

            strategy_counts[strategy] += 1
            self.per_coin_params[symbol] = {
                "strategy_name": strategy,
                "exit_name": exit_strategy,
                "params": best_params,
                "robustness_score": robustness,
            }

        if dropped_blacklist:
            self.logger.info(
                f"  [Gates] Dropped {len(dropped_blacklist)} coin(s) on "
                f"SYMBOL_BLACKLIST: {dropped_blacklist}"
            )
        if dropped_low_rob:
            self.logger.info(
                f"  [Gates] Dropped {len(dropped_low_rob)} coin(s) below "
                f"MIN_ROBUSTNESS_SCORE={min_rob}: {dropped_low_rob}"
            )
        if dropped_strategy_cap:
            self.logger.info(
                f"  [Gates] Dropped {len(dropped_strategy_cap)} coin(s) exceeding "
                f"MAX_COINS_PER_STRATEGY={max_per_strategy}: {dropped_strategy_cap}"
            )
        self.logger.info(f"Loaded optimized parameters for {len(self.per_coin_params)} symbols.")

    def reload_params(self) -> None:
        """Re-detect and reload the latest research params."""
        from ggTrader.utils.state_manager import get_latest_research_run

        latest_res = get_latest_research_run()
        if latest_res:
            self.per_coin_params = {}
            self.load_optimized_params(str(latest_res))
            self.symbols = list(self.per_coin_params.keys())
            self.results_path = str(latest_res)
            self.logger.info(f"Reloaded research params from {latest_res}")
        else:
            self.logger.warning("reload_params: no research results found.")

    def load_state(self) -> None:
        """Load active positions from persistence file."""
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, "r") as f:
                self.active_positions = json.load(f)
            self.logger.info(f"Loaded {len(self.active_positions)} active positions from state.")

    def save_state(self) -> None:
        """Save active positions to persistence file (atomic write)."""
        import tempfile

        dir_name = os.path.dirname(self.persistence_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.active_positions, f, indent=4)
            os.replace(tmp_path, self.persistence_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Reoptimization flag persistence
    # ------------------------------------------------------------------

    def _load_reopt_flag(self) -> None:
        """Load the last completed reoptimization month from disk."""
        try:
            if self._reopt_flag_path.exists():
                self._reopt_done_month = int(self._reopt_flag_path.read_text().strip())
        except (ValueError, OSError):
            self._reopt_done_month = None

    def _save_reopt_flag(self, month: int) -> None:
        """Persist the reoptimization month to disk so restarts don't re-trigger."""
        try:
            self._reopt_flag_path.parent.mkdir(parents=True, exist_ok=True)
            self._reopt_flag_path.write_text(str(month))
        except OSError as e:
            self.logger.warning(f"Could not save reopt flag: {e}")

    # ------------------------------------------------------------------
    # Exchange reconciliation
    # ------------------------------------------------------------------

    def _reconcile_positions(self) -> None:
        """Reconcile local JSON state against actual exchange holdings on startup.

        Queries the exchange for current non-zero balances and compares them
        against self.active_positions. Logs discrepancies but does not auto-
        correct — flags stale or missing entries so the operator can investigate.
        """
        if self.config.get("DRY_RUN", False):
            return

        try:
            balance = self.exchange.fetch_balance()
            held_symbols: Dict[str, float] = {}
            for asset, amounts in balance.get("total", {}).items():
                if amounts and float(amounts) > 0 and asset not in ("USD", "USDT", "USDC"):
                    # Convert asset to our symbol format (e.g. BTC -> BTC-USD)
                    sym = f"{asset}-USD"
                    held_symbols[sym] = float(amounts)

            # Positions in JSON but not held on exchange (TSL may have closed them)
            stale = [s for s in self.active_positions if s not in held_symbols]
            if stale:
                self.logger.warning(f"  [Reconcile] {len(stale)} position(s) in local state "
                      f"but NOT held on exchange — may have been closed: {stale}")
                preserved: list[str] = []
                for s in stale:
                    pos = self.active_positions[s]
                    exit_order_id = pos.get("tsl_order_id") or pos.get("oco_order_id")
                    exit_reason = ("trailing_stop" if pos.get("tsl_order_id")
                                   else "oco_exit")
                    recorded = False
                    if exit_order_id and exit_order_id not in (
                        "dry_run_tsl_id", "dry_run_oco_id"
                    ):
                        recorded = self._record_exit(
                            symbol=s,
                            sell_order_id=exit_order_id,
                            pos=pos,
                            exit_reason=exit_reason,
                            allow_ticker_fallback=False,
                        )
                    if recorded:
                        del self.active_positions[s]
                    else:
                        # Bug B fix: do NOT orphan the position. Mark it for
                        # later repair so the auto-sync at the start of
                        # `ggt pnl-daily` (or a manual `ggt repair`) can pull
                        # the trade from Kraken's authoritative history and
                        # rebuild the position_closes row via FIFO matching.
                        pos["pending_repair"] = True
                        preserved.append(s)
                if preserved:
                    self.logger.warning(
                        f"  [Reconcile] {len(preserved)} stale position(s) "
                        f"preserved with pending_repair=True (no fill price "
                        f"available): {preserved}"
                    )
                self.save_state()

            # Balances on exchange but not tracked in JSON (crash recovery).
            # Filter out dust balances (sub-$1 leftovers from partial fills).
            untracked: list[str] = []
            for s in held_symbols:
                if s in self.active_positions:
                    continue
                try:
                    ticker = self.exchange.fetch_ticker(s)
                    usd_value = held_symbols[s] * (ticker.get("last") or 0)
                except Exception:
                    usd_value = 0.0
                if usd_value < _DUST_THRESHOLD_USD:
                    self.logger.info(
                        f"  [Reconcile] Ignoring dust balance {s}: "
                        f"{held_symbols[s]:.8g} (~${usd_value:.4f})"
                    )
                    continue
                untracked.append(s)
            if untracked:
                self.logger.warning(f"  [Reconcile] {len(untracked)} position(s) held on exchange "
                      f"but NOT in local state — adding as untracked: {untracked}")
                for s in untracked:
                    coin_params = self.per_coin_params.get(s)
                    # Bug C fix: try to seed entry data from the local trade
                    # log so future record_sell calls produce a complete
                    # round-trip in position_closes.csv. Without this, the
                    # exit lands in trade_log only and the report's
                    # realised PnL / win-rate aggregation misses the trade.
                    seeded = self.tracker.find_open_buy_for(s)
                    if seeded:
                        self.logger.info(
                            f"  [Reconcile] {s}: seeded entry from trade_log "
                            f"(price=${seeded['price']:.6g}, "
                            f"time={seeded['time']}, fee=${seeded['fee']:.4f})"
                        )
                        entry_order_id = seeded["order_id"]
                        entry_price: Optional[float] = seeded["price"]
                        entry_time: Optional[str] = seeded["time"]
                        entry_fee: float = seeded["fee"]
                    else:
                        entry_order_id = "unknown_reconciled"
                        entry_price = None
                        entry_time = None
                        entry_fee = 0.0
                    self.active_positions[s] = {
                        "entry_order_id": entry_order_id,
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "entry_fee": entry_fee,
                        "amount": held_symbols[s],
                        "stop_pct": (coin_params["params"].get("stop_pct", 3.0)
                                     if coin_params else 3.0),
                        "exit_name": (coin_params.get("exit_name", "atr_trailing")
                                      if coin_params else "atr_trailing"),
                        "tsl_order_id": None,
                    }
                    if not coin_params:
                        self.logger.warning(
                            f"  [Reconcile] {s} not in current trading universe "
                            f"— using defaults, needs manual review"
                        )
                self.save_state()

            # Clean up dust positions already in local state.  Covers the case
            # where a previous sell left a sub-penny amount on the exchange and
            # the position lingers in active_positions forever because no exit
            # signal ever fires on it (fixed_sl_tp never hits SL/TP at dust
            # size, trailing stops can't be placed on dust).
            dust_in_local: list[str] = []
            for sym, pos in list(self.active_positions.items()):
                amt = float(pos.get("amount", 0) or 0)
                entry = float(pos.get("entry_price") or 0)
                cost_value = amt * entry
                if cost_value >= _DUST_THRESHOLD_USD:
                    continue
                # Cost basis missing or below threshold — confirm with live price.
                try:
                    ticker = self.exchange.fetch_ticker(sym)
                    cur_value = amt * float(ticker.get("last") or 0)
                except Exception:
                    cur_value = cost_value
                if max(cost_value, cur_value) < _DUST_THRESHOLD_USD:
                    dust_in_local.append(sym)
                    self.logger.info(
                        f"  [Reconcile] Removing dust position {sym}: "
                        f"amount={amt:.8g} value=~${max(cost_value, cur_value):.4f}"
                    )
                    del self.active_positions[sym]
            if dust_in_local:
                self.save_state()

            if not stale and not untracked and not dust_in_local:
                self.logger.info(
                    f"  [Reconcile] State verified: "
                    f"{len(self.active_positions)} position(s) match exchange."
                )

        except Exception as e:
            self.logger.warning(
                f"  [Reconcile] Exchange reconciliation failed "
                f"({e!r}) — using local state."
            )

    def _poll_open_exit_orders(self) -> None:
        """Targeted poll: check whether any open TSL/OCO orders have triggered.

        Lightweight alternative to full reconciliation — only fetches order status
        for symbols with a known exit order ID. Called on a 5-minute cadence
        between 4h candle iterations so that stop-out events get recorded
        promptly instead of waiting up to 4 hours.
        """
        if self.config.get("DRY_RUN", False) or not self.active_positions:
            return

        triggered: list[str] = []
        for symbol, pos in list(self.active_positions.items()):
            exit_order_id = pos.get("tsl_order_id") or pos.get("oco_order_id")
            if not exit_order_id or exit_order_id in (
                "dry_run_tsl_id", "dry_run_oco_id"
            ):
                continue
            try:
                pair = symbol.replace("-", "/") if "/" not in symbol else symbol
                order = self.exchange.fetch_order(exit_order_id, pair)
            except Exception as e:
                self.logger.debug(
                    f"  [Poll] fetch_order {exit_order_id} ({symbol}) failed: {e!r}"
                )
                continue

            status = (order.get("status") or "").lower()
            if status not in ("closed", "filled"):
                continue

            exit_reason = "trailing_stop" if pos.get("tsl_order_id") else "oco_exit"
            self.logger.info(
                f"  [Poll] {symbol} exit triggered: order={exit_order_id} "
                f"reason={exit_reason}"
            )
            recorded = self._record_exit(
                symbol=symbol,
                sell_order_id=exit_order_id,
                pos=pos,
                exit_reason=exit_reason,
                sell_order=order,
                allow_ticker_fallback=False,
            )
            if recorded:
                triggered.append(symbol)
            else:
                # Helper already logged the error. Position stays in
                # active_positions with the next reconcile / repair to retry.
                pos["pending_repair"] = True

        for symbol in triggered:
            del self.active_positions[symbol]

        if triggered:
            self.save_state()

    # ------------------------------------------------------------------
    # Exit recording (canonical helper used by every sell path)
    # ------------------------------------------------------------------

    def _record_exit(
        self,
        symbol: str,
        sell_order_id: str,
        pos: Dict[str, Any],
        exit_reason: str,
        sell_order: Optional[Dict[str, Any]] = None,
        allow_ticker_fallback: bool = False,
    ) -> bool:
        """Canonical exit-recording helper used by every sell code path.

        Resolves a fill price via ``_safe_extract_fill_price`` (with optional
        ``fetch_my_trades`` and ticker fallbacks), reads the actual filled
        amount via ``_safe_extract_filled_amount`` (Bug D fix), and writes a
        single ``record_sell`` row through ``self.tracker``.

        Returns ``True`` if a row was successfully written, ``False`` if no
        usable fill price could be determined. The caller decides whether to
        delete the position from ``active_positions`` based on the return —
        Bug B fix: callers preserve the position for later repair on False.

        Args:
            symbol: Dashed symbol like ``BTC-USD``.
            sell_order_id: The sell order ID returned by Kraken.
            pos: The position dict from ``self.active_positions[symbol]`` — used
                for ``entry_price``, ``entry_time``, ``entry_fee`` and (only as
                a last-resort) the stored ``amount``.
            exit_reason: One of ``strategy_signal``, ``trailing_stop``,
                ``oco_exit``, ``emergency_rollback``.
            sell_order: Optional pre-fetched CCXT order dict. If None, the
                helper will fetch it (with a 1-second settling sleep).
            allow_ticker_fallback: When True, if all fill-price extraction
                fails, fall back to a fresh ticker last-price. Safe for paths
                where the sell was just placed (strategy_signal,
                emergency_rollback). Unsafe for reconcile/poll where the sell
                may be hours old.
        """
        if sell_order_id in ("dry_run_sell_id", "dry_run_tsl_id", "dry_run_oco_id"):
            return False

        pair = symbol.replace("-", "/") if "/" not in symbol else symbol

        # Step 1: ensure we have an order dict, fetching if needed.
        if sell_order is None:
            try:
                time.sleep(1)  # brief settling time so Kraken finalises 'cost'/'filled'
                sell_order = self.exchange.fetch_order(sell_order_id, pair)
            except Exception as e:
                self.logger.warning(
                    f"  [Exit] {symbol}: fetch_order({sell_order_id}) failed: {e!r}"
                )
                sell_order = None

        # Step 2: extract a fill price using the cascade.
        exit_price = _safe_extract_fill_price(sell_order)

        # Step 3: if no price, try fetch_my_trades for the order id (Path B's
        # historical fallback — useful when Kraken's order endpoint returns a
        # stale or partial response but the trade fills are queryable).
        if exit_price is None or exit_price <= 0:
            try:
                recent = self.exchange.fetch_my_trades(pair, limit=20)
                matching = [t for t in recent if t.get("order") == sell_order_id]
                if matching:
                    total_amt = sum(float(t.get("amount") or 0) for t in matching)
                    total_val = sum(
                        float(t.get("amount") or 0) * float(t.get("price") or 0)
                        for t in matching
                    )
                    if total_amt > 0:
                        exit_price = total_val / total_amt
            except Exception as e:
                self.logger.debug(
                    f"  [Exit] {symbol}: fetch_my_trades fallback failed: {e!r}"
                )

        # Step 4: optional ticker fallback (only safe for fresh sells we just placed).
        if (exit_price is None or exit_price <= 0) and allow_ticker_fallback:
            try:
                ticker = self.exchange.fetch_ticker(pair)
                t_last = ticker.get("last") or ticker.get("close")
                if t_last and float(t_last) > 0:
                    exit_price = float(t_last)
                    self.logger.warning(
                        f"  [Exit] {symbol}: using ticker last=${exit_price:.6g} "
                        f"as estimated exit price (order fetch returned no fill)"
                    )
            except Exception as e:
                self.logger.warning(
                    f"  [Exit] {symbol}: ticker fallback failed: {e!r}"
                )

        if exit_price is None or exit_price <= 0:
            self.logger.error(
                f"  [Exit] {symbol}: could not determine fill price for order "
                f"{sell_order_id} — REFUSING to write a bogus record. The next "
                f"`ggt repair` (or auto-sync at the start of `ggt pnl-daily`) "
                f"will heal this from Kraken's authoritative trade history."
            )
            return False

        # Step 5: prefer the actual filled amount from the order over the
        # stored position amount (Bug D fix). Falls back to stored amount only
        # if the order doesn't expose ``filled``.
        filled_amount = _safe_extract_filled_amount(sell_order)
        if filled_amount is None:
            filled_amount = float(pos.get("amount") or 0)
            if filled_amount <= 0:
                self.logger.error(
                    f"  [Exit] {symbol}: no filled amount on order and no "
                    f"stored position amount — cannot record exit"
                )
                return False
        else:
            stored = float(pos.get("amount") or 0)
            if stored > 0 and abs(filled_amount - stored) / stored > 0.01:
                self.logger.warning(
                    f"  [Exit] {symbol}: order filled {filled_amount:.6g} but "
                    f"position stored {stored:.6g} — recording the actual fill"
                )

        # Step 6: extract sell-side fee from the order.
        fee_info = (sell_order.get("fee") or {}) if sell_order else {}
        sell_fee = float(fee_info.get("cost", 0) or 0)
        fee_currency = fee_info.get("currency", "USD")

        # Step 7: write the round-trip record.
        self.tracker.record_sell(
            symbol=symbol,
            order_id=sell_order_id,
            price=exit_price,
            amount=filled_amount,
            amount_usd=exit_price * filled_amount,
            fee=sell_fee,
            fee_currency=fee_currency,
            entry_price=pos.get("entry_price"),
            entry_time=pos.get("entry_time"),
            entry_fee=pos.get("entry_fee", 0),
            exit_reason=exit_reason,
        )

        # Step 8: push a rich fill alert. One call covers every sell path
        # (strategy_signal, trailing_stop, oco_exit, emergency_rollback).
        # open_positions count excludes the symbol we're closing now.
        try:
            coin_info = self.per_coin_params.get(symbol) or {}
            strategy = coin_info.get("strategy_name", "?")
            open_count = max(0, len(self.active_positions) - 1)
            msg = _format_exit_alert(
                symbol=symbol,
                strategy=strategy,
                exit_reason=exit_reason,
                entry_price=pos.get("entry_price"),
                exit_price=exit_price,
                amount=filled_amount,
                entry_fee=float(pos.get("entry_fee", 0) or 0),
                exit_fee=sell_fee,
                entry_time=pos.get("entry_time"),
                open_positions=open_count,
                universe_size=len(self.per_coin_params),
            )
            self._notify(msg)
        except Exception as e:
            self.logger.debug(f"  [Notify] exit alert build failed for {symbol}: {e!r}")

        return True

    def _handle_emergency_rollback(self, symbol: str, amount: float) -> None:
        """Flatten a position whose protective stop placement failed.

        Bug A fix: previously the emergency sell was placed but ``record_sell``
        was never called, so the rollback trade and its fees were orphaned
        from local CSVs forever. Now we record the exit through the canonical
        ``_record_exit`` helper before deleting the position.

        Called from the buy-side flow at the OCO and TSL placement failure
        branches. The position is still in ``self.active_positions`` at this
        point with a valid entry_price (we just bought it seconds ago), so
        ``_record_exit`` produces a complete round-trip record.
        """
        if self.config.get("DRY_RUN", False):
            if symbol in self.active_positions:
                del self.active_positions[symbol]
            return

        sell_id = self._execute_market_sell_order(symbol, amount)
        if not sell_id:
            self.logger.error(
                f"  [Order] EMERGENCY: cleanup sell also failed for {symbol} "
                f"— position unprotected on exchange and may need manual close"
            )
            return

        pos = self.active_positions.get(symbol)
        if pos is None:
            self.logger.error(
                f"  [Rollback] {symbol}: active_positions entry vanished before "
                f"recording — using minimal pos dict"
            )
            pos = {"amount": amount}

        recorded = self._record_exit(
            symbol=symbol,
            sell_order_id=sell_id,
            pos=pos,
            exit_reason="emergency_rollback",
            allow_ticker_fallback=True,
        )
        if recorded:
            self.active_positions.pop(symbol, None)
        else:
            # Helper logged the reason. Mark for repair so the next reconcile
            # / `ggt repair` run will pull the trade from Kraken history.
            if symbol in self.active_positions:
                self.active_positions[symbol]["pending_repair"] = True

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_total_portfolio_usd(self) -> Optional[float]:
        """Calculate total USD value of the portfolio (Free USD + held crypto).

        Returns ``None`` on exchange failure rather than a fabricated
        ``START_CASH`` fallback. Bug E fix: callers must handle ``None``
        explicitly. The balance-snapshot writer skips the snapshot entirely
        on ``None`` so we never poison ``balance_snapshots.csv`` with a
        misleading dummy value during a temporary Kraken outage; the
        position-sizing code falls back to ``CAPITAL_PER_TRADE``.
        """
        if self.config.get("DRY_RUN", False):
            return float(self.config.get("START_CASH", 1000.0))

        try:
            balance = self.exchange.fetch_balance()
            total_usd = 0.0
            tickers = self.exchange.fetch_tickers()

            # Normalization map for Kraken legacy codes if CCXT doesn't catch them
            norm_map = {"XBT": "BTC", "XETH": "ETH", "XXLM": "XLM", "ZUSD": "USD"}

            for coin, amount in balance.get("total", {}).items():
                if float(amount) > 0:
                    # Normalize symbol (strip prefix, use map)
                    norm_coin = norm_map.get(coin, coin)
                    is_kraken_legacy = (
                        norm_coin.startswith("X") and len(norm_coin) > 3
                        and norm_coin[1:] in ("BT", "ET", "LM")
                    )
                    if is_kraken_legacy:
                         # Extra safety for XBT, XETH, XXLM
                         norm_coin = norm_map.get(norm_coin, norm_coin)

                    if norm_coin in ("USD", "USDT", "USDC"):
                        total_usd += float(amount)
                    else:
                        pair = f"{norm_coin}/USD"
                        if pair in tickers and "last" in tickers[pair]:
                            total_usd += float(amount) * float(tickers[pair]["last"])
                        else:
                            # Fallback: try checking if the original coin code works as a pair
                            pair_orig = f"{coin}/USD"
                            if pair_orig in tickers and "last" in tickers[pair_orig]:
                                total_usd += float(amount) * float(tickers[pair_orig]["last"])

            return total_usd
        except Exception as e:
            self.logger.warning(
                f"  [Portfolio] Could not calculate total portfolio USD: {e!r} "
                f"— returning None so the snapshot writer can skip this cycle"
            )
            return None

    def _fetch_latest_data(self) -> pd.DataFrame:
        """Fetch the most recent candles for all coins plus BTC (for regime filter)."""
        lookback_limit = self.config.get("LOOKBACK_LIMIT", 200)
        # Always include BTC-USD so the regime filter EMA can be computed even
        # when BTC is not in the trading universe.
        fetch_symbols = list(self.symbols)
        if self.config.get("BTC_REGIME_FILTER", False) and _BTC_SYMBOL not in fetch_symbols:
            fetch_symbols = [_BTC_SYMBOL] + fetch_symbols
        return self.loader.fetch_ohlcv(
            symbols=fetch_symbols, interval=self.interval, limit=lookback_limit
        )

    # ------------------------------------------------------------------
    # Regime filter
    # ------------------------------------------------------------------

    def _compute_live_regime_allowance(self, ohlcv_df: pd.DataFrame) -> Dict[str, bool]:
        """Return {symbol: allow_entry} based on BTC/altcoin regime at the current bar.

        Reuses the same _compute_btc_regime_mask / _apply_tiered logic as Phase 2/3
        but reduces the result to a single per-symbol boolean for the last bar.
        Returns all True (no filtering) if BTC_REGIME_FILTER is disabled.
        """
        if not self.config.get("BTC_REGIME_FILTER", False):
            return {s: True for s in self.symbols}

        btc_regime = _compute_btc_regime_mask(ohlcv_df, self.config)
        alt_regime = (
            _compute_altcoin_index_mask(ohlcv_df, self.config)
            if self.config.get("ALTCOIN_REGIME_FILTER", False)
            else None
        )
        btc_corrs = _compute_btc_correlations(ohlcv_df, self.config)

        btc_min_corr = float(self.config.get("BTC_REGIME_FILTER_MIN_CORRELATION", 0.5))
        alt_min_corr = float(self.config.get("ALTCOIN_REGIME_FILTER_CORR_MIN", 0.3))

        # Current-bar regime status
        btc_bull_now = bool(btc_regime.iloc[-1]) if btc_regime is not None else True
        alt_bull_now = bool(alt_regime.iloc[-1]) if alt_regime is not None else True

        allowance: Dict[str, bool] = {}
        for symbol in self.symbols:
            corr = btc_corrs.get(symbol, 1.0)
            if corr >= btc_min_corr:
                allowance[symbol] = btc_bull_now
            elif corr >= alt_min_corr and alt_regime is not None:
                allowance[symbol] = alt_bull_now
            else:
                allowance[symbol] = True  # low BTC correlation — trade freely

        if btc_regime is not None:
            n_warmup = int(self.config.get("EMA_WARMUP_BARS", 200))
            blocked = [s for s, ok in allowance.items() if not ok]
            self.logger.info(
                f"  [Regime] EMA({n_warmup}) — BTC bull={btc_bull_now}, "
                f"alt bull={alt_bull_now}, blocked={blocked or 'none'}"
            )

        return allowance

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_latest_signals(self, ohlcv_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute entry/exit signals for each coin using its WFO-optimised strategy."""
        signals_results = {}

        available_symbols = ohlcv_df.columns.get_level_values(0).unique()
        for symbol in self.symbols:
            if symbol not in available_symbols:
                continue

            coin_info = self.per_coin_params.get(symbol)
            if not coin_info:
                continue

            strat_name = coin_info["strategy_name"]
            exit_name = coin_info["exit_name"]
            params = coin_info["params"]

            strat_cls = self.STRATEGY_MAP.get(strat_name)
            if not strat_cls:
                continue

            param_grid = {k: [v] for k, v in params.items()}
            symbol_df = ohlcv_df[[symbol]]
            close = symbol_df[(symbol, "close")]
            high = symbol_df[(symbol, "high")]
            low = symbol_df[(symbol, "low")]
            precomputer = IndicatorPrecomputer(close, high, low)

            entries, _ = strat_cls().compute_entries(precomputer, param_grid)

            exit_cls = self.EXIT_MAP.get(exit_name)
            if not exit_cls:
                continue
            exits, stops, prices = exit_cls().compute_exits(entries, precomputer, param_grid, 1)

            # Extract current ATR value for the coin so the stop placement
            # code can compute a proper initial trailing stop from fill price.
            atr_value = float("nan")
            if exit_name == "atr_trailing":
                atr_length = int(params.get("atr_length", 14))
                try:
                    atr_ind = precomputer.compute_atr(atr_length)
                    atr_arr = atr_ind.atrr.values if hasattr(atr_ind.atrr, "values") else atr_ind.atrr
                    last_atr = float(atr_arr[-1]) if atr_arr.ndim == 1 else float(atr_arr[-1, 0])
                    if not math.isnan(last_atr):
                        atr_value = last_atr
                except Exception:
                    pass

            signals_results[symbol] = {
                "entry": bool(entries[-1, 0]),
                "exit": bool(exits[-1, 0]),
                "exit_name": exit_name,
                "stop_price": float(stops[-1, 0]),
                "fill_price": float(prices[-1, 0]),
                "current_price": float(close.iloc[-1]),
                "atr_value": atr_value,
            }

        return signals_results

    # ------------------------------------------------------------------
    # Order execution helpers
    # ------------------------------------------------------------------

    def _validate_entry_preconditions(self, symbol: str, capital_usd: float) -> bool:
        """Return True if it's safe to place a buy order for this symbol.

        Checks:
        1. Free USD balance >= capital requested
        2. Resulting coin amount meets the exchange's minimum order size
        """
        if self.config.get("DRY_RUN", False):
            return True
        try:
            balance = self.exchange.fetch_balance()
            free_usd = float(balance.get("free", {}).get("USD", 0.0))
            if free_usd < capital_usd * 0.95:
                self.logger.warning(
                    f"  [Order] SKIP {symbol}: insufficient free balance "
                    f"(need ${capital_usd:.2f}, have ${free_usd:.2f})"
                )
                return False

            pair = symbol.replace("-", "/") if "-" in symbol else symbol
            if not self.exchange.markets:
                self.exchange.load_markets()
            market = self.exchange.market(pair)
            ticker = self.exchange.fetch_ticker(pair)
            price = ticker["last"]
            coin_amount = capital_usd / price
            min_amount = (market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0
            if coin_amount < float(min_amount):
                self.logger.warning(
                    f"  [Order] SKIP {symbol}: order size {coin_amount:.6f} below "
                    f"exchange minimum {min_amount}"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"  [Order] Precondition check failed for {symbol} ({e!r}) — BLOCKING"
            )
            return False
        return True

    def _execute_market_buy_order(self, symbol: str, amount_usd: float) -> Optional[str]:
        """Execute a market buy order via CCXT."""
        self.logger.info(f"EXECUTING MARKET BUY: {amount_usd} USD for {symbol}")
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: Order skipped.")
            return "dry_run_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            ticker = self.exchange.fetch_ticker(pair)
            price = ticker["last"]
            base_amount = amount_usd / price

            market = self.exchange.market(pair)
            base_amount_precise = self.exchange.amount_to_precision(market["symbol"], base_amount)

            order = self.exchange.create_market_buy_order(market["symbol"], base_amount_precise)
            return order["id"]
        except Exception as e:
            self.logger.error(f"Error placing market buy for {symbol}: {e}")
            return None

    def _execute_market_sell_order(self, symbol: str, amount: float) -> Optional[str]:
        """Execute a market sell order via CCXT (used for strategy exit signals)."""
        self.logger.info(f"EXECUTING MARKET SELL: {amount} {symbol}")
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: Sell order skipped.")
            return "dry_run_sell_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)
            amount_precise = self.exchange.amount_to_precision(market["symbol"], amount)
            order = self.exchange.create_market_sell_order(market["symbol"], amount_precise)
            return order["id"]
        except Exception as e:
            self.logger.error(f"Error placing market sell for {symbol}: {e}")
            return None

    def _execute_trailing_stop_order(
        self, symbol: str, amount: float, stop_pct: float
    ) -> Optional[str]:
        """Place a trailing stop loss order on Kraken.

        Kraken trailing-stop requires the trailing offset as a price value
        (not a percentage string). We fetch the current price and compute
        the dollar offset from the percentage.
        """
        self.logger.info(f"PLACING TRAILING STOP: {amount} {symbol} at -{stop_pct}%")
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: Order skipped.")
            return "dry_run_tsl_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)
            ticker = self.exchange.fetch_ticker(market["symbol"])
            current_price = ticker["last"]

            # Kraken expects the trailing offset as an absolute price distance
            trailing_offset = current_price * (stop_pct / 100.0)

            # Ensure offset meets Kraken's minimum tick size.
            # For micro-cap coins (e.g. PEPE ~$0.000007), a 1% offset can
            # round to 0 after price_to_precision, causing rejection.
            price_precision = market.get("precision", {}).get("price")
            if price_precision is not None:
                min_tick = 10 ** (-price_precision) if isinstance(price_precision, int) else price_precision
                if trailing_offset < min_tick:
                    self.logger.info(
                        f"  [TSL] Trailing offset for {symbol} (${trailing_offset:.10f}) "
                        f"below min tick (${min_tick:.10f}) — clamping to min tick"
                    )
                    trailing_offset = min_tick

            trailing_offset = float(
                self.exchange.price_to_precision(market["symbol"], trailing_offset)
            )

            # Final guard: reject if offset is still zero after rounding
            if not trailing_offset or trailing_offset <= 0:
                self.logger.error(
                    f"  [TSL] Trailing offset rounded to zero for {symbol} — cannot place TSL"
                )
                return None

            amount_precise = self.exchange.amount_to_precision(market["symbol"], amount)
            order = self.exchange.create_order(
                market["symbol"], "trailing-stop", "sell", amount_precise, None,
                {"trailingAmount": trailing_offset},
            )
            return order["id"]
        except Exception as e:
            self.logger.error(f"Error placing trailing stop for {symbol}: {e}")
            return None

    def _execute_oco_exit_order(
        self, symbol: str, amount: float, sl_price: float, tp_price: float
    ) -> Optional[str]:
        """Place a limit sell with stop-loss and take-profit on Kraken.

        Uses CCXT unified params (stopLossPrice / takeProfitPrice) which
        Kraken maps to conditional close orders attached to a limit sell.
        The limit price is set to the take-profit level.
        """
        self.logger.info(
            f"PLACING OCO EXIT: {amount} {symbol} SL={sl_price:.4f}, TP={tp_price:.4f}"
        )
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: OCO order skipped.")
            return "dry_run_oco_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)

            sl_price_f = float(self.exchange.price_to_precision(market["symbol"], sl_price))
            tp_price_f = float(self.exchange.price_to_precision(market["symbol"], tp_price))
            amount_precise = self.exchange.amount_to_precision(market["symbol"], amount)

            if sl_price_f == tp_price_f:
                self.logger.error(
                    f"  [OCO] SL and TP rounded to same price ({sl_price_f}) "
                    f"for {symbol} — cannot place OCO"
                )
                return None

            # CCXT unified params: Kraken creates conditional close orders
            # (stop-loss + take-profit) attached to a limit sell order.
            params = {
                "stopLossPrice": sl_price_f,
                "takeProfitPrice": tp_price_f,
            }
            order = self.exchange.create_order(
                market["symbol"], "limit", "sell", amount_precise, tp_price_f, params
            )
            return order["id"]
        except Exception as e:
            self.logger.error(f"Error placing OCO exit for {symbol}: {e}")
            return None

    def _cancel_open_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol (called before a strategy exit sell)."""
        if self.config.get("DRY_RUN", False):
            return
        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            open_orders = self.exchange.fetch_open_orders(pair)
            for o in open_orders:
                self.exchange.cancel_order(o["id"], pair)
                self.logger.info(f"  [Exit] Cancelled order {o['id']} for {symbol}")
        except Exception as e:
            self.logger.warning(f"  [Exit] Could not cancel orders for {symbol}: {e!r}")

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _estimate_stop_pct_for_sizing(
        self, symbol: str, sig: Dict[str, Any]
    ) -> float:
        """Pre-buy estimate of the exit's stop distance as a percentage.

        Mirrors the post-buy stop computation in ``_execute_trade_logic`` so
        adaptive sizing can pick a position that respects the eventual stop.
        The real stop is recomputed after fill — this is only used to budget
        risk at signal time. Returns a floor of 0.5% to avoid blowing the
        cap when ATR is transiently tiny.
        """
        coin_params = self.per_coin_params[symbol]["params"]
        exit_name = sig.get("exit_name", "atr_trailing")
        if exit_name == "atr_trailing":
            atr_value = sig.get("atr_value", float("nan"))
            atr_mult = float(coin_params.get("atr_multiplier", 3.0))
            price = float(sig.get("current_price", 0) or 0)
            if (
                not math.isnan(atr_value)
                and atr_value > 0
                and price > 0
            ):
                stop_pct = (atr_mult * atr_value / price) * 100.0
            else:
                # ATR unavailable — fall back to multiplier-as-percent
                # (same fallback used by the live stop placement path).
                stop_pct = atr_mult
        elif exit_name == "trailing_stop":
            stop_pct = float(coin_params.get("trailing_stop_pct", 3.0))
        else:  # fixed_sl_tp
            stop_pct = float(coin_params.get("stop_pct", 3.0))
        # Mirror the live trailing-stop floor so adaptive sizing budgets risk
        # against the actual clamped stop, not a fictitiously tight one.
        if exit_name in ("atr_trailing", "trailing_stop"):
            floor_key = (
                "MIN_ATR_TRAILING_PCT" if exit_name == "atr_trailing"
                else "MIN_TRAILING_STOP_PCT"
            )
            stop_pct = max(stop_pct, float(self.config.get(floor_key, 4.0)))
        return max(stop_pct, 0.5)

    def _compute_adaptive_position_usd(
        self, symbol: str, sig: Dict[str, Any], base_capital: float
    ) -> Optional[float]:
        """Volatility-normalized position size.

        ``position = (portfolio * TARGET_RISK_PCT) / stop_pct`` — at the stop,
        the loss is bounded to ``TARGET_RISK_PCT`` of portfolio regardless of
        how wide the coin's stop is. High-vol coins (wide ATR stop) get
        smaller positions, low-vol coins get larger ones. Capped at
        ``MAX_POSITION_PCT`` of portfolio; skipped if below
        ``MIN_POSITION_USD``.

        Returns ``None`` to signal "skip this entry" (sized too small to be
        worth paying fees + slippage on).
        """
        target_risk_pct = float(self.config.get("TARGET_RISK_PCT", 0.01))
        max_position_pct = float(self.config.get("MAX_POSITION_PCT", 0.15))
        min_position_usd = float(self.config.get("MIN_POSITION_USD", 15.0))

        stop_pct = self._estimate_stop_pct_for_sizing(symbol, sig)
        risk_usd = base_capital * target_risk_pct
        raw_position = risk_usd / (stop_pct / 100.0)
        cap_position = base_capital * max_position_pct
        position_usd = min(raw_position, cap_position)

        if position_usd < min_position_usd:
            self.logger.info(
                f"  [Sizing] {symbol}: adaptive size ${position_usd:.2f} "
                f"(stop≈{stop_pct:.2f}%, risk {target_risk_pct * 100:.2f}% "
                f"of ${base_capital:.2f}) below MIN_POSITION_USD="
                f"${min_position_usd:.2f} — skipping entry"
            )
            return None

        capped = raw_position > cap_position
        self.logger.info(
            f"  [Sizing] {symbol}: adaptive "
            f"(risk {target_risk_pct * 100:.2f}% @ stop≈{stop_pct:.2f}%) "
            f"-> ${position_usd:.2f}"
            + (f"  [CAPPED at {max_position_pct * 100:.0f}% of portfolio]"
               if capped else "")
        )
        return position_usd

    # ------------------------------------------------------------------
    # Trade logic
    # ------------------------------------------------------------------

    def _execute_trade_logic(
        self,
        signals_dict: Dict[str, Dict[str, Any]],
        regime_allowance: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Evaluate signals and execute orders."""
        if regime_allowance is None:
            regime_allowance = {}
        for symbol, sig in signals_dict.items():
            if sig["entry"] and symbol not in self.active_positions:
                # Regime gate — block entries in bear markets
                if not regime_allowance.get(symbol, True):
                    self.logger.info(f"  [Regime] BLOCKED entry for {symbol} (bear regime)")
                    continue

                adaptive = bool(self.config.get("ADAPTIVE_SIZING", False))

                if adaptive:
                    base_capital = self._get_total_portfolio_usd()
                    if base_capital is None:
                        # Bug E fix: refuse to size off a fabricated portfolio
                        # number when the exchange is down.
                        self.logger.warning(
                            f"  [Sizing] {symbol}: portfolio valuation unavailable "
                            f"— skipping entry until next cycle"
                        )
                        continue
                    adaptive_usd = self._compute_adaptive_position_usd(
                        symbol, sig, base_capital
                    )
                    if adaptive_usd is None:
                        # Too small to bother with (fees + slippage eat it).
                        continue
                    capital_per_trade = adaptive_usd
                else:
                    capital_per_trade = self.config.get("CAPITAL_PER_TRADE", 100.0)
                    self.logger.info(
                        f"  [Sizing] Fixed capital for {symbol}: ${capital_per_trade:.2f}"
                    )

                # Pre-flight: balance and min-order-size checks
                if not self._validate_entry_preconditions(symbol, capital_per_trade):
                    continue

                order_id = self._execute_market_buy_order(symbol, capital_per_trade)

                if order_id:
                    time.sleep(1)  # Brief wait for fill
                    order: Optional[Dict[str, Any]] = None
                    try:
                        order = self.exchange.fetch_order(
                            order_id,
                            symbol.replace("-", "/") if "/" not in symbol else symbol,
                        )
                        filled_amount = float(order.get("filled", 0.0) or 0.0)
                    except Exception as e:
                        self.logger.warning(
                            f"  [Order] fetch_order failed for {symbol} ({e!r}) — "
                            "estimating fill amount from ticker"
                        )
                        ticker = self.exchange.fetch_ticker(
                            symbol.replace("-", "/") if "/" not in symbol else symbol
                        )
                        filled_amount = capital_per_trade / ticker["last"]

                    if not filled_amount or filled_amount <= 0:
                        self.logger.error(
                            f"  [Order] Zero fill for {symbol} — skipping position record"
                        )
                        continue

                    # Determine actual fill price. Cascade through fallbacks:
                    # 1. order["average"] (from fetch_order)
                    # 2. weighted average of order["trades"] fills
                    # 3. order["price"] for closed limit orders
                    # 4. fresh ticker price (last resort if order fetch failed)
                    fill_price = _safe_extract_fill_price(order)
                    if fill_price is None:
                        try:
                            ticker = self.exchange.fetch_ticker(
                                symbol.replace("-", "/") if "/" not in symbol else symbol
                            )
                            fill_price = float(ticker["last"])
                            self.logger.warning(
                                f"  [Order] {symbol}: no fill price from order — "
                                f"using ticker last=${fill_price:.6g}"
                            )
                        except Exception:
                            fill_price = float(sig["current_price"])
                            self.logger.error(
                                f"  [Order] {symbol}: no fill price available — "
                                f"using stale signal price=${fill_price:.6g}"
                            )
                    fee_info = (order.get("fee") or {}) if order else {}
                    buy_fee = float(fee_info.get("cost", 0) or 0)
                    self.tracker.record_buy(
                        symbol=symbol, order_id=order_id, price=fill_price,
                        amount=filled_amount, amount_usd=capital_per_trade,
                        fee=buy_fee, fee_currency=fee_info.get("currency", "USD"),
                    )

                    # Determine the correct trailing stop percentage based on exit type.
                    # WFO optimises different params per exit:
                    #   - atr_trailing: atr_length + atr_multiplier (volatility-aware)
                    #   - trailing_stop: trailing_stop_pct
                    #   - fixed_sl_tp:  stop_pct + take_profit_pct
                    exit_name = sig["exit_name"]
                    coin_params = self.per_coin_params[symbol]["params"]

                    if exit_name == "atr_trailing":
                        # Compute the initial trailing stop from the current ATR
                        # value and the actual fill price.  The old approach used
                        # sig["stop_price"] from the backtest, but that tracks a
                        # historical peak across many bars and can be above the
                        # current price — nonsensical for a new entry.
                        atr_value = sig.get("atr_value", float("nan"))
                        atr_mult = float(coin_params.get("atr_multiplier", 3.0))

                        if not math.isnan(atr_value) and fill_price and atr_value > 0:
                            atr_stop = fill_price - atr_mult * atr_value
                            stop_pct = ((fill_price - atr_stop) / fill_price) * 100.0
                            self.logger.info(
                                f"  [Stop] {symbol} atr_trailing: ATR stop=${atr_stop:.6g}, "
                                f"price=${fill_price:.6g} -> trailing {stop_pct:.2f}%"
                            )
                        else:
                            # ATR unavailable (NaN / insufficient warmup) —
                            # fall back to atr_multiplier as a percentage.
                            stop_pct = atr_mult
                            self.logger.warning(
                                f"  [Stop] {symbol} atr_trailing: ATR is NaN "
                                f"(insufficient data warmup or zero range) — "
                                f"falling back to atr_multiplier={stop_pct:.2f}%"
                            )
                    elif exit_name == "trailing_stop":
                        stop_pct = float(coin_params.get("trailing_stop_pct", 3.0))
                        self.logger.info(
                            f"  [Stop] {symbol} trailing_stop: using WFO trailing_stop_pct={stop_pct}%"
                        )
                    else:
                        # fixed_sl_tp — stop_pct is the correct WFO-optimised param
                        stop_pct = float(coin_params.get("stop_pct", 3.0))

                    # Floor: 4h crypto bars routinely traverse tight ATR-derived
                    # stops in normal noise. Clamp upward so a too-aggressive
                    # WFO param can't decapitate a position in the first hour.
                    floor_key = (
                        "MIN_ATR_TRAILING_PCT" if exit_name == "atr_trailing"
                        else "MIN_TRAILING_STOP_PCT"
                    )
                    if exit_name in ("atr_trailing", "trailing_stop"):
                        floor = float(self.config.get(floor_key, 4.0))
                        if stop_pct < floor:
                            self.logger.info(
                                f"  [Stop] {symbol}: WFO stop_pct {stop_pct:.2f}% "
                                f"clamped up to {floor_key} floor {floor:.2f}%"
                            )
                            stop_pct = floor

                    # CRITICAL: store the actual fill price (not the stale signal close).
                    # Storing sig["current_price"] caused position_closes.csv to record
                    # bogus entry prices, leading to fake -36% to -54% losses on profitable
                    # trades. The fill_price comes from order["average"] (or fallbacks).
                    self.active_positions[symbol] = {
                        "entry_order_id": order_id,
                        "entry_price": fill_price,
                        "entry_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "amount": filled_amount,
                        "stop_pct": stop_pct,
                        "exit_name": exit_name,
                        "tsl_order_id": None,
                        "entry_fee": buy_fee,
                    }

                    # Place exit order on Kraken.
                    # fixed_sl_tp -> Kraken OCO (stop-loss + take-profit)
                    # atr_trailing / trailing_stop -> Kraken trailing-stop order
                    if exit_name == "fixed_sl_tp":
                        tp_pct = float(coin_params.get("take_profit_pct", 6.0))
                        # fill_price was already computed above using safe extraction
                        sl_price = fill_price * (1.0 - (stop_pct / 100.0))
                        tp_price = fill_price * (1.0 + (tp_pct / 100.0))

                        oco_id = self._execute_oco_exit_order(
                            symbol, filled_amount, sl_price, tp_price
                        )
                        if oco_id is None:
                            self.logger.error(
                                f"  [Order] OCO placement failed for {symbol} "
                                f"— closing position via emergency rollback sell"
                            )
                            self._handle_emergency_rollback(symbol, filled_amount)
                        else:
                            self.active_positions[symbol]["oco_order_id"] = oco_id
                    else:
                        tsl_id = self._execute_trailing_stop_order(symbol, filled_amount, stop_pct)
                        if tsl_id is None:
                            self.logger.error(
                                f"  [Order] TSL placement failed for {symbol} "
                                f"— closing position via emergency rollback sell"
                            )
                            self._handle_emergency_rollback(symbol, filled_amount)
                        else:
                            self.active_positions[symbol]["tsl_order_id"] = tsl_id

                    self.save_state()

                    # Entry alert — only fire if the emergency rollback path
                    # didn't remove the position. This also guarantees we never
                    # alert from state-reload paths, since this block only runs
                    # inside the `sig["entry"] and symbol not in active_positions`
                    # branch of a live signal evaluation.
                    if symbol in self.active_positions:
                        try:
                            strategy = self.per_coin_params.get(symbol, {}).get(
                                "strategy_name", "?"
                            )
                            msg = _format_entry_alert(
                                symbol=symbol,
                                strategy=strategy,
                                exit_name=exit_name,
                                fill_price=fill_price,
                                amount=filled_amount,
                                cost_usd=capital_per_trade,
                                stop_pct=stop_pct,
                                open_positions=len(self.active_positions),
                                universe_size=len(self.per_coin_params),
                            )
                            self._notify(msg)
                        except Exception as e:
                            self.logger.debug(
                                f"  [Notify] entry alert build failed for {symbol}: {e!r}"
                            )

            elif symbol in self.active_positions:
                if sig["exit"]:
                    exit_name = self.active_positions[symbol].get("exit_name", sig["exit_name"])
                    if exit_name == "fixed_sl_tp":
                        # fixed_sl_tp: strategy computes the exit condition — execute it
                        pos = self.active_positions[symbol]
                        amount = pos["amount"]
                        entry_price = pos.get("entry_price") or 0
                        pos_value = amount * entry_price

                        # Dust check: if the position is too small to sell,
                        # remove it silently instead of retrying every loop.
                        if pos_value < _DUST_THRESHOLD_USD:
                            self.logger.info(
                                f"  [Exit] Removing dust position {symbol}: "
                                f"{amount:.8g} (~${pos_value:.4f})"
                            )
                            del self.active_positions[symbol]
                            self.save_state()
                            continue

                        self.logger.info(f"  [Exit] STRATEGY EXIT (fixed_sl_tp) for {symbol}")
                        self._cancel_open_orders(symbol)
                        sell_id = self._execute_market_sell_order(symbol, amount)
                        if sell_id:
                            recorded = self._record_exit(
                                symbol=symbol,
                                sell_order_id=sell_id,
                                pos=pos,
                                exit_reason="strategy_signal",
                                allow_ticker_fallback=True,
                            )
                            pos["exit_time"] = time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            )
                            pos["exit_reason"] = "strategy_signal"
                            if recorded:
                                del self.active_positions[symbol]
                            else:
                                # Fill price could not be determined. Mark for
                                # repair so the next reconcile / auto-sync can
                                # heal it from Kraken's authoritative history.
                                pos["pending_repair"] = True
                            self.save_state()
                    else:
                        # atr_trailing / trailing_stop: Kraken's TSL handles the exit
                        self.logger.info(f"  [Exit] Strategy exit signal for {symbol} "
                              f"(exit={exit_name}) — TSL is active, no action needed")

    # ------------------------------------------------------------------
    # Monthly reoptimization
    # ------------------------------------------------------------------

    def _run_monthly_reoptimization(self) -> bool:
        """Run research + production pipeline and reload params.

        Returns True on success, False on failure (caller should retry).
        """
        import subprocess
        import sys
        from datetime import datetime, timedelta

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.logger.info("=" * 50)
        self.logger.info("  MONTHLY REOPTIMIZATION STARTED")
        self.logger.info(f"  End date for WFO: {yesterday}")
        self.logger.info("=" * 50)

        project_root = Path(__file__).resolve().parent.parent.parent.parent

        # Phase 1: Research (WFO)
        research_cmd = [
            sys.executable, "-u", "ggt.py",
            "research", "--top", "50", "--workers", "2",
            "--end-date", yesterday,
        ]
        self.logger.info(f"Running research: {' '.join(research_cmd)}")
        try:
            result = subprocess.run(
                research_cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            if result.returncode != 0:
                self.logger.error(f"Research failed (exit {result.returncode}): {result.stderr[-500:]}")
                return False
            self.logger.info("Research completed successfully.")
        except subprocess.TimeoutExpired:
            self.logger.error("Research timed out after 2 hours.")
            return False
        except Exception as e:
            self.logger.error(f"Research failed with exception: {e}")
            return False

        # Phase 2: Production (portfolio competition)
        production_cmd = [sys.executable, "-u", "ggt.py", "production"]
        self.logger.info(f"Running production: {' '.join(production_cmd)}")
        try:
            result = subprocess.run(
                production_cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            if result.returncode != 0:
                self.logger.error(
                    f"Production failed (exit {result.returncode}): {result.stderr[-500:]}"
                )
                return False
            self.logger.info("Production completed successfully.")
        except subprocess.TimeoutExpired:
            self.logger.error("Production timed out after 30 minutes.")
            return False
        except Exception as e:
            self.logger.error(f"Production failed with exception: {e}")
            return False

        # Phase 3: Reload the new params into this engine
        self.reload_params()
        self.logger.info("=" * 50)
        self.logger.info("  MONTHLY REOPTIMIZATION COMPLETE")
        self.logger.info(f"  Active symbols: {len(self.per_coin_params)}")
        self.logger.info("=" * 50)
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_event_loop(self) -> None:
        """Poll for new data every 4h and execute trades."""
        import signal
        import sys

        def _handle_shutdown(signum, frame):
            self.logger.info("Received shutdown signal — saving state and exiting.")
            self.save_state()
            self.state = "STOPPED"
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_shutdown)

        self.state = "EVENT_LOOP"
        self.logger.info("=========================================")
        self.logger.info(
            f"  ggTrader Live Execution Engine Started (DRY_RUN={self.config.get('DRY_RUN')})"
        )
        self.logger.info("=========================================")

        try:
            while True:
                # 0a. Monthly reoptimization on the 1st of each month
                from datetime import datetime as _dt

                _now = _dt.now()
                if _now.day == 1 and self._reopt_done_month != _now.month:
                    if self._run_monthly_reoptimization():
                        self._reopt_done_month = _now.month
                        self._save_reopt_flag(_now.month)
                    else:
                        self.logger.error(
                            "Monthly reoptimization failed — will retry next cycle"
                        )

                # 0b. Sync local memory with actual Kraken exchange (catch server-side OCO exits)
                self._reconcile_positions()

                # 0c. Record balance snapshot for performance tracking.
                # Bug E fix: skip the snapshot entirely if we couldn't fetch a
                # real total — never write a fabricated START_CASH value into
                # balance_snapshots.csv (it would poison every equity-curve
                # metric: max drawdown, Sharpe, Sortino, deposit-adjusted return).
                try:
                    total_usd = self._get_total_portfolio_usd()
                    if total_usd is None:
                        self.logger.warning(
                            "  [Tracker] Skipping balance snapshot this cycle "
                            "(portfolio valuation failed — see [Portfolio] log)"
                        )
                    else:
                        balance = self.exchange.fetch_balance()
                        free_usd = float(balance.get("free", {}).get("USD", 0))
                        positions_usd = total_usd - free_usd
                        self.tracker.record_balance_snapshot(
                            total_usd=total_usd, free_usd=free_usd,
                            positions_usd=positions_usd,
                            num_positions=len(self.active_positions),
                        )
                except Exception as e:
                    self.logger.warning(f"  [Tracker] Balance snapshot failed: {e!r}")

                # 1. Fetch live data (includes BTC for regime filter)
                latest_df = self._fetch_latest_data()

                if not latest_df.empty:
                    # 2. Compute regime allowance for this bar
                    regime_allowance = self._compute_live_regime_allowance(latest_df)

                    # 3. Compute signals per coin
                    latest_signals = self._compute_latest_signals(latest_df)

                    # 4. Execute / Manage trades
                    self._execute_trade_logic(latest_signals, regime_allowance)

                # 5. Wait for next 4h candle close, polling stop orders every N minutes
                now = time.gmtime()
                next_hour = ((now.tm_hour // 4) + 1) * 4
                if next_hour >= 24:
                    wait_seconds = (24 - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec
                else:
                    wait_seconds = (next_hour - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec

                # Buffer (2 minutes) to ensure candle data has fully settled
                wait_seconds += 120
                self.logger.info(
                    f"Loop complete. Sleeping {wait_seconds}s until next 4h candle "
                    f"(polling stop orders every {POLL_INTERVAL_SEC}s)."
                )

                # Sleep in chunks, polling open exit orders between chunks so that
                # stops triggered intra-candle get recorded immediately rather than
                # waiting up to 4h for the next reconcile.
                elapsed = 0
                while elapsed < wait_seconds:
                    chunk = min(POLL_INTERVAL_SEC, wait_seconds - elapsed)
                    time.sleep(max(chunk, 1))
                    elapsed += chunk
                    if elapsed < wait_seconds and self.active_positions:
                        try:
                            self._poll_open_exit_orders()
                        except Exception as e:
                            self.logger.warning(
                                f"  [Poll] interim exit-order poll failed: {e!r}"
                            )

        except KeyboardInterrupt:
            self.state = "STOPPED"
            self.logger.info("Execution Engine stopped by user.")
