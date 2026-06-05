"""Live crypto execution engine for Kraken/CCXT."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from ggTrader.utils.result_db_manager import ResultDBManager

from ggTrader.core.base_execution_engine import (
    BaseExecutionEngine,
    _format_entry_alert,
    _format_exit_alert,
)
from ggTrader.data.live.cached_loader import CachedExchangeLoader
from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.strategies import (
    AdxFilteredMeanReversionEntry,
    AtrTrailingExit,
    BollingerMeanReversionEntry,
    DonchianBreakoutEntry,
    EmaCrossEntry,
    FixedStopTakeProfit,
    KeltnerBreakoutEntry,
    MacdCrossEntry,
    MultiTimeframeMomentumEntry,
    PsarAdxEntry,
    RsiReversalEntry,
    StochRsiReversalEntry,
    SupertrendFlipEntry,
    TrailingStopExit,
)


def _safe_extract_filled_amount(order: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract cumulative filled quantity from CCXT order."""
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
    """Extract VWAP fill price from CCXT order response."""
    if not order:
        return None
    cost = order.get("cost")
    filled = order.get("filled")
    if cost and filled and float(cost) > 0 and float(filled) > 0:
        return float(cost) / float(filled)
    avg = order.get("average")
    if avg and float(avg) > 0:
        return float(avg)
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
    price = order.get("price")
    if price and float(price) > 0 and order.get("status") in ("closed", "filled"):
        return float(price)
    return None


_DUST_THRESHOLD_USD = 1.00
POLL_INTERVAL_SEC = 300


class CryptoExecutionEngine(BaseExecutionEngine):
    """Live crypto trading bot for Kraken/CCXT."""

    STRATEGY_MAP = {
        "psar_adx": PsarAdxEntry,
        "ema_cross": EmaCrossEntry,
        "mtf_momentum": MultiTimeframeMomentumEntry,
        "rsi_reversal": RsiReversalEntry,
        "adx_filtered_rsi": AdxFilteredMeanReversionEntry,
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
        config["ASSET_CLASS"] = "crypto"
        super().__init__(config, results_path, db_manager, run_id)

        self.exchange_id = config.get("EXCHANGE", "kraken")
        self.loader = CachedExchangeLoader(exchange_id=self.exchange_id)
        self.exchange = self.loader.exchange

        if not self.exchange.apiKey:
            if self.exchange_id == "binanceus":
                self.exchange.apiKey = os.getenv("BINANCE_API_LIVE_KEY")
                self.exchange.secret = os.getenv("BINANCE_SECRET_LIVE_KEY")
            else:
                self.exchange.apiKey = os.getenv("KRAKEN_KEY")
                self.exchange.secret = os.getenv("KRAKEN_SECRET")

        self._reopt_done_month: int | None = None
        self._reopt_state_key = f"last_reopt_month_{self.asset_class}"
        self._load_reopt_flag()

        if results_path:
            self.load_optimized_params(results_path)
            if not self.symbols:
                self.symbols = list(self.per_coin_params.keys())

        self.load_state()
        self._reconcile_positions()

    def _load_reopt_flag(self) -> None:
        if self.db_manager is None:
            self._reopt_done_month = None
            return
        val = self.db_manager.get_state(self._reopt_state_key)
        try:
            self._reopt_done_month = int(val) if val is not None else None
        except (TypeError, ValueError):
            self._reopt_done_month = None

    def _save_reopt_flag(self, month: int) -> None:
        if self.db_manager is None:
            self._reopt_done_month = month
            return
        try:
            self.db_manager.set_state(self._reopt_state_key, int(month))
            self._reopt_done_month = month
        except Exception as e:
            self.logger.warning(f"Could not save reopt flag: {e}")

    def reload_params(self) -> None:
        from ggTrader.utils.state_manager import get_latest_research_run

        latest_res = get_latest_research_run()
        if latest_res:
            self.per_coin_params = {}
            self.load_optimized_params(latest_res)
            self.symbols = list(self.per_coin_params.keys())
            self.results_path = latest_res
            self.logger.info(f"Reloaded research params from run {latest_res.run_id}")

    def _reconcile_positions(self) -> None:
        if self.config.get("DRY_RUN", False):
            return

        try:
            balance = self.exchange.fetch_balance()
            held_symbols: Dict[str, float] = {}
            for asset, amounts in balance.get("total", {}).items():
                if amounts and float(amounts) > 0 and asset not in ("USD", "USDT", "USDC"):
                    sym = f"{asset}-USD"
                    held_symbols[sym] = float(amounts)

            stale = [s for s in self.active_positions if s not in held_symbols]
            if stale:
                self.logger.warning(
                    f"  [Reconcile] {len(stale)} position(s) in local state but "
                    f"NOT held on exchange: {stale}"
                )
                for s in stale:
                    pos = self.active_positions[s]
                    exit_id = pos.get("tsl_order_id") or pos.get("oco_order_id")
                    reason = "trailing_stop" if pos.get("tsl_order_id") else "oco_exit"
                    if exit_id and exit_id not in ("dry_run_tsl_id", "dry_run_oco_id"):
                        if self._record_exit(s, exit_id, pos, reason):
                            del self.active_positions[s]
                        else:
                            pos["pending_repair"] = True
                self.save_state()

            untracked: list[str] = []
            for s in held_symbols:
                if s in self.active_positions:
                    continue
                # CCXT symbols use `BASE/QUOTE`; our internal symbol is `BASE-USD`.
                ccxt_sym = s.replace("-", "/")
                try:
                    ticker = self.exchange.fetch_ticker(ccxt_sym)
                    usd_val = held_symbols[s] * (ticker.get("last") or 0)
                except Exception:
                    usd_val = 0.0
                if usd_val < _DUST_THRESHOLD_USD:
                    self.logger.info(
                        f"  [Reconcile] Ignoring dust balance {s}: "
                        f"{held_symbols[s]:.8g} (~${usd_val:.4f})"
                    )
                    continue
                untracked.append(s)

            if untracked:
                self.logger.warning(
                    f"  [Reconcile] {len(untracked)} untracked positions found: {untracked}"
                )
                for s in untracked:
                    coin_params = self.per_coin_params.get(s)
                    seeded = self.tracker.find_open_buy_for(s)
                    self.active_positions[s] = {
                        "entry_order_id": seeded["order_id"] if seeded else "unknown",
                        "entry_price": seeded["price"] if seeded else None,
                        "entry_time": seeded["time"] if seeded else None,
                        "entry_fee": seeded["fee"] if seeded else 0.0,
                        "amount": held_symbols[s],
                        "stop_pct": coin_params["params"].get("stop_pct", 3.0)
                        if coin_params
                        else 3.0,
                        "exit_name": coin_params.get("exit_name", "atr_trailing")
                        if coin_params
                        else "atr_trailing",
                        "tsl_order_id": None,
                    }
                self.save_state()

            # Dust cleanup in local state
            for sym, pos in list(self.active_positions.items()):
                amt = float(pos.get("amount", 0) or 0)
                entry = float(pos.get("entry_price") or 0)
                if amt * entry < _DUST_THRESHOLD_USD:
                    try:
                        ticker = self.exchange.fetch_ticker(sym)
                        cur_val = amt * float(ticker.get("last") or 0)
                        if cur_val < _DUST_THRESHOLD_USD:
                            self.logger.info(f"  [Reconcile] Removing local dust position {sym}")
                            del self.active_positions[sym]
                    except Exception:
                        pass
            self.save_state()

        except Exception as e:
            self.logger.warning(f"  [Reconcile] Failed: {e!r}")

    def _poll_open_exit_orders(self) -> None:
        if self.config.get("DRY_RUN", False) or not self.active_positions:
            return

        triggered: list[str] = []
        for symbol, pos in list(self.active_positions.items()):
            exit_id = pos.get("tsl_order_id") or pos.get("oco_order_id")
            if not exit_id or exit_id in ("dry_run_tsl_id", "dry_run_oco_id"):
                continue
            try:
                pair = symbol.replace("-", "/")
                order = self.exchange.fetch_order(exit_id, pair)
                if order.get("status") in ("closed", "filled"):
                    reason = "trailing_stop" if pos.get("tsl_order_id") else "oco_exit"
                    if self._record_exit(symbol, exit_id, pos, reason, sell_order=order):
                        triggered.append(symbol)
                    else:
                        pos["pending_repair"] = True
            except Exception:
                continue

        for s in triggered:
            del self.active_positions[s]
        if triggered:
            self.save_state()

    def _record_exit(
        self,
        symbol: str,
        sell_order_id: str,
        pos: Dict[str, Any],
        exit_reason: str,
        sell_order: Optional[Dict[str, Any]] = None,
        allow_ticker_fallback: bool = False,
    ) -> bool:
        if sell_order_id in ("dry_run_sell_id", "dry_run_tsl_id", "dry_run_oco_id"):
            return False

        pair = symbol.replace("-", "/")
        if sell_order is None:
            try:
                time.sleep(1)
                sell_order = self.exchange.fetch_order(sell_order_id, pair)
            except Exception:
                return False

        exit_price = _safe_extract_fill_price(sell_order)
        if exit_price is None or exit_price <= 0:
            # Try fetch_my_trades
            try:
                recent = self.exchange.fetch_my_trades(pair, limit=20)
                matching = [t for t in recent if t.get("order") == sell_order_id]
                if matching:
                    exit_price = sum(
                        float(t["amount"]) * float(t["price"]) for t in matching
                    ) / sum(float(t["amount"]) for t in matching)
            except Exception:
                pass

        if (exit_price is None or exit_price <= 0) and allow_ticker_fallback:
            try:
                ticker = self.exchange.fetch_ticker(pair)
                exit_price = float(ticker.get("last") or 0)
            except Exception:
                pass

        if not exit_price or exit_price <= 0:
            return False

        filled_amount = _safe_extract_filled_amount(sell_order) or float(pos.get("amount") or 0)
        fee_info = sell_order.get("fee") or {}
        sell_fee = float(fee_info.get("cost", 0) or 0)

        self.tracker.record_sell(
            symbol=symbol,
            order_id=sell_order_id,
            price=exit_price,
            amount=filled_amount,
            amount_usd=exit_price * filled_amount,
            fee=sell_fee,
            fee_currency=fee_info.get("currency", "USD"),
            entry_price=pos.get("entry_price"),
            entry_time=pos.get("entry_time"),
            entry_fee=pos.get("entry_fee", 0),
            exit_reason=exit_reason,
        )

        try:
            strategy = self.per_coin_params.get(symbol, {}).get("strategy_name", "?")
            msg = _format_exit_alert(
                symbol,
                strategy,
                exit_reason,
                pos.get("entry_price"),
                exit_price,
                filled_amount,
                float(pos.get("entry_fee", 0)),
                sell_fee,
                pos.get("entry_time"),
                len(self.active_positions) - 1,
                len(self.per_coin_params),
            )
            self._notify(msg)
        except Exception:
            pass
        return True

    def _handle_emergency_rollback(self, symbol: str, amount: float) -> None:
        if self.config.get("DRY_RUN", False):
            self.active_positions.pop(symbol, None)
            return
        sell_id = self._execute_market_sell_order(symbol, amount)
        if sell_id:
            pos = self.active_positions.get(symbol) or {"amount": amount}
            if self._record_exit(
                symbol, sell_id, pos, "emergency_rollback", allow_ticker_fallback=True
            ):
                self.active_positions.pop(symbol, None)
            elif symbol in self.active_positions:
                self.active_positions[symbol]["pending_repair"] = True

    def _get_total_portfolio_usd(self) -> Optional[float]:
        if self.config.get("DRY_RUN", False):
            return float(self.config.get("START_CASH", 1000.0))
        try:
            bal = self.exchange.fetch_balance()
            total_usd = 0.0
            tickers = self.exchange.fetch_tickers()
            norm_map = {"XBT": "BTC", "XETH": "ETH", "XXLM": "XLM", "ZUSD": "USD"}
            # Restrict valuation to the configured crypto universe so co-located
            # holdings on the same Kraken account (notably tokenized xStocks
            # like AAPLx/USD) don't inflate the crypto portfolio total.
            allowed_bases = {s.split("-")[0] for s in self.symbols}
            allowed_bases.update(s.split("-")[0] for s in self.active_positions.keys())
            for coin, amount in bal.get("total", {}).items():
                if float(amount) <= 0:
                    continue
                norm_coin = norm_map.get(coin, coin)
                if norm_coin in ("USD", "USDT", "USDC"):
                    total_usd += float(amount)
                    continue
                if norm_coin not in allowed_bases:
                    continue
                pair = f"{norm_coin}/USD"
                if pair in tickers:
                    total_usd += float(amount) * float(tickers[pair]["last"])
            return total_usd
        except Exception:
            return None

    def _fetch_latest_data(self) -> pd.DataFrame:
        return self.loader.fetch_ohlcv(
            symbols=list(self.symbols), interval=self.interval, limit=200
        )

    def _compute_latest_signals(self, ohlcv_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        results = {}
        for s in self.symbols:
            if s not in ohlcv_df.columns.get_level_values(0):
                continue
            info = self.per_coin_params.get(s)
            if not info:
                continue
            strat = self.STRATEGY_MAP.get(info["strategy_name"])
            exit_s = self.EXIT_MAP.get(info["exit_name"])
            if not strat or not exit_s:
                continue

            close, high, low = ohlcv_df[(s, "close")], ohlcv_df[(s, "high")], ohlcv_df[(s, "low")]
            pc = IndicatorPrecomputer(close, high, low)
            grid = {k: [v] for k, v in info["params"].items()}
            ent, _ = strat().compute_entries(pc, grid)
            ext, stops, pxs = exit_s().compute_exits(ent, pc, grid, 1)

            atr = float("nan")
            if info["exit_name"] == "atr_trailing":
                try:
                    atr = float(
                        pc.compute_atr(int(info["params"].get("atr_length", 14))).atrr.values[-1]
                    )
                except Exception:
                    pass

            results[s] = {
                "entry": bool(ent[-1, 0]),
                "exit": bool(ext[-1, 0]),
                "exit_name": info["exit_name"],
                "stop_price": float(stops[-1, 0]),
                "fill_price": float(pxs[-1, 0]),
                "current_price": float(close.iloc[-1]),
                "atr_value": atr,
            }
        return results

    def _validate_entry_preconditions(self, symbol: str, capital_usd: float) -> bool:
        if self.config.get("DRY_RUN"):
            return True
        try:
            bal = self.exchange.fetch_balance()
            free_usd = float(bal.get("free", {}).get("USD", 0))
            if free_usd < capital_usd * 0.95:
                self.logger.warning(
                    f"[Precondition] {symbol}: insufficient USD balance "
                    f"(free=${free_usd:.2f}, need~${capital_usd * 0.95:.2f})"
                )
                return False
            pair = symbol.replace("-", "/")
            if not self.exchange.markets:
                self.exchange.load_markets()
            m = self.exchange.market(pair)
            ticker = self.exchange.fetch_ticker(pair)
            coin_amt = capital_usd / ticker["last"]
            min_amt = float(m.get("limits", {}).get("amount", {}).get("min") or 0)
            if coin_amt < min_amt:
                self.logger.warning(
                    f"[Precondition] {symbol}: order amount {coin_amt:.6f} below "
                    f"exchange min {min_amt} (capital=${capital_usd:.2f}, last={ticker['last']})"
                )
                return False
            return True
        except Exception as e:
            self.logger.warning(f"[Precondition] {symbol}: validation raised — {e!r}")
            return False

    def _execute_market_buy_order(self, symbol: str, amount_usd: float) -> Optional[str]:
        self.logger.info(f"BUY {symbol} ${amount_usd}")
        if self.config.get("DRY_RUN"):
            return "dry_run_id"
        try:
            pair = symbol.replace("-", "/")
            ticker = self.exchange.fetch_ticker(pair)
            amt = self.exchange.amount_to_precision(pair, amount_usd / ticker["last"])
            order = self.exchange.create_market_buy_order(pair, amt)
            return order["id"]
        except Exception as e:
            self.logger.error(f"Buy failed: {e}")
            return None

    def _execute_market_sell_order(self, symbol: str, amount: float) -> Optional[str]:
        self.logger.info(f"SELL {symbol} {amount}")
        if self.config.get("DRY_RUN"):
            return "dry_run_sell_id"
        try:
            pair = symbol.replace("-", "/")
            amt = self.exchange.amount_to_precision(pair, amount)
            order = self.exchange.create_market_sell_order(pair, amt)
            return order["id"]
        except Exception as e:
            self.logger.error(f"Sell failed: {e}")
            return None

    def _execute_trailing_stop_order(
        self, symbol: str, amount: float, stop_pct: float
    ) -> Optional[str]:
        self.logger.info(f"TSL {symbol} -{stop_pct}%")
        if self.config.get("DRY_RUN"):
            return "dry_run_tsl_id"
        try:
            pair = symbol.replace("-", "/")
            ticker = self.exchange.fetch_ticker(pair)
            offset = ticker["last"] * (stop_pct / 100.0)
            m = self.exchange.market(pair)
            prec = m.get("precision", {}).get("price")
            min_t = 10 ** (-prec) if isinstance(prec, int) else prec
            offset = float(self.exchange.price_to_precision(pair, max(offset, min_t)))
            amt_p = self.exchange.amount_to_precision(pair, amount)
            order = self.exchange.create_order(
                pair, "trailing-stop", "sell", amt_p, None, {"trailingAmount": offset}
            )
            return order["id"]
        except Exception as e:
            self.logger.error(f"TSL failed: {e}")
            return None

    def _execute_oco_exit_order(
        self, symbol: str, amount: float, sl: float, tp: float
    ) -> Optional[str]:
        self.logger.info(f"OCO {symbol} SL={sl} TP={tp}")
        if self.config.get("DRY_RUN"):
            return "dry_run_oco_id"
        try:
            pair = symbol.replace("-", "/")
            sl_f = float(self.exchange.price_to_precision(pair, sl))
            tp_f = float(self.exchange.price_to_precision(pair, tp))
            amt_p = self.exchange.amount_to_precision(pair, amount)
            order = self.exchange.create_order(
                pair, "limit", "sell", amt_p, tp_f, {"stopLossPrice": sl_f, "takeProfitPrice": tp_f}
            )
            return order["id"]
        except Exception as e:
            self.logger.error(f"OCO failed: {e}")
            return None

    def _cancel_open_orders(self, symbol: str) -> None:
        if self.config.get("DRY_RUN"):
            return
        try:
            pair = symbol.replace("-", "/")
            for o in self.exchange.fetch_open_orders(pair):
                self.exchange.cancel_order(o["id"], pair)
        except Exception:
            pass

    def _estimate_stop_pct_for_sizing(self, symbol: str, sig: Dict[str, Any]) -> float:
        """Pre-buy estimate of the exit's stop distance as a percentage.

        Mirrors the post-buy stop computation so adaptive sizing picks a
        position consistent with the eventual stop. Floor of 0.5% to avoid
        blowing the cap when ATR is transiently tiny.
        """
        coin_params = self.per_coin_params[symbol]["params"]
        exit_name = sig.get("exit_name", "atr_trailing")
        if exit_name == "atr_trailing":
            atr_value = sig.get("atr_value", float("nan"))
            atr_mult = float(coin_params.get("atr_multiplier", 3.0))
            price = float(sig.get("current_price", 0) or 0)
            if not math.isnan(atr_value) and atr_value > 0 and price > 0:
                stop_pct = (atr_mult * atr_value / price) * 100.0
            else:
                stop_pct = atr_mult
        elif exit_name == "trailing_stop":
            stop_pct = float(coin_params.get("trailing_stop_pct", 3.0))
        else:
            stop_pct = float(coin_params.get("stop_pct", 3.0))
        return max(stop_pct, 0.5)

    def _compute_post_buy_stop_pct(
        self, symbol: str, sig: Dict[str, Any], fill_price: float
    ) -> float:
        """Final stop-pct for placing the actual exit order after a fill.

        Thin wrapper around ``core.trailing_stop_utils.compute_post_buy_stop_pct``.
        """
        from ggTrader.core.trailing_stop_utils import compute_post_buy_stop_pct

        coin_params = self.per_coin_params[symbol]["params"]
        return compute_post_buy_stop_pct(
            symbol=symbol,
            coin_params=coin_params,
            exit_name=sig.get("exit_name", "atr_trailing"),
            fill_price=fill_price,
            atr_value=sig.get("atr_value", float("nan")),
            config=self.config,
            logger=self.logger,
        )

    def _compute_adaptive_position_usd(
        self, symbol: str, sig: Dict[str, Any], base_capital: float
    ) -> Optional[float]:
        """Volatility-normalized position size:
        position = (portfolio * TARGET_RISK_PCT) / stop_pct.
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
                f"(stop≈{stop_pct:.2f}%, risk {target_risk_pct * 100:.2f}% of "
                f"${base_capital:.2f}) below MIN_POSITION_USD=${min_position_usd:.2f} — skipping"
            )
            return None

        capped = raw_position > cap_position
        self.logger.info(
            f"  [Sizing] {symbol}: adaptive (risk {target_risk_pct * 100:.2f}% @ "
            f"stop≈{stop_pct:.2f}%) -> ${position_usd:.2f}"
            + (f"  [CAPPED at {max_position_pct * 100:.0f}% of portfolio]" if capped else "")
        )
        return position_usd

    def _execute_trade_logic(
        self,
        signals_dict: Dict[str, Dict[str, Any]],
    ) -> None:
        if self.circuit_breaker_triggered:
            self.logger.info("  [CircuitBreaker] ACTIVE")
            return

        weighted = bool(self.config.get("WEIGHTED_SIZING", False))
        adaptive = bool(self.config.get("ADAPTIVE_SIZING", False))
        min_position_usd = float(self.config.get("MIN_POSITION_USD", 15.0))

        for s, sig in signals_dict.items():
            if sig["entry"] and s not in self.active_positions:
                if weighted:
                    base_capital = self._get_total_portfolio_usd()
                    if base_capital is None:
                        self.logger.warning(
                            f"  [Sizing] {s}: portfolio valuation unavailable — skipping entry"
                        )
                        continue
                    weight = float(self.allocation_weights.get(s, 0.0))
                    if weight <= 0.0:
                        self.logger.info(
                            f"  [Sizing] {s}: research allocation weight = 0% — skipping entry"
                        )
                        continue
                    cap = base_capital * weight
                    if cap < min_position_usd:
                        self.logger.info(
                            f"  [Sizing] {s}: weighted ${cap:.2f} "
                            f"({weight * 100:.2f}% × ${base_capital:.2f}) "
                            f"below MIN_POSITION_USD=${min_position_usd:.2f} — skipping"
                        )
                        continue
                    self.logger.info(
                        f"  [Sizing] {s}: weighted {weight * 100:.2f}% × "
                        f"${base_capital:.2f} -> ${cap:.2f}"
                    )
                elif adaptive:
                    base_capital = self._get_total_portfolio_usd()
                    if base_capital is None:
                        self.logger.warning(
                            f"  [Sizing] {s}: portfolio valuation unavailable — skipping entry"
                        )
                        continue
                    cap = self._compute_adaptive_position_usd(s, sig, base_capital)
                    if cap is None:
                        continue
                else:
                    cap = self.config.get("CAPITAL_PER_TRADE")
                    if cap is None:
                        cap = 100.0
                    cap = float(cap)
                    self.logger.info(f"  [Sizing] {s}: fixed ${cap:.2f}")

                if not self._validate_entry_preconditions(s, cap):
                    continue
                oid = self._execute_market_buy_order(s, cap)
                if oid:
                    time.sleep(1)
                    ord = self.exchange.fetch_order(oid, s.replace("-", "/"))
                    px = _safe_extract_fill_price(ord) or sig["current_price"]
                    amt = float(ord.get("filled", 0)) or (cap / px)
                    self.tracker.record_buy(s, oid, px, amt, px * amt, 0, "USD")

                    exit_n = sig["exit_name"]
                    stop_p = self._compute_post_buy_stop_pct(s, sig, px)
                    # Universal trailing: every live exit uses Kraken-native trailing-stop.
                    # Legacy fixed_sl_tp WFO selections (from production_weights predating
                    # the EXIT_TOURNAMENT trim) get converted at placement time so the
                    # take-profit leg is dropped and the stop ratchets up with price.
                    if exit_n == "fixed_sl_tp":
                        self.logger.info(
                            f"  [Exit] {s} fixed_sl_tp converted to trailing-stop "
                            f"at placement (stop_pct={stop_p:.2f}%)"
                        )
                    tslid = self._execute_trailing_stop_order(s, amt, stop_p)
                    self.active_positions[s] = {
                        "entry_order_id": oid,
                        "entry_price": px,
                        "entry_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "amount": amt,
                        "stop_pct": stop_p,
                        "exit_name": "trailing_stop" if exit_n == "fixed_sl_tp" else exit_n,
                        "exit_name_original": exit_n,
                        "tsl_order_id": tslid,
                    }
                    self.save_state()
                    try:
                        msg = _format_entry_alert(
                            s,
                            self.per_coin_params.get(s, {}).get("strategy_name", "?"),
                            self.active_positions[s]["exit_name"],
                            px,
                            amt,
                            px * amt,
                            stop_p,
                            len(self.active_positions),
                            len(self.per_coin_params),
                        )
                        self._notify(msg)
                    except Exception:
                        pass
            elif s in self.active_positions and sig["exit"]:
                pos = self.active_positions[s]
                if pos.get("exit_name") == "fixed_sl_tp":
                    self._cancel_open_orders(s)
                    sid = self._execute_market_sell_order(s, pos["amount"])
                    if sid and self._record_exit(
                        s, sid, pos, "strategy_signal", allow_ticker_fallback=True
                    ):
                        del self.active_positions[s]
                        self.save_state()

    def run_event_loop(self) -> None:
        self.state = "EVENT_LOOP"
        self.logger.info("ggTrader Crypto Engine Started")
        try:
            while True:
                now = datetime.now(timezone.utc)
                if now.day == 1 and self._reopt_done_month != now.month:
                    if self._run_monthly_reoptimization():
                        self._reopt_done_month = now.month
                        self._save_reopt_flag(now.month)

                self._reconcile_positions()
                self._check_circuit_breaker()

                total_usd = self._get_total_portfolio_usd()
                if total_usd:
                    try:
                        bal = self.exchange.fetch_balance()
                        free_usd = float(bal.get("free", {}).get("USD", 0))
                        self.tracker.record_balance_snapshot(
                            total_usd, free_usd, total_usd - free_usd, len(self.active_positions)
                        )
                    except Exception:
                        pass
                    try:
                        self.db_manager.add_positions_snapshot(
                            asset_class=self.asset_class,
                            timestamp=datetime.now(timezone.utc),
                            positions=self.active_positions,
                            circuit_breaker_triggered=self.circuit_breaker_triggered,
                        )
                    except Exception:
                        pass

                df = self._fetch_latest_data()
                if not df.empty:
                    self._execute_trade_logic(self._compute_latest_signals(df))

                time.sleep(POLL_INTERVAL_SEC)
                if self.active_positions:
                    self._poll_open_exit_orders()
        except KeyboardInterrupt:
            self.state = "STOPPED"

    def _run_monthly_reoptimization(self) -> bool:
        import subprocess
        import sys
        from datetime import datetime, timedelta

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.logger.info("  MONTHLY REOPTIMIZATION STARTED")
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "ggt.py",
                    "research",
                    "--top",
                    "50",
                    # Volume floor matches the manual research default. On Binance.US only
                    # ~11 USD pairs clear $50K/30d; --top is just a safety cap above the floor.
                    "--min-volume",
                    "50000",
                    "--workers",
                    "2",
                    "--end-date",
                    yesterday,
                ],
                cwd=str(project_root),
                check=True,
                timeout=7200,
            )
            subprocess.run(
                [sys.executable, "-u", "ggt.py", "production"],
                cwd=str(project_root),
                check=True,
                timeout=1800,
            )
            self.reload_params()
            return True
        except Exception as e:
            self.logger.error(f"Monthly reopt failed: {e}")
            return False
