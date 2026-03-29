"""Live execution engine for transition from optimization to real-world trading."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ggTrader.core.regime_filtering import (
    _compute_altcoin_index_mask,
    _compute_btc_correlations,
    _compute_btc_regime_mask,
)
from ggTrader.data.live.cached_loader import CachedExchangeLoader
from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.strategies import (
    AtrTrailingExit,
    BollingerMeanReversionEntry,
    DonchianBreakoutEntry,
    EmaCrossEntry,
    FixedStopTakeProfit,
    MacdCrossEntry,
    PsarAdxEntry,
    RsiReversalEntry,
    SupertrendFlipEntry,
    TrailingStopExit,
)

_BTC_SYMBOL = "BTC-USD"


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
    }

    EXIT_MAP = {
        "atr_trailing": AtrTrailingExit,
        "fixed_sl_tp": FixedStopTakeProfit,
        "trailing_stop": TrailingStopExit,
    }

    def __init__(self, config: Dict[str, Any], results_path: Optional[str] = None):
        """Initialize engine.

        If results_path is provided, it loads per-coin optimized parameters.
        Otherwise, it uses the global params in config.
        """
        self.config = config
        self.results_path = results_path
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
        self.portfolio_weights = {}
        self.persistence_path = config.get("PERSISTENCE_PATH", "data/active_positions.json")
        self.weights_path = config.get("WEIGHTS_PATH")

        if results_path:
            self.load_optimized_params(results_path)
            # Override symbols if symbols not provided in config
            if not self.symbols:
                self.symbols = list(self.per_coin_params.keys())

        if self.weights_path:
            self.load_portfolio_weights(self.weights_path)

        self.load_state()
        self._reconcile_positions()

    # ------------------------------------------------------------------
    # Parameter and state loading
    # ------------------------------------------------------------------

    def load_optimized_params(self, json_path: str) -> None:
        """Load per-coin parameters from WFO run_results.json.

        Applies risk-control gates from config:
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

        strategy_counts: Dict[str, int] = defaultdict(int)
        dropped_low_rob: List[str] = []
        dropped_strategy_cap: List[str] = []

        for symbol, result in per_coin.items():
            best_params = result.get("best_params", {})
            strategy = result.get("best_strategy")
            exit_strategy = result.get("best_exit")
            robustness = float(result.get("robustness_score", 0.0))

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

    def load_portfolio_weights(self, json_path: str) -> None:
        """Load capital allocation weights from portfolio_weights.json."""
        if not os.path.exists(json_path):
            self.logger.warning(f"Weights file {json_path} not found.")
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            self.portfolio_weights = data.get("weights", {})
            # Apply MAX_COIN_ALLOCATION cap
            max_alloc = float(self.config.get("MAX_COIN_ALLOCATION", 1.0))
            if max_alloc < 1.0:
                capped = {s: min(w, max_alloc) for s, w in self.portfolio_weights.items()}
                n_capped = sum(1 for s, w in self.portfolio_weights.items() if w > max_alloc)
                if n_capped:
                    self.logger.info(
                        f"  [Gates] Capped {n_capped} weight(s) to "
                        f"MAX_COIN_ALLOCATION={max_alloc:.0%}"
                    )
                self.portfolio_weights = capped
            self.logger.info(f"Loaded portfolio weights for {len(self.portfolio_weights)} symbols.")
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")

    def load_state(self) -> None:
        """Load active positions from persistence file."""
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, "r") as f:
                self.active_positions = json.load(f)
            self.logger.info(f"Loaded {len(self.active_positions)} active positions from state.")

    def save_state(self) -> None:
        """Save active positions to persistence file."""
        dir_name = os.path.dirname(self.persistence_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(self.active_positions, f, indent=4)

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
                self.logger.info("  [Reconcile] Removing stale positions from local state.")
                for s in stale:
                    del self.active_positions[s]
                self.save_state()

            # Balances on exchange but not tracked in JSON (crash recovery)
            untracked = [s for s in held_symbols if s in self.per_coin_params
                         and s not in self.active_positions]
            if untracked:
                self.logger.warning(f"  [Reconcile] {len(untracked)} position(s) held on exchange "
                      f"but NOT in local state — adding as untracked: {untracked}")
                for s in untracked:
                    self.active_positions[s] = {
                        "entry_order_id": "unknown_reconciled",
                        "entry_price": None,
                        "amount": held_symbols[s],
                        "stop_pct": self.per_coin_params[s]["params"].get("stop_pct", 3.0),
                        "tsl_order_id": None,
                    }
                self.save_state()

            if not stale and not untracked:
                self.logger.info(
                    f"  [Reconcile] State verified: "
                    f"{len(self.active_positions)} position(s) match exchange."
                )

        except Exception as e:
            self.logger.warning(
                f"  [Reconcile] Exchange reconciliation failed "
                f"({e!r}) — using local state."
            )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_total_portfolio_usd(self) -> float:
        """Calculate total USD value of the portfolio (Free USD + held crypto)."""
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
            self.logger.warning(f"  [Portfolio] Could not calculate total portfolio USD: {e}")
            return float(self.config.get("START_CASH", 1000.0))

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

            signals_results[symbol] = {
                "entry": bool(entries[-1, 0]),
                "exit": bool(exits[-1, 0]),
                "exit_name": exit_name,
                "stop_price": float(stops[-1, 0]),
                "fill_price": float(prices[-1, 0]),
                "current_price": float(close.iloc[-1]),
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
            self._ensure_markets_loaded()
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
            self.logger.warning(
                f"  [Order] Precondition check failed for {symbol} ({e!r}) — allowing"
            )
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
        """Place a trailing stop loss order on Kraken."""
        self.logger.info(f"PLACING TRAILING STOP: {amount} {symbol} at -{stop_pct}%")
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: Order skipped.")
            return "dry_run_tsl_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)
            params = {"ordertype": "trailing-stop", "trailing_amount": f"{stop_pct}%"}
            order = self.exchange.create_order(
                market["symbol"], "trailing-stop", "sell", amount, None, params
            )
            return order["id"]
        except Exception as e:
            self.logger.error(f"Error placing trailing stop for {symbol}: {e}")
            return None

    def _execute_oco_exit_order(
        self, symbol: str, amount: float, sl_price: float, tp_price: float
    ) -> Optional[str]:
        """Place an OCO (stop-loss-profit) order on Kraken."""
        self.logger.info(
            f"PLACING OCO EXIT: {amount} {symbol} SL={sl_price:.4f}, TP={tp_price:.4f}"
        )
        if self.config.get("DRY_RUN", False):
            self.logger.info("DRY RUN: OCO order skipped.")
            return "dry_run_oco_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)

            sl_price_str = self.exchange.price_to_precision(market["symbol"], sl_price)
            tp_price_str = self.exchange.price_to_precision(market["symbol"], tp_price)

            # Kraken accepts price2 as the take profit limit
            params = {"price2": tp_price_str}
            order = self.exchange.create_order(
                market["symbol"], "stop-loss-profit", "sell", amount, sl_price_str, params
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
                print(f"  [Exit] Cancelled order {o['id']} for {symbol}")
        except Exception as e:
            print(f"  [Exit] WARNING: could not cancel orders for {symbol}: {e!r}")

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
                    print(f"  [Regime] BLOCKED entry for {symbol} (bear regime)")
                    continue

                weight = self.portfolio_weights.get(symbol)
                if weight is not None:
                    base_capital = self._get_total_portfolio_usd()
                    capital_per_trade = base_capital * float(weight)
                    print(f"Using dynamic weight for {symbol}: {weight * 100:.1f}% of total "
                          f"portfolio (${base_capital:.2f}) -> {capital_per_trade:.2f} USD")
                else:
                    capital_per_trade = self.config.get("CAPITAL_PER_TRADE", 100.0)
                    print(f"Using fixed capital for {symbol}: {capital_per_trade:.2f} USD")

                # Pre-flight: balance and min-order-size checks
                if not self._validate_entry_preconditions(symbol, capital_per_trade):
                    continue

                order_id = self._execute_market_buy_order(symbol, capital_per_trade)

                if order_id:
                    time.sleep(1)  # Brief wait for fill
                    try:
                        order = self.exchange.fetch_order(
                            order_id,
                            symbol.replace("-", "/") if "/" not in symbol else symbol,
                        )
                        filled_amount = order.get("filled", 0.0)
                    except Exception:
                        ticker = self.exchange.fetch_ticker(
                            symbol.replace("-", "/") if "/" not in symbol else symbol
                        )
                        filled_amount = capital_per_trade / ticker["last"]

                    stop_pct = self.per_coin_params[symbol]["params"].get("stop_pct", 3.0)
                    self.active_positions[symbol] = {
                        "entry_order_id": order_id,
                        "entry_price": sig["current_price"],
                        "entry_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "amount": filled_amount,
                        "stop_pct": stop_pct,
                        "exit_name": sig["exit_name"],
                        "tsl_order_id": None,
                    }

                    # Place TSL for atr_trailing / trailing_stop exits.
                    # fixed_sl_tp is handled natively by Kraken OCO parameters in CCXT.
                    exit_name = sig["exit_name"]
                    if exit_name == "fixed_sl_tp":
                        tp_pct = self.per_coin_params[symbol]["params"].get("take_profit_pct", 6.0)
                        fill_price = sig["current_price"]
                        # attempt to use actual fill price if available
                        if "order" in locals() and "average" in order and order["average"]:
                            fill_price = float(order["average"])

                        sl_price = fill_price * (1.0 - (stop_pct / 100.0))
                        tp_price = fill_price * (1.0 + (tp_pct / 100.0))

                        oco_id = self._execute_oco_exit_order(
                            symbol, filled_amount, sl_price, tp_price
                        )
                        if oco_id is None:
                            print(f"  [Order] OCO placement failed for {symbol} — closing position")
                            self._execute_market_sell_order(symbol, filled_amount)
                            del self.active_positions[symbol]
                        else:
                            self.active_positions[symbol]["oco_order_id"] = oco_id
                    else:
                        tsl_id = self._execute_trailing_stop_order(symbol, filled_amount, stop_pct)
                        if tsl_id is None:
                            # TSL failed — close immediately to avoid unprotected position
                            print(f"  [Order] TSL placement failed for {symbol} — closing position")
                            self._execute_market_sell_order(symbol, filled_amount)
                            del self.active_positions[symbol]
                        else:
                            self.active_positions[symbol]["tsl_order_id"] = tsl_id

                    self.save_state()

            elif symbol in self.active_positions:
                if sig["exit"]:
                    exit_name = self.active_positions[symbol].get("exit_name", sig["exit_name"])
                    if exit_name == "fixed_sl_tp":
                        # fixed_sl_tp: strategy computes the exit condition — execute it
                        print(f"[{time.strftime('%X')}] STRATEGY EXIT (fixed_sl_tp) FOR {symbol}")
                        amount = self.active_positions[symbol]["amount"]
                        self._cancel_open_orders(symbol)
                        sell_id = self._execute_market_sell_order(symbol, amount)
                        if sell_id:
                            self.active_positions[symbol]["exit_time"] = time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            )
                            self.active_positions[symbol]["exit_reason"] = "strategy_signal"
                            del self.active_positions[symbol]
                            self.save_state()
                    else:
                        # atr_trailing / trailing_stop: Kraken's TSL handles the exit
                        print(f"  [Exit] Strategy exit signal for {symbol} "
                              f"(exit={exit_name}) — TSL is active, no action needed")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_event_loop(self) -> None:
        """Poll for new data every 4h and execute trades."""
        self.state = "EVENT_LOOP"
        self.logger.info("=========================================")
        self.logger.info(
            f"  ggTrader Live Execution Engine Started (DRY_RUN={self.config.get('DRY_RUN')})"
        )
        self.logger.info("=========================================")

        try:
            while True:
                # 0. Sync local memory with actual Kraken exchange (catch server-side OCO exits)
                self._reconcile_positions()

                # 1. Fetch live data (includes BTC for regime filter)
                latest_df = self._fetch_latest_data()

                if not latest_df.empty:
                    # 2. Compute regime allowance for this bar
                    regime_allowance = self._compute_live_regime_allowance(latest_df)

                    # 3. Compute signals per coin
                    latest_signals = self._compute_latest_signals(latest_df)

                    # 4. Execute / Manage trades
                    self._execute_trade_logic(latest_signals, regime_allowance)

                # 5. Wait for next 4h candle close
                now = time.gmtime()
                next_hour = ((now.tm_hour // 4) + 1) * 4
                if next_hour >= 24:
                    wait_seconds = (24 - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec
                else:
                    wait_seconds = (next_hour - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec

                # Buffer (2 minutes) to ensure candle data has fully settled
                wait_seconds += 120
                self.logger.info(f"Loop complete. Sleeping {wait_seconds}s until next 4h candle.")
                time.sleep(max(wait_seconds, 60))

        except KeyboardInterrupt:
            self.state = "STOPPED"
            self.logger.info("Execution Engine stopped by user.")
