"""Live execution engine for transition from optimization to real-world trading."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import pandas as pd

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
)


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
    }

    def __init__(self, config: Dict[str, Any], results_path: Optional[str] = None):
        """Initialize engine.

        If results_path is provided, it loads per-coin optimized parameters.
        Otherwise, it uses the global params in config.
        """
        self.config = config
        self.results_path = results_path
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

    def load_optimized_params(self, json_path: str) -> None:
        """Load per-coin parameters from WFO run_results.json."""
        if not os.path.exists(json_path):
            print(f"Warning: Results file {json_path} not found.")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        per_coin = data.get("per_coin_results", {})
        for symbol, result in per_coin.items():
            best_params = result.get("best_params", {})
            strategy = result.get("best_strategy")
            exit_strategy = result.get("best_exit")

            self.per_coin_params[symbol] = {
                "strategy_name": strategy,
                "exit_name": exit_strategy,
                "params": best_params,
            }
        print(f"Loaded optimized parameters for {len(self.per_coin_params)} symbols.")

    def load_portfolio_weights(self, json_path: str) -> None:
        """Load capital allocation weights from portfolio_weights.json."""
        if not os.path.exists(json_path):
            print(f"Warning: Weights file {json_path} not found.")
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            self.portfolio_weights = data.get("weights", {})
            print(f"Loaded portfolio weights for {len(self.portfolio_weights)} symbols.")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def load_state(self) -> None:
        """Load active positions from persistence file."""
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, "r") as f:
                self.active_positions = json.load(f)
            print(f"Loaded {len(self.active_positions)} active positions from state.")

    def save_state(self) -> None:
        """Save active positions to persistence file."""
        dir_name = os.path.dirname(self.persistence_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(self.active_positions, f, indent=4)

    def _fetch_latest_data(self) -> pd.DataFrame:
        """Fetch the most recent candles for calculating signals."""
        lookback_limit = self.config.get("LOOKBACK_LIMIT", 200)
        return self.loader.fetch_ohlcv(
            symbols=self.symbols, interval=self.interval, limit=lookback_limit
        )

    def _compute_latest_signals(self, ohlcv_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute signals for each coin using its unique strategy and parameters."""
        signals_results = {}

        for symbol in self.symbols:
            if symbol not in ohlcv_df.columns.levels[0]:
                continue

            coin_info = self.per_coin_params.get(symbol)
            if not coin_info:
                continue

            strat_name = coin_info["strategy_name"]
            exit_name = coin_info["exit_name"]
            params = coin_info["params"]

            # Setup Entry Strategy
            strat_cls = self.STRATEGY_MAP.get(strat_name)
            if not strat_cls:
                continue
            strat_inst = strat_cls()

            # Pass parameters as grid (as per the EntryStrategy protocol)
            param_grid = {k: [v] for k, v in params.items()}

            # Isolate this symbol's OHLCV for precomputation
            symbol_df = ohlcv_df[[symbol]]
            close = symbol_df[(symbol, "close")]
            high = symbol_df[(symbol, "high")]
            low = symbol_df[(symbol, "low")]
            symbol_precomputer = IndicatorPrecomputer(close, high, low)

            entries, _ = strat_inst.compute_entries(symbol_precomputer, param_grid)

            # Setup Exit Strategy
            exit_cls = self.EXIT_MAP.get(exit_name)
            if not exit_cls:
                continue
            exit_inst = exit_cls()

            # Compute Exits
            exits, stops, prices = exit_inst.compute_exits(
                entries, symbol_precomputer, param_grid, 1
            )

            # Grab the last timestamp's result
            signals_results[symbol] = {
                "entry": bool(entries[-1, 0]),
                "exit": bool(exits[-1, 0]),
                "stop_price": float(stops[-1, 0]),
                "fill_price": float(prices[-1, 0]),
                "current_price": float(symbol_df[symbol]["close"].iloc[-1]),
            }

        return signals_results

    def _execute_market_buy_order(self, symbol: str, amount_usd: float) -> Optional[str]:
        """Execute a market buy order via CCXT."""
        print(f"[{time.strftime('%X')}] EXECUTING MARKET BUY: {amount_usd} USD for {symbol}")
        if self.config.get("DRY_RUN", False):
            print("DRY RUN: Order skipped.")
            return "dry_run_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            ticker = self.exchange.fetch_ticker(pair)
            price = ticker["last"]
            base_amount = amount_usd / price

            # Adjust to lot_size / precision
            market = self.exchange.market(pair)
            base_amount_precise = self.exchange.amount_to_precision(market["symbol"], base_amount)

            order = self.exchange.create_market_buy_order(market["symbol"], base_amount_precise)
            return order["id"]
        except Exception as e:
            print(f"Error placing market buy for {symbol}: {e}")
            return None

    def _execute_trailing_stop_order(
        self, symbol: str, amount: float, stop_pct: float
    ) -> Optional[str]:
        """Place a trailing stop loss order on Kraken."""
        print(f"[{time.strftime('%X')}] PLACING TRAILING STOP: {amount} {symbol} at -{stop_pct}%")
        if self.config.get("DRY_RUN", False):
            print("DRY RUN: Order skipped.")
            return "dry_run_tsl_id"

        try:
            pair = symbol.replace("-", "/") if "/" not in symbol else symbol
            market = self.exchange.market(pair)
            # Kraken trailing stop requires params with 'trailing_amount'
            params = {"ordertype": "trailing-stop", "trailing_amount": f"{stop_pct}%"}
            order = self.exchange.create_order(
                market["symbol"], "trailing-stop", "sell", amount, None, params
            )
            return order["id"]
        except Exception as e:
            print(f"Error placing trailing stop for {symbol}: {e}")
            return None

    def _execute_trade_logic(self, signals_dict: Dict[str, Dict[str, Any]]) -> None:
        """Evaluate signals and execute orders."""
        for symbol, sig in signals_dict.items():
            if sig["entry"] and symbol not in self.active_positions:
                # Handle Entry
                # Priority: 1. Dynamic Weight * Total Equity, 2. Fixed Capital
                weight = self.portfolio_weights.get(symbol)
                if weight is not None:
                    # If we have weights, they are percentages (e.g. 0.1 for 10%)
                    # We multiply by START_CASH or current balance if we had it.
                    # For now, we use START_CASH as the total 'Bankroll' for the engine.
                    base_capital = self.config.get("START_CASH", 1000.0)
                    capital_per_trade = base_capital * float(weight)
                    print(
                        f"Using dynamic weight for {symbol}: {weight * 100:.1f}% -> "
                        f"{capital_per_trade:.2f} USD"
                    )
                else:
                    capital_per_trade = self.config.get("CAPITAL_PER_TRADE", 100.0)
                    print(f"Using fixed capital for {symbol}: {capital_per_trade:.2f} USD")

                order_id = self._execute_market_buy_order(symbol, capital_per_trade)

                if order_id:
                    # For market order, we need the filled amount
                    time.sleep(1)  # Brief wait for matching
                    try:
                        order = self.exchange.fetch_order(
                            order_id,
                            symbol.replace("-", "/") if "/" not in symbol else symbol,
                        )
                        filled_amount = order.get("filled", 0.0)
                    except Exception:
                        # Fallback for dry run or API issues
                        ticker = self.exchange.fetch_ticker(
                            symbol.replace("-", "/") if "/" not in symbol else symbol
                        )
                        filled_amount = capital_per_trade / ticker["last"]

                    self.active_positions[symbol] = {
                        "entry_order_id": order_id,
                        "entry_price": sig["current_price"],
                        "amount": filled_amount,
                        "stop_pct": self.per_coin_params[symbol]["params"].get("stop_pct", 3.0),
                    }

                    # Place Trailing Stop
                    tsl_id = self._execute_trailing_stop_order(
                        symbol, filled_amount, self.active_positions[symbol]["stop_pct"]
                    )
                    self.active_positions[symbol]["tsl_order_id"] = tsl_id
                    self.save_state()

            elif symbol in self.active_positions:
                # If Kraken handles the TSL, we don't need to manually exit
                # UNLESS the strategy has a specific 'exit' signal (not just stop-loss)
                if sig["exit"]:
                    print(f"[{time.strftime('%X')}] STRATEGY EXIT SIGNAL FOR {symbol}")
                    # We could cancel TSL and sell market here, if required.
                    # Currently we prioritize Kraken's native TSL logic for safety.

    def run_event_loop(self) -> None:
        """Poll for new data every 4h and execute trades."""
        self.state = "EVENT_LOOP"
        print("=========================================")
        print(f"  ggTrader Live Execution Engine Started (DRY_RUN={self.config.get('DRY_RUN')})")
        print("=========================================")

        try:
            while True:
                # 1. Fetch live data
                latest_df = self._fetch_latest_data()

                if not latest_df.empty:
                    # 2. Compute signals per coin
                    latest_signals = self._compute_latest_signals(latest_df)

                    # 3. Execute / Manage Trades
                    self._execute_trade_logic(latest_signals)

                # 4. Wait for next 4h candle close
                now = time.gmtime()
                next_hour = ((now.tm_hour // 4) + 1) * 4
                if next_hour >= 24:
                    wait_seconds = (24 - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec
                else:
                    wait_seconds = (next_hour - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec

                wait_seconds += 30  # Buffer to ensure candle registration
                print(
                    f"[{time.strftime('%X')}] Loop complete. Sleeping {wait_seconds}s "
                    "until next 4h candle."
                )
                time.sleep(max(wait_seconds, 60))

        except KeyboardInterrupt:
            self.state = "STOPPED"
            print("\nExecution Engine stopped by user.")
