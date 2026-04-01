"""Local trade logging and Kraken sync for live trading performance tracking."""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("ggTraderLive")

TRADE_LOG_HEADERS = [
    "timestamp", "symbol", "side", "order_id", "price", "amount",
    "amount_usd", "fee", "fee_currency",
]

POSITION_CLOSE_HEADERS = [
    "close_timestamp", "symbol", "entry_time", "exit_time", "entry_price",
    "exit_price", "amount", "gross_pnl", "net_pnl", "pnl_pct", "fee_entry",
    "fee_exit", "fee_total", "hold_duration_hours", "exit_reason",
]

BALANCE_SNAPSHOT_HEADERS = [
    "timestamp", "total_usd", "free_usd", "positions_usd", "num_positions",
]


class TradeTracker:
    """Records trades, balance snapshots, and round-trip P&L to local CSV files."""

    def __init__(self, data_dir: str = "data/live") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.trade_log_path = self.data_dir / "trade_log.csv"
        self.position_closes_path = self.data_dir / "position_closes.csv"
        self.balance_snapshots_path = self.data_dir / "balance_snapshots.csv"

        self._ensure_headers(self.trade_log_path, TRADE_LOG_HEADERS)
        self._ensure_headers(self.position_closes_path, POSITION_CLOSE_HEADERS)
        self._ensure_headers(self.balance_snapshots_path, BALANCE_SNAPSHOT_HEADERS)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_headers(path: Path, headers: List[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    @staticmethod
    def _append_row(path: Path, row: List[Any]) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_buy(
        self,
        symbol: str,
        order_id: str,
        price: float,
        amount: float,
        amount_usd: float,
        fee: float = 0.0,
        fee_currency: str = "USD",
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or self._now_iso()
        self._append_row(self.trade_log_path, [
            ts, symbol, "buy", order_id, price, amount, amount_usd,
            fee, fee_currency,
        ])
        logger.info(f"  [Tracker] BUY  {symbol}: {amount:.6f} @ ${price:.4f} "
                     f"(${amount_usd:.2f}) fee={fee:.4f} {fee_currency}")

    def record_sell(
        self,
        symbol: str,
        order_id: str,
        price: float,
        amount: float,
        amount_usd: float,
        fee: float = 0.0,
        fee_currency: str = "USD",
        entry_price: Optional[float] = None,
        entry_time: Optional[str] = None,
        entry_fee: float = 0.0,
        exit_reason: str = "unknown",
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or self._now_iso()
        self._append_row(self.trade_log_path, [
            ts, symbol, "sell", order_id, price, amount, amount_usd,
            fee, fee_currency,
        ])

        # Write round-trip record if we have entry data
        if entry_price is not None and entry_price > 0:
            gross_pnl = (price - entry_price) * amount
            fee_total = entry_fee + fee
            net_pnl = gross_pnl - fee_total
            pnl_pct = ((price / entry_price) - 1.0) * 100.0 if entry_price else 0.0

            hold_hours = 0.0
            if entry_time:
                try:
                    t0 = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hold_hours = (t1 - t0).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    pass

            self._append_row(self.position_closes_path, [
                ts, symbol, entry_time, ts, entry_price, price, amount,
                round(gross_pnl, 6), round(net_pnl, 6), round(pnl_pct, 4),
                entry_fee, fee, round(fee_total, 6), round(hold_hours, 2),
                exit_reason,
            ])
            logger.info(f"  [Tracker] SELL {symbol}: {amount:.6f} @ ${price:.4f} "
                         f"PnL=${net_pnl:.2f} ({pnl_pct:+.2f}%) reason={exit_reason}")
        else:
            logger.info(f"  [Tracker] SELL {symbol}: {amount:.6f} @ ${price:.4f} "
                         f"(no entry data) reason={exit_reason}")

    def record_balance_snapshot(
        self,
        total_usd: float,
        free_usd: float,
        positions_usd: float,
        num_positions: int,
    ) -> None:
        self._append_row(self.balance_snapshots_path, [
            self._now_iso(), round(total_usd, 2), round(free_usd, 2),
            round(positions_usd, 2), num_positions,
        ])

    # ------------------------------------------------------------------
    # Kraken sync / backfill
    # ------------------------------------------------------------------

    def sync_from_kraken(
        self, exchange: Any, since_timestamp: Optional[int] = None
    ) -> int:
        """Pull historical trades from Kraken via CCXT and merge into local CSVs.

        Returns the number of new trades added.
        """
        existing_order_ids: set[str] = set()
        if self.trade_log_path.exists():
            try:
                df = pd.read_csv(self.trade_log_path)
                if "order_id" in df.columns:
                    existing_order_ids = set(df["order_id"].astype(str))
            except Exception:
                pass

        all_trades: List[Dict] = []
        since = since_timestamp or 0

        while True:
            try:
                trades = exchange.fetch_my_trades(symbol=None, since=since, limit=500)
            except Exception as e:
                logger.warning(f"  [Sync] fetch_my_trades failed: {e}")
                # Fall back to per-symbol fetching
                trades = self._fetch_trades_per_symbol(exchange, since)
                all_trades.extend(trades)
                break

            if not trades:
                break
            all_trades.extend(trades)
            since = trades[-1]["timestamp"] + 1
            if len(trades) < 500:
                break
            time.sleep(1)  # Kraken rate limit

        new_count = 0
        for t in all_trades:
            oid = str(t.get("order", t.get("id", "")))
            if oid in existing_order_ids:
                continue

            symbol = t.get("symbol", "").replace("/", "-")
            fee_info = t.get("fee", {})
            self._append_row(self.trade_log_path, [
                t.get("datetime", ""),
                symbol,
                t.get("side", ""),
                oid,
                t.get("price", 0),
                t.get("amount", 0),
                t.get("cost", 0),
                fee_info.get("cost", 0),
                fee_info.get("currency", "USD"),
            ])
            existing_order_ids.add(oid)
            new_count += 1

        if new_count:
            logger.info(f"  [Sync] Added {new_count} new trades from Kraken")
            self._rebuild_position_closes()

        return new_count

    def _fetch_trades_per_symbol(
        self, exchange: Any, since: int
    ) -> List[Dict]:
        """Fallback: fetch trades symbol-by-symbol if bulk fetch unsupported."""
        all_trades: List[Dict] = []
        try:
            markets = exchange.load_markets()
            usd_pairs = [s for s in markets if s.endswith("/USD")]
        except Exception:
            return all_trades

        for pair in usd_pairs:
            try:
                trades = exchange.fetch_my_trades(symbol=pair, since=since, limit=500)
                all_trades.extend(trades)
                time.sleep(0.5)
            except Exception:
                continue
        return all_trades

    def _rebuild_position_closes(self) -> None:
        """Rebuild position_closes.csv from trade_log.csv by pairing buys and sells."""
        try:
            df = pd.read_csv(self.trade_log_path)
        except Exception:
            return

        if df.empty:
            return

        df = df.sort_values("timestamp")

        # Track open buys per symbol (FIFO)
        open_buys: Dict[str, List[Dict]] = {}
        closes: List[List] = []

        for _, row in df.iterrows():
            symbol = row["symbol"]
            if row["side"] == "buy":
                open_buys.setdefault(symbol, []).append({
                    "timestamp": row["timestamp"],
                    "price": float(row["price"]),
                    "amount": float(row["amount"]),
                    "fee": float(row.get("fee", 0)),
                })
            elif row["side"] == "sell" and symbol in open_buys and open_buys[symbol]:
                buy = open_buys[symbol].pop(0)  # FIFO
                exit_price = float(row["price"])
                amount = float(row["amount"])
                entry_price = buy["price"]
                fee_entry = buy["fee"]
                fee_exit = float(row.get("fee", 0))

                gross_pnl = (exit_price - entry_price) * amount
                fee_total = fee_entry + fee_exit
                net_pnl = gross_pnl - fee_total
                pnl_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else 0.0

                hold_hours = 0.0
                try:
                    t0 = datetime.fromisoformat(str(buy["timestamp"]).replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                    hold_hours = (t1 - t0).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    pass

                closes.append([
                    row["timestamp"], symbol, buy["timestamp"], row["timestamp"],
                    entry_price, exit_price, amount,
                    round(gross_pnl, 6), round(net_pnl, 6), round(pnl_pct, 4),
                    fee_entry, fee_exit, round(fee_total, 6),
                    round(hold_hours, 2), "synced",
                ])

        # Rewrite position_closes.csv
        with open(self.position_closes_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(POSITION_CLOSE_HEADERS)
            writer.writerows(closes)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_trade_log(self) -> pd.DataFrame:
        if not self.trade_log_path.exists():
            return pd.DataFrame(columns=TRADE_LOG_HEADERS)
        df = pd.read_csv(self.trade_log_path, parse_dates=["timestamp"])
        return df

    def get_closed_positions(self) -> pd.DataFrame:
        if not self.position_closes_path.exists():
            return pd.DataFrame(columns=POSITION_CLOSE_HEADERS)
        df = pd.read_csv(self.position_closes_path, parse_dates=["close_timestamp"])
        return df

    def get_balance_history(self) -> pd.DataFrame:
        if not self.balance_snapshots_path.exists():
            return pd.DataFrame(columns=BALANCE_SNAPSHOT_HEADERS)
        df = pd.read_csv(self.balance_snapshots_path, parse_dates=["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def compute_summary_stats(self) -> Dict[str, Any]:
        closes = self.get_closed_positions()
        trades = self.get_trade_log()
        balances = self.get_balance_history()

        stats: Dict[str, Any] = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
            "total_fees": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "best_trade_pnl": 0.0,
            "best_trade_symbol": "",
            "worst_trade_pnl": 0.0,
            "worst_trade_symbol": "",
            "current_balance": None,
            "first_snapshot": None,
            "latest_snapshot": None,
        }

        if not closes.empty:
            stats["total_trades"] = len(closes)
            winners = closes[closes["net_pnl"] > 0]
            losers = closes[closes["net_pnl"] <= 0]
            stats["wins"] = len(winners)
            stats["losses"] = len(losers)
            stats["win_rate"] = (len(winners) / len(closes)) * 100.0 if len(closes) else 0.0

            stats["total_gross_pnl"] = float(closes["gross_pnl"].sum())
            stats["total_net_pnl"] = float(closes["net_pnl"].sum())
            stats["total_fees"] = float(closes["fee_total"].sum())

            stats["avg_win"] = float(winners["net_pnl"].mean()) if not winners.empty else 0.0
            stats["avg_loss"] = float(losers["net_pnl"].mean()) if not losers.empty else 0.0

            total_wins = float(winners["net_pnl"].sum()) if not winners.empty else 0.0
            total_losses = abs(float(losers["net_pnl"].sum())) if not losers.empty else 0.0
            stats["profit_factor"] = total_wins / total_losses if total_losses > 0 else float("inf")

            best_idx = closes["net_pnl"].idxmax()
            worst_idx = closes["net_pnl"].idxmin()
            stats["best_trade_pnl"] = float(closes.loc[best_idx, "net_pnl"])
            stats["best_trade_symbol"] = str(closes.loc[best_idx, "symbol"])
            stats["worst_trade_pnl"] = float(closes.loc[worst_idx, "net_pnl"])
            stats["worst_trade_symbol"] = str(closes.loc[worst_idx, "symbol"])

        # Also tally fees from trade_log for buys that haven't closed yet
        if not trades.empty and "fee" in trades.columns:
            stats["total_fees_all"] = float(trades["fee"].astype(float).sum())

        if not balances.empty:
            stats["current_balance"] = float(balances["total_usd"].iloc[-1])
            stats["first_snapshot"] = str(balances["timestamp"].iloc[0])
            stats["latest_snapshot"] = str(balances["timestamp"].iloc[-1])

        return stats
