"""Live trade logging and Kraken sync.

Persistence is fully DB-backed: orders → ``orders``, position closes → ``trades``,
balance snapshots → ``live_balance_snapshots``. The legacy ``data/live*/*.csv``
files are no longer written or read.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from ggTrader.utils.result_db_manager import ResultDBManager

logger = logging.getLogger("ggTraderLive")

# Column shape returned by get_balance_history / get_trade_log /
# get_closed_positions — preserved for backwards compatibility with the
# pnl_report_builder and any other consumer.
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


def _parse_iso(ts: str | datetime | None) -> Optional[datetime]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class TradeTracker:
    """DB-backed live trade tracker (orders, closed positions, balance snapshots)."""

    def __init__(
        self,
        data_dir: str = "data/live",  # retained for API compatibility; unused
        db_manager: Optional[ResultDBManager] = None,
        run_id: str = "LIVE",
    ) -> None:
        if db_manager is None:
            from ggTrader.utils.result_db_manager import ResultDBManager
            db_manager = ResultDBManager()
        self.db_manager = db_manager
        self.run_id = run_id
        self.asset_class = "crypto"

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

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
        ts = _parse_iso(timestamp) or self._now_utc()
        try:
            self.db_manager.add_order(
                run_id=self.run_id, symbol=symbol, side="buy", order_id=order_id,
                price=price, amount=amount, amount_usd=amount_usd,
                fee=fee, fee_currency=fee_currency, timestamp=ts,
            )
        except Exception as e:
            logger.warning(f"  [Tracker] DB record_buy failed: {e}")
        logger.info(
            f"  [Tracker] BUY  {symbol}: {amount:.6f} @ ${price:.4f} "
            f"(${amount_usd:.2f}) fee={fee:.4f} {fee_currency}"
        )

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
        ts = _parse_iso(timestamp) or self._now_utc()
        try:
            self.db_manager.add_order(
                run_id=self.run_id, symbol=symbol, side="sell", order_id=order_id,
                price=price, amount=amount, amount_usd=amount_usd,
                fee=fee, fee_currency=fee_currency, timestamp=ts,
            )
        except Exception as e:
            logger.warning(f"  [Tracker] DB record_sell (order) failed: {e}")

        if entry_price is not None and entry_price > 0:
            gross_pnl = (price - entry_price) * amount
            fee_total = entry_fee + fee
            net_pnl = gross_pnl - fee_total
            pnl_pct = ((price / entry_price) - 1.0) * 100.0 if entry_price else 0.0

            t0 = _parse_iso(entry_time)
            hold_hours = ((ts - t0).total_seconds() / 3600.0) if t0 else 0.0

            try:
                df_trade = pd.DataFrame([{
                    "symbol": symbol,
                    "entry_time": t0 or ts,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "profit": net_pnl,
                    "profit_pct": pnl_pct,
                    "status": "closed",
                    "amount": amount,
                    "gross_pnl": round(gross_pnl, 6),
                    "fee_entry": entry_fee,
                    "fee_exit": fee,
                    "fee_total": round(fee_total, 6),
                    "hold_duration_hours": round(hold_hours, 2),
                    "exit_reason": exit_reason,
                }])
                self.db_manager.add_trades(run_id=self.run_id, df_trades=df_trade)
            except Exception as e:
                logger.warning(f"  [Tracker] DB record_sell (trade) failed: {e}")

            logger.info(
                f"  [Tracker] SELL {symbol}: {amount:.6f} @ ${price:.4f} "
                f"PnL=${net_pnl:.2f} ({pnl_pct:+.2f}%) reason={exit_reason}"
            )
        else:
            logger.info(
                f"  [Tracker] SELL {symbol}: {amount:.6f} @ ${price:.4f} "
                f"(no entry data) reason={exit_reason}"
            )

    def record_balance_snapshot(
        self,
        total_usd: float,
        free_usd: float,
        positions_usd: float,
        num_positions: int,
    ) -> None:
        ts = self._now_utc()
        try:
            self.db_manager.add_balance_snapshot(
                asset_class=self.asset_class,
                timestamp=ts,
                total_usd=total_usd,
                free_usd=free_usd,
                positions_usd=positions_usd,
                num_positions=num_positions,
            )
        except Exception as e:
            logger.warning(f"  [Tracker] DB record_balance_snapshot failed: {e}")

    # ------------------------------------------------------------------
    # Kraken sync — pulls authoritative trade history into the orders table
    # then rebuilds closed-position rows in the trades table via FIFO matching.
    # ------------------------------------------------------------------

    def sync_from_kraken(
        self, exchange: Any, since_timestamp: Optional[int] = None
    ) -> int:
        """Pull historical trades from Kraken via CCXT and upsert into the DB.

        Returns the number of new fills inserted.
        """
        existing_orders_df = self.db_manager.get_orders_df(self.run_id)
        existing_order_ids: set[str] = set(
            existing_orders_df["order_id"].astype(str)
        ) if not existing_orders_df.empty else set()

        all_trades: List[Dict] = []
        since = since_timestamp or 0
        while True:
            try:
                trades = exchange.fetch_my_trades(symbol=None, since=since, limit=500)
            except Exception as e:
                logger.warning(f"  [Sync] fetch_my_trades failed: {e}")
                trades = self._fetch_trades_per_symbol(exchange, since)
                all_trades.extend(trades)
                break
            if not trades:
                break
            all_trades.extend(trades)
            since = trades[-1]["timestamp"] + 1
            if len(trades) < 500:
                break
            time.sleep(1)

        new_count = 0
        for t in all_trades:
            oid = str(t.get("order", t.get("id", "")))
            if oid in existing_order_ids:
                continue
            fee_info = t.get("fee", {}) or {}
            ts = _parse_iso(t.get("datetime")) or self._now_utc()
            symbol = t.get("symbol", "").replace("/", "-")
            try:
                self.db_manager.add_order(
                    run_id=self.run_id,
                    symbol=symbol,
                    side=t.get("side", ""),
                    order_id=oid,
                    price=float(t.get("price", 0) or 0),
                    amount=float(t.get("amount", 0) or 0),
                    amount_usd=float(t.get("cost", 0) or 0),
                    fee=float(fee_info.get("cost", 0) or 0),
                    fee_currency=fee_info.get("currency", "USD"),
                    timestamp=ts,
                )
            except Exception as e:
                logger.warning(f"  [Sync] add_order({oid}) failed: {e}")
                continue
            existing_order_ids.add(oid)
            new_count += 1

        if new_count:
            logger.info(f"  [Sync] Added {new_count} new trades from Kraken")
            self._rebuild_position_closes()

        return new_count

    def _fetch_trades_per_symbol(self, exchange: Any, since: int) -> List[Dict]:
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
        """Pair buys and sells from the orders table via FIFO and upsert into trades.

        Replaces the old CSV-based rebuild. Wipes existing closed rows for this
        run_id (they're derived) and reinserts the freshly computed set.
        """
        df = self.db_manager.get_orders_df(self.run_id)
        if df.empty:
            return

        df = df.sort_values("timestamp")
        open_buys: Dict[str, List[Dict]] = {}
        closes: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            symbol = row["symbol"]
            side = str(row["side"]).lower()
            if side == "buy":
                open_buys.setdefault(symbol, []).append({
                    "timestamp": row["timestamp"],
                    "price": float(row["price"]),
                    "amount": float(row["amount"]),
                    "fee": float(row.get("fee", 0) or 0),
                })
            elif side == "sell" and symbol in open_buys and open_buys[symbol]:
                buy = open_buys[symbol].pop(0)
                exit_price = float(row["price"])
                amount = float(row["amount"])
                entry_price = buy["price"]
                fee_entry = buy["fee"]
                fee_exit = float(row.get("fee", 0) or 0)
                gross_pnl = (exit_price - entry_price) * amount
                fee_total = fee_entry + fee_exit
                net_pnl = gross_pnl - fee_total
                pnl_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else 0.0
                hold_hours = 0.0
                t0 = _parse_iso(buy["timestamp"])
                t1 = _parse_iso(row["timestamp"])
                if t0 and t1:
                    hold_hours = (t1 - t0).total_seconds() / 3600.0

                closes.append({
                    "symbol": symbol,
                    "entry_time": t0 or buy["timestamp"],
                    "exit_time": t1 or row["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": round(net_pnl, 6),
                    "profit_pct": round(pnl_pct, 4),
                    "status": "closed",
                    "amount": amount,
                    "gross_pnl": round(gross_pnl, 6),
                    "fee_entry": fee_entry,
                    "fee_exit": fee_exit,
                    "fee_total": round(fee_total, 6),
                    "hold_duration_hours": round(hold_hours, 2),
                    "exit_reason": "synced",
                })

        # Replace this run's closed rows with the freshly computed set.
        from sqlalchemy import text as _text
        with self.db_manager.engine.begin() as conn:
            conn.execute(
                _text("DELETE FROM trades WHERE run_id = :rid AND status = 'closed'"),
                {"rid": self.run_id},
            )
        if closes:
            self.db_manager.add_trades(run_id=self.run_id, df_trades=pd.DataFrame(closes))

    # ------------------------------------------------------------------
    # Data access — stable surface area for the report builder
    # ------------------------------------------------------------------

    def get_trade_log(self) -> pd.DataFrame:
        df = self.db_manager.get_orders_df(self.run_id)
        if df.empty:
            return pd.DataFrame(columns=TRADE_LOG_HEADERS)
        return df

    def get_closed_positions(self) -> pd.DataFrame:
        df = self.db_manager.get_closed_positions_df(self.run_id)
        if df.empty:
            return pd.DataFrame(columns=POSITION_CLOSE_HEADERS)
        return df

    def get_balance_history(self) -> pd.DataFrame:
        df = self.db_manager.get_balance_history(self.asset_class)
        if df.empty:
            return pd.DataFrame(columns=BALANCE_SNAPSHOT_HEADERS)
        return df

    def find_open_buy_for(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the oldest unmatched BUY for ``symbol`` from the orders table.

        Used by the live trader's reconciler when it discovers a position on
        the exchange that has no entry in ``active_positions``.
        """
        df = self.get_trade_log()
        if df.empty:
            return None
        df = df[df["symbol"] == symbol].copy()
        if df.empty:
            return None
        df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("_ts")

        open_buys: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            side = str(row["side"]).lower()
            if side == "buy":
                open_buys.append({
                    "order_id": str(row["order_id"]),
                    "price": float(row["price"]),
                    "time": str(row["timestamp"]),
                    "fee": float(row["fee"]) if pd.notna(row["fee"]) else 0.0,
                    "amount": float(row["amount"]),
                })
            elif side == "sell" and open_buys:
                open_buys.pop(0)

        return open_buys[0] if open_buys else None

    # ------------------------------------------------------------------
    # Summary stats — used by pnl_report_builder
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

        if not trades.empty and "fee" in trades.columns:
            stats["total_fees_all"] = float(trades["fee"].astype(float).sum())

        if not balances.empty:
            stats["current_balance"] = float(balances["total_usd"].iloc[-1])
            stats["first_snapshot"] = str(balances["timestamp"].iloc[0])
            stats["latest_snapshot"] = str(balances["timestamp"].iloc[-1])

        return stats

    # ------------------------------------------------------------------
    # Backfill — kept for the existing `ggt db sync-live` admin command;
    # now reads from the legacy CSVs (if present) and writes to the DB,
    # which is exactly what we need for the migration cutover.
    # ------------------------------------------------------------------

    def backfill_to_db(self, legacy_dir: str = "data/live") -> Dict[str, int]:
        """One-time import of legacy CSVs into the DB. No-op if files absent."""
        from pathlib import Path
        counts = {"orders": 0, "trades": 0, "equity": 0}
        legacy = Path(legacy_dir)

        log_path = legacy / "trade_log.csv"
        closes_path = legacy / "position_closes.csv"
        balances_path = legacy / "balance_snapshots.csv"

        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                for _, row in df.iterrows():
                    try:
                        ts = _parse_iso(row["timestamp"]) or self._now_utc()
                        self.db_manager.add_order(
                            run_id=self.run_id,
                            symbol=row["symbol"],
                            side=row["side"],
                            order_id=str(row["order_id"]),
                            price=float(row["price"]),
                            amount=float(row["amount"]),
                            amount_usd=float(row.get("amount_usd", 0)),
                            fee=float(row.get("fee", 0)),
                            fee_currency=row.get("fee_currency", "USD"),
                            timestamp=ts,
                        )
                        counts["orders"] += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"  [Backfill] Orders failed: {e}")

        if closes_path.exists():
            try:
                df = pd.read_csv(closes_path)
                rows: list[dict] = []
                for _, row in df.iterrows():
                    try:
                        rows.append({
                            "symbol": row["symbol"],
                            "entry_time": _parse_iso(row["entry_time"]),
                            "exit_time": _parse_iso(row["exit_time"]),
                            "entry_price": float(row["entry_price"]),
                            "exit_price": float(row["exit_price"]),
                            "profit": float(row["net_pnl"]),
                            "profit_pct": float(row["pnl_pct"]),
                            "status": "closed",
                            "amount": float(row.get("amount", 0)),
                            "gross_pnl": float(row.get("gross_pnl", 0)),
                            "fee_entry": float(row.get("fee_entry", 0)),
                            "fee_exit": float(row.get("fee_exit", 0)),
                            "fee_total": float(row.get("fee_total", 0)),
                            "hold_duration_hours": float(row.get("hold_duration_hours", 0)),
                            "exit_reason": row.get("exit_reason", ""),
                        })
                    except Exception:
                        continue
                if rows:
                    self.db_manager.add_trades(self.run_id, pd.DataFrame(rows))
                    counts["trades"] = len(rows)
            except Exception as e:
                logger.warning(f"  [Backfill] Trades failed: {e}")

        if balances_path.exists():
            try:
                df = pd.read_csv(balances_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
                for _, row in df.iterrows():
                    try:
                        self.db_manager.add_balance_snapshot(
                            asset_class=self.asset_class,
                            timestamp=row["timestamp"].to_pydatetime(),
                            total_usd=float(row["total_usd"]),
                            free_usd=float(row["free_usd"]),
                            positions_usd=float(row["positions_usd"]),
                            num_positions=int(row["num_positions"]),
                        )
                        counts["equity"] += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"  [Backfill] Equity failed: {e}")

        return counts
