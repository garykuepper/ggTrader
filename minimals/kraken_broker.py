# brokers/kraken_broker.py
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from kraken.spot import Trade, Market  # python-kraken-sdk

class KrakenBroker:
    """
    Minimal live broker for spot trading on Kraken.
    - Places GTD limit BUY/SELL with expiretm at next interval boundary.
    - Rounds price/volume to exchange precision and checks order minimums.
    - Tracks open orders via txid and returns txid on submit.
    """
    def __init__(self, api_key: str, api_secret: str, base_ccy: str = "USD"):
        self.trade = Trade(key=api_key, secret=api_secret)
        self.mkt = Market()  # public endpoints
        self.base_ccy = base_ccy
        # cache for pair metadata
        self._pair_meta: Dict[str, Dict[str, Any]] = {}

    # ---------- metadata / precision ----------

    def _get_pair(self, pair: str) -> Dict[str, Any]:
        """Fetch AssetPairs info and cache it. `pair` should be Kraken altname, e.g. 'ETHUSD'."""
        if pair not in self._pair_meta:
            data = self.mkt.get_tradable_asset_pairs(pair=pair)  # returns dict with 'result'
            # result comes keyed by pair code (canonical), grab first
            res = data.get("result") or {}
            meta = next(iter(res.values())) if res else {}
            if not meta:
                raise ValueError(f"Unknown/unsupported pair: {pair}")
            self._pair_meta[pair] = meta
        return self._pair_meta[pair]

    def _round_price_qty(self, pair: str, price: float, qty: float) -> tuple[float, float]:
        meta = self._get_pair(pair)
        p_dec = int(meta.get("pair_decimals", 5))
        q_dec = int(meta.get("lot_decimals", 8))
        # round *down* conservatively
        rp = float(f"{price:.{p_dec}f}")
        rq = float(f"{qty:.{q_dec}f}")
        return rp, rq

    def _check_mins(self, pair: str, price: float, qty: float) -> None:
        meta = self._get_pair(pair)
        ordermin = float(meta.get("ordermin", 0)) if meta.get("ordermin") else 0.0
        costmin = float(meta.get("costmin", 0)) if meta.get("costmin") else 0.0
        notional = price * qty
        if ordermin and qty + 1e-15 < ordermin:
            raise ValueError(f"qty {qty} < ordermin {ordermin} for {pair}")
        if costmin and notional + 1e-12 < costmin:
            raise ValueError(f"notional ${notional:.2f} < costmin ${costmin:.2f} for {pair}")

    # ---------- helpers ----------

    @staticmethod
    def _gtd_expire(next_interval_hours: int) -> int:
        """
        Return a UNIX seconds timestamp used as expiretm for GTD.
        If you want the order to live until the *next* run, pass the number of hours until next run.
        """
        return int(time.time()) + next_interval_hours * 3600

    def _place_limit(self, pair: str, side: str, price: float, qty: float,
                     tif_hours: int, client_id: Optional[str] = None, post_only: bool = False) -> str:
        """
        Submit a GTD limit order with expiretm ~ now + tif_hours*3600.
        Returns Kraken txid.
        """
        price, qty = self._round_price_qty(pair, price, qty)
        self._check_mins(pair, price, qty)

        extra = {
            "ordertype": "limit",
            "timeinforce": "GTD",                            # GTC|IOC|GTD
            "expiretm": self._gtd_expire(tif_hours),         # UNIX seconds
        }
        if post_only:
            extra["oflags"] = "post"                         # maker-only, optional

        if client_id:
            extra["cl_ord_id"] = client_id                   # client order id (mutually exclusive w/ userref)

        # The SDK expects plain strings for side/pair and price/volume
        resp = self.trade.add_order(
            pair=pair,
            side=side,               # "buy" | "sell"
            ordertype="limit",
            price=str(price),
            volume=str(qty),
            extra_params=extra
        )
        # REST returns {'result': {'txid': ['...'], ...}}; SDK unwraps to dict
        # Normalize txid(s)
        tx = resp.get("txid") or resp.get("result", {}).get("txid")
        if isinstance(tx, list):
            txid = tx[0]
        else:
            txid = str(tx)
        if not txid:
            raise RuntimeError(f"Kraken add_order returned no txid: {resp}")
        return txid

    # ---------- public API (compatible-ish with your PaperBroker) ----------

    def buy_limit_gtd(self, pair: str, qty: float, limit_price: float, tif_hours: int,
                      client_id: Optional[str] = None, post_only: bool = False) -> str:
        return self._place_limit(pair, "buy", limit_price, qty, tif_hours, client_id, post_only)

    def sell_limit_gtd(self, pair: str, qty: float, limit_price: float, tif_hours: int,
                       client_id: Optional[str] = None, post_only: bool = False) -> str:
        return self._place_limit(pair, "sell", limit_price, qty, tif_hours, client_id, post_only)

    def cancel(self, txid: str) -> None:
        try:
            self.trade.cancel_order_batch([txid])
        except Exception as e:
            print(f"[WARN] cancel failed {txid}: {e}")
