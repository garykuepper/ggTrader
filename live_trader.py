# live_trader.py
import os
import time
import hmac
import json
import math
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests
import pandas as pd
from pymongo import MongoClient, ASCENDING

from data_manager import DataManager, CryptoDataManager
from trading_strategy import EMAStrategy
from trailing_stop import TrailingStop

# ---------------------
# Config and constants
# ---------------------
load_dotenv()
INTERVAL = '4h'
FEE_PCT = 0.001  # 0.1% per trade assumption
MAX_ALLOCATION_PCT = 0.10  # 10% of total equity per position
TOP_N = 10
MIN_VOLUME = 100000
STATE_FILE = 'live_trader_state.json'
SYMBOLS_REFRESH_HOURS = 24  # refresh top list daily
EMA_FAST_DEFAULT = 14
EMA_SLOW_DEFAULT = 21
TRAILING_PCT_DEFAULT = 0.03  # 3%

BINANCE_BASE = 'https://api.binance.us'
API_KEY = os.getenv('BINANCE_API_LIVE_KEY', '')
API_SECRET = os.getenv('BINANCE_SECRET_LIVE_KEY', '')

MATRIX_BASE = os.getenv('MATRIX_HOMESERVER', '')
MATRIX_TOKEN = os.getenv('MATRIX_ACCESS_TOKEN', '')
MATRIX_ROOM_ID = os.getenv('MATRIX_ROOM_ID', '')

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB = os.getenv('MONGO_DB', 'market_data')
MONGO_TRADES_COLL = os.getenv('MONGO_TRADES_COLL', 'live_trade_logs')

# -------------
# Logging setup
# -------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# -------------------
# Utility functions
# -------------------
def now_utc():
    return datetime.now(timezone.utc)

def align_end_to_interval(dt: datetime, interval: str) -> datetime:
    if interval == '4h':
        aligned_hour = (dt.hour // 4) * 4
        return dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)

def next_interval_time(current: datetime, interval: str) -> datetime:
    aligned = align_end_to_interval(current, interval)
    if current <= aligned:
        return aligned + timedelta(hours=4)
    return align_end_to_interval(current + timedelta(hours=4), interval)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step

# -------------------------
# Simple Matrix notifier
# -------------------------
class MatrixNotifier:
    def __init__(self, base_url: str, access_token: str, room_id: str):
        self.base_url = base_url.rstrip('/') if base_url else ''
        self.token = access_token
        self.room_id = room_id

    def send_text(self, text: str):
        if not (self.base_url and self.token and self.room_id):
            logging.info("Matrix notifier not configured; skipping message.")
            return
        try:
            txn_id = f"txn_{int(time.time() * 1000)}"
            url = f"{self.base_url}/_matrix/client/v3/rooms/{self.room_id}/send/m.room.message/{txn_id}"
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {"msgtype": "m.text", "body": text}
            r = requests.put(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
        except Exception as e:
            logging.warning(f"Matrix send failed: {e}")

# -------------------------
# MongoDB trade logger
# -------------------------
class MongoTradeLogger:
    def __init__(self, uri: str, db_name: str, coll_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.coll = self.db[coll_name]
        self._ensure_indexes()

    def _ensure_indexes(self):
        try:
            # Idempotent indexes
            self.coll.create_index([('orderId', ASCENDING)], unique=False, sparse=True, name='orderId_idx')
            self.coll.create_index([('ts', ASCENDING)], name='ts_idx')
            self.coll.create_index([('symbol', ASCENDING), ('ts', ASCENDING)], name='symbol_ts_idx')
        except Exception as e:
            logging.warning(f"Mongo index creation failed: {e}")

    def log_trade(
        self,
        *,
        ts: datetime,
        symbol: str,
        side: str,
        qty: float,
        avg_price: float,
        fee_usdt: float,
        reason: str,
        order: Optional[dict],
        exchange: str,
        interval: str,
        params: dict,
        equity_snapshot: dict,
    ):
        try:
            doc = {
                'ts': ts,
                'symbol': symbol,
                'side': side.upper(),
                'qty': float(qty),
                'avg_price': float(avg_price),
                'notional_usdt': float(qty * avg_price),
                'fee_usdt': float(fee_usdt),
                'reason': reason,
                'exchange': exchange,
                'interval': interval,
                'params': {
                    'ema_fast': int(params.get('ema_fast', 0)),
                    'ema_slow': int(params.get('ema_slow', 0)),
                    'trailing_pct': float(params.get('trailing_pct', 0.0)),
                },
                'equity_snapshot': {
                    'equity_usdt': float(equity_snapshot.get('equity_usdt', 0.0)),
                    'cash_usdt': float(equity_snapshot.get('cash_usdt', 0.0)),
                    'positions_value_usdt': float(equity_snapshot.get('positions_value_usdt', 0.0)),
                },
                'order_meta': {},
                'source': 'live_trader_v1',
            }
            if isinstance(order, dict):
                # Capture common fields if present
                for k in ('orderId', 'clientOrderId', 'transactTime', 'status', 'executedQty', 'cummulativeQuoteQty'):
                    if k in order:
                        doc['order_meta'][k] = order.get(k)
            self.coll.insert_one(doc)
            logging.info(f"Logged trade to MongoDB: {symbol} {side} {qty:.8f} @ {avg_price:.6f}")
        except Exception as e:
            logging.warning(f"Failed to log trade to MongoDB: {e}")

# -------------------------
# Binance US REST client
# -------------------------
class BinanceUSClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = BINANCE_BASE):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        self._filters = {}  # symbol -> filter info

    def _sign(self, params: Dict) -> Dict:
        q = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        sig = hmac.new(self.api_secret, q.encode(), hashlib.sha256).hexdigest()
        params['signature'] = sig
        return params

    def _request(self, method: str, path: str, params: Optional[Dict] = None, signed: bool = False):
        url = f"{self.base_url}{path}"
        params = params or {}
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params = self._sign(params)
        if method == 'GET':
            r = self.session.get(url, params=params, timeout=15)
        elif method == 'POST':
            r = self.session.post(url, params=params, timeout=15)
        elif method == 'DELETE':
            r = self.session.delete(url, params=params, timeout=15)
        else:
            raise ValueError("Unsupported method")
        r.raise_for_status()
        return r.json()

    def exchange_info(self):
        data = self._request('GET', '/api/v3/exchangeInfo')
        for s in data.get('symbols', []):
            sym = s.get('symbol')
            filters = {f['filterType']: f for f in s.get('filters', [])}
            lot = filters.get('LOT_SIZE', {})
            min_notional = filters.get('MIN_NOTIONAL', {})
            self._filters[sym] = {
                'stepSize': float(lot.get('stepSize', 0.0) or 0.0),
                'minQty': float(lot.get('minQty', 0.0) or 0.0),
                'minNotional': float(min_notional.get('minNotional', 0.0) or 0.0),
            }
        return self._filters

    def get_filters(self, symbol: str):
        if symbol not in self._filters:
            self.exchange_info()
        return self._filters.get(symbol, {'stepSize': 0.0, 'minQty': 0.0, 'minNotional': 0.0})

    def account(self):
        return self._request('GET', '/api/v3/account', signed=True)

    def ticker_price(self, symbol: str) -> float:
        data = self._request('GET', '/api/v3/ticker/price', params={'symbol': symbol})
        return float(data['price'])

    def klines(self, symbol: str, interval: str, start: datetime, end: datetime, limit=1000) -> pd.DataFrame:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': to_ms(start),
            'endTime': to_ms(end),
            'limit': limit
        }
        data = self._request('GET', '/api/v3/klines', params=params)
        if not data:
            cols = ['open', 'high', 'low', 'close', 'volume']
            return pd.DataFrame(columns=cols).set_index(pd.DatetimeIndex([], name='date'))
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(data, columns=cols)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        out = pd.DataFrame({
            'open': pd.to_numeric(df['open'], errors='coerce'),
            'high': pd.to_numeric(df['high'], errors='coerce'),
            'low': pd.to_numeric(df['low'], errors='coerce'),
            'close': pd.to_numeric(df['close'], errors='coerce'),
            'volume': pd.to_numeric(df['quote_asset_volume'], errors='coerce'),
        }, index=df['open_time'])
        out.index.name = 'date'
        return out

    def order_market(self, symbol: str, side: str, quantity: float):
        f = self.get_filters(symbol)
        qty = floor_to_step(quantity, f['stepSize'])
        if qty <= 0 or qty < f['minQty']:
            raise ValueError("Quantity too small after rounding.")
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': f"{qty:.16f}".rstrip('0').rstrip('.'),
        }
        return self._request('POST', '/api/v3/order', params=params, signed=True)

# -------------------------
# Strategy signal via EMAStrategy
# -------------------------
def compute_latest_signal(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> int:
    """
    Uses EMAStrategy to compute the signal on the last closed bar.
    Returns: 1 (bullish cross), -1 (bearish cross), 0 otherwise.
    """
    if df is None or df.empty:
        return 0
    params = {'ema_fast': int(ema_fast), 'ema_slow': int(ema_slow)}
    strat = EMAStrategy(name=f"EMA({ema_fast},{ema_slow})", params=params, trailing_pct=None)
    strat.find_crossovers(df)
    s = strat.signal_df['signal'] if hasattr(strat, 'signal_df') else None
    if s is None or len(s) == 0:
        return 0
    try:
        return int(s.iloc[-1])
    except Exception:
        return 0

# -------------------------
# Portfolio helpers
# -------------------------
def get_balances(client: BinanceUSClient) -> Dict[str, float]:
    acct = client.account()
    return {a['asset']: (float(a['free']) + float(a['locked'])) for a in acct.get('balances', [])}

def total_equity_usdt(balances: Dict[str, float], symbols: List[str], prices: Dict[str, float]) -> float:
    usdt = float(balances.get('USDT', 0.0))
    pos_val = 0.0
    for sym in symbols:
        if not sym.endswith('USDT'):
            continue
        base = sym[:-4]
        qty = float(balances.get(base, 0.0))
        if qty > 0:
            price = prices.get(sym, 0.0)
            pos_val += qty * price
    return usdt + pos_val

def held_positions(balances: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
    positions = {}
    for sym in symbols:
        if not sym.endswith('USDT'):
            continue
        base = sym[:-4]
        qty = float(balances.get(base, 0.0))
        if qty > 0:
            positions[sym] = qty
    return positions

# -------------------------
# Trailing stop state
# -------------------------
class State:
    def __init__(self, path: str):
        self.path = path
        self.trailing: Dict[str, Dict] = {}  # sym -> {'pct': float, 'highest': float, 'stop': float}
        self.symbols_last_refresh: Optional[str] = None
        self.top_symbols: List[str] = []

    def load(self):
        try:
            with open(self.path, 'r') as f:
                raw = json.load(f)
            self.trailing = raw.get('trailing', {})
            self.symbols_last_refresh = raw.get('symbols_last_refresh')
            self.top_symbols = raw.get('top_symbols', [])
        except Exception:
            self.trailing = {}
            self.symbols_last_refresh = None
            self.top_symbols = []

    def save(self):
        data = {
            'trailing': self.trailing,
            'symbols_last_refresh': self.symbols_last_refresh,
            'top_symbols': self.top_symbols,
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_trailing_stop(self, symbol: str, trailing_pct: float, initial_price: float) -> TrailingStop:
        info = self.trailing.get(symbol)
        ts = TrailingStop(trailing_pct, initial_price)
        if info:
            ts.highest_price = float(info.get('highest', initial_price))
            ts.stop_price = float(info.get('stop', initial_price * (1 - trailing_pct)))
        else:
            self.trailing[symbol] = {
                'pct': trailing_pct,
                'highest': ts.highest_price,
                'stop': ts.stop_price,
            }
        return ts

    def update_trailing(self, symbol: str, ts: TrailingStop, trailing_pct: float):
        self.trailing[symbol] = {
            'pct': trailing_pct,
            'highest': ts.highest_price,
            'stop': ts.stop_price,
        }

# -------------------------
# Symbols and params
# -------------------------
def fetch_top_symbols(state: State) -> List[str]:
    # daily refresh
    try:
        last = datetime.fromisoformat(state.symbols_last_refresh) if state.symbols_last_refresh else None
    except Exception:
        last = None
    if last and (now_utc() - last) < timedelta(hours=SYMBOLS_REFRESH_HOURS) and state.top_symbols:
        return state.top_symbols

    pairs = CryptoDataManager.get_24hr_top_binance(
        top_n=TOP_N,
        quote='USDT',
        min_change=0.0,
        min_trades=50,
        min_volume=MIN_VOLUME
    )
    symbols = [p.get('symbol', '') for p in pairs if isinstance(p, dict)]
    banned = ('UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT', 'USDCUSDT')
    cleaned = [s for s in symbols if s.endswith('USDT') and not any(b in s for b in banned)]
    deduped = list(dict.fromkeys(cleaned))[:TOP_N]
    state.top_symbols = deduped
    state.symbols_last_refresh = now_utc().isoformat()
    state.save()
    return deduped

def load_latest_params(interval: str) -> Tuple[int, int, float]:
    dm = DataManager()
    latest = dm.get_latest_strategy_params(strategy='EMA_trailing', interval=interval)
    if latest and all(k in latest for k in ('ema_fast', 'ema_slow', 'trailing_pct')):
        return int(latest['ema_fast']), int(latest['ema_slow']), float(latest['trailing_pct'])
    return EMA_FAST_DEFAULT, EMA_SLOW_DEFAULT, TRAILING_PCT_DEFAULT

# -------------------------
# Orders, notifications, logging
# -------------------------
def place_buy(client: BinanceUSClient, symbol: str, usdt_to_spend: float, price: float) -> Optional[dict]:
    f = client.get_filters(symbol)
    min_notional = max(10.0, float(f.get('minNotional', 0.0) or 0.0))
    if usdt_to_spend < min_notional:
        logging.info(f"Skip BUY {symbol}: notional too small ({usdt_to_spend:.2f} < {min_notional:.2f})")
        return None
    qty_est = usdt_to_spend / price
    return client.order_market(symbol, 'BUY', quantity=qty_est)

def place_sell(client: BinanceUSClient, symbol: str, quantity: float) -> Optional[dict]:
    if quantity <= 0:
        return None
    return client.order_market(symbol, 'SELL', quantity=quantity)

def parse_fill_metrics(order: Optional[dict], fallback_price: float) -> Tuple[float, float]:
    """
    Returns (executed_qty, avg_price). Falls back to cummulativeQuoteQty/executedQty or the given price.
    """
    if not isinstance(order, dict):
        return 0.0, fallback_price
    qty = 0.0
    notional = 0.0
    fills = order.get('fills')
    if isinstance(fills, list) and fills:
        for f in fills:
            q = float(f.get('qty', 0.0) or 0.0)
            p = float(f.get('price', fallback_price) or fallback_price)
            qty += q
            notional += q * p
    # If fills missing, try top-level fields
    if qty <= 0.0:
        try:
            qty = float(order.get('executedQty', 0.0) or 0.0)
        except Exception:
            qty = 0.0
    if notional <= 0.0 and qty > 0.0:
        try:
            cq = float(order.get('cummulativeQuoteQty', 0.0) or 0.0)
            if cq > 0:
                notional = cq
        except Exception:
            pass
    avg_price = (notional / qty) if qty > 0 else float(fallback_price)
    return qty, avg_price

def estimate_fee_usdt(order: Optional[dict], notional: float) -> float:
    """
    Try to sum commission from fills; if absent, estimate using FEE_PCT.
    """
    if isinstance(order, dict) and isinstance(order.get('fills'), list):
        total_comm_quote = 0.0
        for f in order['fills']:
            # Some venues report commission and commissionAsset
            try:
                commission = float(f.get('commission', 0.0) or 0.0)
                commission_asset = f.get('commissionAsset', 'USDT')
                # If commission asset is USDT, treat as is; otherwise fallback to estimate
                if commission_asset == 'USDT':
                    total_comm_quote += commission
            except Exception:
                pass
        if total_comm_quote > 0:
            return total_comm_quote
    return float(notional * FEE_PCT)

def send_trade(notifier: MatrixNotifier, side: str, symbol: str, qty: float, price: float, reason: str):
    notifier.send_text(f"{side} {symbol}\nQty: {qty:.8f}\nPrice: {price:.6f}\nReason: {reason}")

def send_daily_summary(notifier: MatrixNotifier, equity: float, usdt: float, positions_val: float):
    notifier.send_text(
        f"Daily portfolio update\nTotal Equity: {equity:.2f} USDT\nUSDT: {usdt:.2f}\nPositions Value: {positions_val:.2f} USDT"
    )

# -------------------------
# Main tick
# -------------------------
def run_once(client: BinanceUSClient, notifier: MatrixNotifier, state: State, trade_logger: MongoTradeLogger):
    ema_fast, ema_slow, trailing_pct = load_latest_params(INTERVAL)
    params_used = {'ema_fast': ema_fast, 'ema_slow': ema_slow, 'trailing_pct': trailing_pct}

    symbols = fetch_top_symbols(state)
    if not symbols:
        logging.info("No symbols to trade.")
        return

    # Prices
    prices = {}
    for sym in symbols:
        try:
            prices[sym] = client.ticker_price(sym)
        except Exception as e:
            logging.warning(f"Price fetch failed for {sym}: {e}")

    # Balances and equity snapshot
    balances = get_balances(client)
    positions = held_positions(balances, symbols)
    equity = total_equity_usdt(balances, symbols, prices)
    usdt = float(balances.get('USDT', 0.0))
    positions_val = equity - usdt
    snapshot = {'equity_usdt': equity, 'cash_usdt': usdt, 'positions_value_usdt': positions_val}
    logging.info(f"Equity: {equity:.2f} USDT (cash {usdt:.2f}, positions {positions_val:.2f})")

    max_per_pos = MAX_ALLOCATION_PCT * equity

    # Get last closed bar and fetch history
    end_time = align_end_to_interval(now_utc(), INTERVAL)
    start_time = end_time - timedelta(days=90)

    for sym in symbols:
        try:
            df = client.klines(sym, INTERVAL, start=start_time, end=end_time)
            if df.empty:
                continue
            # Signal from EMAStrategy (same as backtests)
            signal = compute_latest_signal(df, ema_fast, ema_slow)
            last_close = float(df['close'].iloc[-1])

            in_pos = sym in positions and positions[sym] > 0.0
            qty_pos = float(positions.get(sym, 0.0))

            # Maintain trailing stop
            ts = state.get_trailing_stop(sym, trailing_pct, initial_price=last_close)
            ts.update(last_close)
            state.update_trailing(sym, ts, trailing_pct)

            # Entry
            if not in_pos and signal == 1:
                allocation = min(max_per_pos, usdt)
                if allocation > 0:
                    try:
                        order = place_buy(client, sym, allocation, price=last_close)
                        if order:
                            exec_qty, avg_price = parse_fill_metrics(order, last_close)
                            notional = (exec_qty * avg_price) if exec_qty > 0 else allocation
                            fee_usdt = estimate_fee_usdt(order, notional)
                            send_trade(notifier, 'BUY', sym, exec_qty if exec_qty > 0 else allocation / last_close, avg_price, 'EMA bull cross')
                            # Reset trailing from entry price
                            ts = state.get_trailing_stop(sym, trailing_pct, initial_price=avg_price)
                            ts.update(avg_price)
                            state.update_trailing(sym, ts, trailing_pct)
                            state.save()
                            # Log to Mongo
                            trade_logger.log_trade(
                                ts=now_utc(),
                                symbol=sym,
                                side='BUY',
                                qty=float(exec_qty if exec_qty > 0 else allocation / avg_price),
                                avg_price=float(avg_price),
                                fee_usdt=float(fee_usdt),
                                reason='EMA bull cross',
                                order=order,
                                exchange='binance.us',
                                interval=INTERVAL,
                                params=params_used,
                                equity_snapshot=snapshot,
                            )
                    except Exception as e:
                        logging.warning(f"BUY failed for {sym}: {e}")

            # Exit on bear cross or trailing stop
            elif in_pos and (signal == -1 or ts.is_triggered(last_close)):
                try:
                    order = place_sell(client, sym, qty_pos)
                    if order:
                        exec_qty, avg_price = parse_fill_metrics(order, last_close)
                        notional = (exec_qty * avg_price) if exec_qty > 0 else (qty_pos * last_close)
                        fee_usdt = estimate_fee_usdt(order, notional)
                        reason = 'EMA bear cross' if signal == -1 else 'Trailing stop'
                        send_trade(notifier, 'SELL', sym, qty_pos, avg_price, reason)
                        # Clear trailing state
                        if sym in state.trailing:
                            del state.trailing[sym]
                        state.save()
                        # Log to Mongo
                        trade_logger.log_trade(
                            ts=now_utc(),
                            symbol=sym,
                            side='SELL',
                            qty=float(exec_qty if exec_qty > 0 else qty_pos),
                            avg_price=float(avg_price),
                            fee_usdt=float(fee_usdt),
                            reason=reason,
                            order=order,
                            exchange='binance.us',
                            interval=INTERVAL,
                            params=params_used,
                            equity_snapshot=snapshot,
                        )
                except Exception as e:
                    logging.warning(f"SELL failed for {sym}: {e}")

        except Exception as e:
            logging.warning(f"Decision error for {sym}: {e}")

    # Daily summary at 00:00 UTC
    current = now_utc()
    if current.hour == 0:
        balances = get_balances(client)
        prices = {sym: client.ticker_price(sym) for sym in symbols}
        equity = total_equity_usdt(balances, symbols, prices)
        usdt = float(balances.get('USDT', 0.0))
        positions_val = equity - usdt
        send_daily_summary(notifier, equity, usdt, positions_val)

# -------------------------
# Entrypoint
# -------------------------
def main():
    if not (API_KEY and API_SECRET):
        logging.error("Binance API credentials missing. Set BINANCE_API_KEY and BINANCE_API_SECRET.")
        return

    notifier = MatrixNotifier(MATRIX_BASE, MATRIX_TOKEN, MATRIX_ROOM_ID)
    client = BinanceUSClient(API_KEY, API_SECRET)
    state = State(STATE_FILE)
    state.load()
    trade_logger = MongoTradeLogger(MONGO_URI, MONGO_DB, MONGO_TRADES_COLL)

    logging.info("Starting live trader (4h cadence).")
    while True:
        try:
            current = now_utc()
            next_t = next_interval_time(current, INTERVAL)
            sleep_for = (next_t - current).total_seconds()
            logging.info(f"Sleeping until next 4h close: {next_t.isoformat()} ({int(max(0, sleep_for))}s)")
            if sleep_for > 0:
                time.sleep(sleep_for)

            run_once(client, notifier, state, trade_logger)
            time.sleep(5)  # small buffer
        except KeyboardInterrupt:
            logging.info("Stopping live trader.")
            break
        except Exception as e:
            logging.exception(f"Fatal loop error: {e}")
            time.sleep(15)

if __name__ == '__main__':
    main()
