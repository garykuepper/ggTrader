# Post-only limit entry — design proposal (review-only, no code yet)

> ## SHELVED — 2026-05-12
>
> **Superseded by venue change to Binance.US.** Maker-only on Kraken is no longer the cheapest cost-reduction option: Binance.US taker (0.02%/side) dominates Kraken Pro maker (0.25%/side) by an order of magnitude. The premise of this proposal — that switching execution mode on Kraken is the cheapest available cost reduction — was correct as of the WFO investigation arc but obsoleted by the venue decision the same day.
>
> Keep this document as reference in case the Binance.US migration is later reversed (account restrictions, withdrawal/deposit issues, or regional/legal constraint), in which case Kraken maker-only becomes the fallback path and this design is the starting point.
>
> Active execution-cost work pivots to the Binance.US migration sequence.

---

**Status:** proposal for review. No implementation in this commit.

**Motivation:** the WFO textbook-reset investigation arc closed with the finding that 35–36 strategy-coin combos pass the 4 textbook gates under frictionless conditions but **zero pass at the live trader's current taker rate (Kraken Pro $0+ tier = 0.40%/side, 0.80% round-trip)**. Switching entries to maker (0.25%/side, 0.50% round-trip) is the cheapest cost reduction available on the current exchange.

**Per Q1 audit (live trade history, 45 closed trades Apr 1 – May 11):** entries are 100% taker today. Every entry path uses `create_market_buy_order`. This is the code change that flips entries to maker.

**Per Q2 drift analysis (signal-conditional, 1h post-signal close):**

| Coin | abs(median) drift | Miss-up rate at 1h | Maker viability |
|---|---|---|---|
| TRX | 0.18% | 9.8% | clearly viable |
| BTC | 0.19% | 11.3% | viable (borderline on median heuristic) |
| ETH | 0.26% | 16.6% | marginal |
| DOGE | 0.42% | 18.2% | marginal-to-poor |

Of 36 frictionless passers: 19 are in TRX/BTC (clean maker viability), 17 in ETH/DOGE (marginal). This proposal targets the clean-viability subset first.

---

## Code-change shape

### Current entry path (`crypto_execution_engine.py:466-477`)

```python
def _execute_market_buy_order(self, symbol: str, amount_usd: float) -> Optional[str]:
    self.logger.info(f"BUY {symbol} ${amount_usd}")
    if self.config.get("DRY_RUN"): return "dry_run_id"
    try:
        pair = symbol.replace("-", "/")
        ticker = self.exchange.fetch_ticker(pair)
        amt = self.exchange.amount_to_precision(pair, amount_usd / ticker["last"])
        order = self.exchange.create_market_buy_order(pair, amt)
        return order["id"]
    except Exception as e:
        self.logger.error(f"Buy failed: {e}")
        return None
```

Call site (`crypto_execution_engine.py:656-662`):

```python
oid = self._execute_market_buy_order(s, cap)
if oid:
    time.sleep(1)
    ord = self.exchange.fetch_order(oid, s.replace("-", "/"))
    px = _safe_extract_fill_price(ord) or sig["current_price"]
    amt = float(ord.get("filled", 0)) or (cap / px)
    self.tracker.record_buy(s, oid, px, amt, px * amt, 0, "USD")
    # ... place trailing-stop exit
```

Assumes synchronous fill within 1 second — works for market orders, **not for limit orders**.

### Proposed replacement: `_execute_postonly_limit_buy_order`

```python
def _execute_postonly_limit_buy_order(
    self,
    symbol: str,
    amount_usd: float,
    wait_seconds: int,
    fallback: str,  # one of: "skip", "market", "reprice_once"
) -> Optional[Dict[str, Any]]:
    """
    Place a post-only limit buy at current best bid.

    Returns dict with order_id, fill_price, fill_amount on success;
    None if no fill and fallback="skip"; falls back to market or
    re-prices once depending on policy.
    """
    if self.config.get("DRY_RUN"):
        return {"order_id": "dry_run_id", "fill_price": 0.0, "fill_amount": 0.0}

    pair = symbol.replace("-", "/")
    try:
        ob = self.exchange.fetch_order_book(pair, limit=5)
        best_bid = ob["bids"][0][0]
        amt = self.exchange.amount_to_precision(pair, amount_usd / best_bid)
        price = self.exchange.price_to_precision(pair, best_bid)

        # Kraken ccxt convention: postOnly via params
        order = self.exchange.create_order(
            pair, "limit", "buy", amt, price,
            {"postOnly": True},  # OR {"oflags": "post"} per Kraken native
        )
        order_id = order["id"]
        self.logger.info(f"BUY {symbol} ${amount_usd} limit@{price} postOnly id={order_id}")
    except Exception as e:
        # postOnly rejection (would-cross), order book empty, etc.
        self.logger.warning(f"Post-only limit BUY {symbol} placement failed: {e!r}")
        return self._fallback_after_limit_fail(symbol, amount_usd, fallback)

    # Poll for fill with timeout
    poll_interval = 30  # seconds between status checks
    elapsed = 0
    while elapsed < wait_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            ord_info = self.exchange.fetch_order(order_id, pair)
            status = ord_info.get("status", "").lower()
            filled = float(ord_info.get("filled") or 0)
            if status == "closed" and filled > 0:
                fill_px = _safe_extract_fill_price(ord_info)
                self.logger.info(f"  [Limit] {symbol} filled at {fill_px} after {elapsed}s")
                return {"order_id": order_id, "fill_price": fill_px, "fill_amount": filled}
            elif status in ("canceled", "rejected"):
                # Exchange canceled it (e.g., market re-opening with crossing spread)
                self.logger.warning(f"  [Limit] {symbol} order {order_id} {status} mid-wait")
                return self._fallback_after_limit_fail(symbol, amount_usd, fallback)
        except Exception as e:
            self.logger.warning(f"  [Limit] {symbol} fetch_order failed: {e!r}")
            continue

    # Timeout — cancel the unfilled order, apply fallback
    try:
        self.exchange.cancel_order(order_id, pair)
    except Exception as e:
        self.logger.warning(f"  [Limit] {symbol} cancel after timeout failed: {e!r}")

    self.logger.info(f"  [Limit] {symbol} timeout after {wait_seconds}s, applying fallback={fallback}")
    return self._fallback_after_limit_fail(symbol, amount_usd, fallback)


def _fallback_after_limit_fail(
    self, symbol: str, amount_usd: float, policy: str
) -> Optional[Dict[str, Any]]:
    if policy == "skip":
        return None
    if policy == "market":
        # Existing taker path; eats the maker savings for this trade
        oid = self._execute_market_buy_order(symbol, amount_usd)
        if not oid: return None
        time.sleep(1)
        ord_info = self.exchange.fetch_order(oid, symbol.replace("-", "/"))
        return {
            "order_id": oid,
            "fill_price": _safe_extract_fill_price(ord_info),
            "fill_amount": float(ord_info.get("filled") or 0),
        }
    if policy == "reprice_once":
        # One re-attempt at fresh best bid; if that also times out, skip
        return self._execute_postonly_limit_buy_order(
            symbol, amount_usd, wait_seconds=self.config.get("MAKER_WAIT_SECONDS", 3600),
            fallback="skip",
        )
    raise ValueError(f"unknown fallback policy: {policy}")
```

### Call-site change at line 656

```python
result = self._execute_postonly_limit_buy_order(
    s, cap,
    wait_seconds=self.config.get("MAKER_WAIT_SECONDS", 3600),  # 1h default
    fallback=self.config.get("MAKER_FALLBACK", "skip"),
)
if result:
    oid = result["order_id"]
    px = result["fill_price"] or sig["current_price"]
    amt = result["fill_amount"] or (cap / px)
    self.tracker.record_buy(s, oid, px, amt, px * amt, 0, "USD")
    # ... rest unchanged: place trailing-stop exit
```

### Config knobs to add (run_config.py)

```python
# Maker-only entry mode. When True, entries are placed as post-only limit
# orders at current best bid with a wait/timeout loop. Set False to keep
# the legacy market-buy taker path.
"MAKER_ONLY_ENTRIES": False,

# Wait window before the limit order is canceled and the fallback policy
# fires. Per Q2 drift analysis, 1h captures 88-90% of fill opportunities
# on TRX/BTC and 80-85% on ETH/DOGE. Extending to 2-4h raises fill rate
# marginally but lets stale signals execute on price movement that no
# longer reflects the signal.
"MAKER_WAIT_SECONDS": 3600,  # 1 hour default

# Fallback policy when the post-only limit doesn't fill within the wait window.
# - "skip":          cancel and abandon the trade entirely
# - "reprice_once":  cancel, place ONE more post-only limit at fresh best bid
# - "market":        cancel and fall back to a taker market buy
"MAKER_FALLBACK": "skip",
```

---

## Wait-window and fallback options for review

### Wait window

Per Q2: at 1h, miss-up rate is 9-11% for TRX/BTC, 17-18% for ETH/DOGE. Beyond 1h the fill rate improves slowly (price mean-reverts in 4h windows but the marginal extra fills are small) while signal staleness rises monotonically. The drift analysis specifically measured a 1h-post-close window — that's the empirical anchor.

| Window | Pros | Cons |
|---|---|---|
| **30 min** | Fresh signal; clear fail-fast | Lower fill rate (extrapolated ~75-85%) |
| **1 hour (proposed default)** | Matches the drift analysis anchor; fill rate 80-90% across the universe | Reasonable for 4h-bar strategies |
| **2 hours** | Slightly higher fill rate (~85-92%) | Begins to overlap with signal staleness; still well within a 4h bar |
| **4 hours (one full bar)** | Highest fill rate (mean-reversion across a bar is common) | Stale signals execute after price has already moved meaningfully; defeats the signal premise |

**Recommendation: default `MAKER_WAIT_SECONDS = 3600` (1h)**. Tunable per coin via config if needed (e.g., TRX could afford 30min; DOGE might need 2h). Don't go beyond 2h — beyond that the signal is no longer trading the same market state.

### Fallback policy

| Policy | What happens at timeout | Implication |
|---|---|---|
| **`skip` (proposed default)** | Cancel, abandon the trade | The 10-20% "missed" trades are exactly the strongest-signal trades (price ran away because momentum was real). You lose those, but every executed trade pays maker fee. Cleanest discipline. |
| **`reprice_once`** | Cancel, place a fresh post-only limit at the new best bid, with same wait | Adds 1 hour of latency. Captures some of the runaway trades that came back. Risk: chasing — you're effectively bidding 0.5%+ above your original price after the signal aged. |
| **`market`** | Cancel, fall back to taker market buy | You capture all trades but pay taker (0.40%) on the worst ones. Hybrid — defeats maker discipline on the trades that need it most. |

**Recommendation: default `MAKER_FALLBACK = "skip"`**. The drift analysis showed that median post-signal drift is near zero, but the >0.5% upward-drift trades are the exact ones a `reprice_once` or `market` fallback would catch — and those are also the trades most likely to be following a real momentum burst that's already moved away from the original signal premise. Skipping them is methodologically clean: every executed maker trade is one where price stayed near the signal, which is what the strategy was designed to trade.

Alternative views worth discussing before implementation:

- **Hybrid `market` fallback for TRX/BTC only** — the maker-feasible subset has miss rates of 10-11%; falling back to taker on those means paying 0.40% × 10% = 4bp annualized drag on those coins. Could be acceptable if it raises trade-execution coverage from 90% to 100%.
- **Tiered wait/fallback by coin volatility** — TRX (low drift) uses 30min/skip; BTC uses 1h/skip; DOGE uses 2h/reprice. Configurable per-coin in `per_coin_params`.

### Edge cases to handle in implementation

1. **postOnly rejection on placement.** Kraken rejects with a specific error code if the limit would cross the book. Treat as immediate fallback (the spread crossed; market is faster than us).
2. **Partial fills.** A post-only limit can partially fill before the wait elapses. Decision: treat any non-zero fill as success and place the trailing-stop exit on the filled amount; cancel the remainder. Current code already handles partial fills downstream.
3. **Exchange canceling our order mid-wait.** Some venues cancel maker orders during volatile periods to prevent self-trade or to enforce post-only on republished levels. Handle as fallback.
4. **Trailing stop placement after maker entry.** The trailing-stop exit is taker on trigger. No way around this without converting to a maker-only exit (would require limit-sell that may not fill in a fast move). Out of scope for this proposal.
5. **State persistence.** The wait loop runs synchronously; if the trader process restarts during a wait, the order is left resting on Kraken. On startup, the engine should reconcile any open limit orders (the existing reconcile loop already handles this for trailing-stops; verify it handles limit-buys too).

---

## What this proposal does NOT cover

- **Exit-side maker.** Exits remain taker (trailing-stop triggers as market). A separate proposal would address maker exits via stop-limit, but those carry slippage-on-fill risk that's different from entry mechanics.
- **Exchange comparison / migration.** Q3 showed OKX/Bybit/Hyperliquid have 3-6× lower fees than Kraken Pro at $0 volume. Switching exchange is a separate decision and a larger lift (CCXT exchange object swap, withdrawal/deposit flow, key management).
- **Backtest fee-model alignment.** If maker-only goes live, the WFO `FEES` config should reflect the actual maker rate (0.25% currently in flight as the diagnostic) so research and live executions agree.
- **The cv-explosion methodology issue.** Independent of maker work. Deferred per prior decision.

---

## Open questions for review

1. **Wait window default**: 1h (proposed) or 2h?
2. **Fallback default**: `skip` (proposed), `reprice_once`, or `market`?
3. **Per-coin tuning**: should `MAKER_WAIT_SECONDS` and `MAKER_FALLBACK` be coin-specific from day one, or start global and tune later?
4. **Rollout**: ship behind `MAKER_ONLY_ENTRIES=False` flag and flip per-coin, or flip the universe at once?
5. **Live trader params**: the live trader is still on legacy pre-reset params. If maker-only goes live, does it apply to legacy params or wait for textbook-validated params from the cleaned + maker-rate run?
