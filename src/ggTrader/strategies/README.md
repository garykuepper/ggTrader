# strategies

Strategy implementations following the `Strategy` ABC (spec §5.7). Each
strategy is a pure function `(ts, feature_store) -> list[Signal]` — it does
not call brokers, place orders, or compute features inline.

Subpackages (created as each lands):
- `carry/` — cash-and-carry, funding arbitrage (Phase 3)
- `momentum/` — cross-sectional, trend-following (Phase 3+)
- `meanrev/` — cointegration pairs, extreme reversal
- `ml/` — meta-labeling, triple-barrier XGBoost
- `overlay/` — regime gates (HMM, BOCPD)

Legacy TA strategies stay in `indicators/strategies.py` until **Phase 7**
migrates them here.
