# features

The Feature Catalog: named, time-indexed values derived from raw data,
cached in TimescaleDB and reusable across strategies (spec §5.6).

Populated in **Phase 2** by porting the indicators in `indicators/signals.py`
(EMA, MACD, RSI, BB, PSAR, ADX, ATR, etc.) into `Feature` subclasses, then
extended in Phase 3+ with `derivatives.py` (funding_z, basis_apr), `flow.py`
(CVD, OFI), `onchain.py`, `options.py`, `macro.py`, and `regime.py`.

Strategies declare features by name; they never compute features inline.
