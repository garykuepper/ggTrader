# execution

Concrete `Broker` adapters (spec §5.9). Populated in **Phase 1** by wrapping
the existing ccxt code from `data/live/exchange_loader.py` and
`core/crypto_execution_engine.py`:

- `kraken_spot.py` — Phase 1
- `kraken_futures.py` — Phase 1
- `alpaca.py` — Phase 4 (equities)
- `kraken_securities.py` — Phase 4 (stub, pending API)
- `paper.py` — Phase 5 (paper broker with configurable slippage)
