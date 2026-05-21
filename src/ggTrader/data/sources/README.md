# data/sources

Concrete adapters implementing the `BarDataSource` / `TradeDataSource` /
`FundingRateDataSource` / etc. Protocols (spec §5.5). One file per provider:
`kraken_spot.py`, `kraken_futures.py`, `alpaca.py`, `eodhd.py`, `deribit.py`,
`glassnode.py`, `cryptoquant.py`, `coinalyze.py`.

Populated incrementally starting in **Phase 1** (broker abstraction wraps the
existing `data/live/exchange_loader.py`) and **Phase 4** (Alpaca + EODHD for
equities).
