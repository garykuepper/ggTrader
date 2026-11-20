"""The paper test package: importing it lifts the live-run refusal of a non-durable ledger.

The harnesses run live traders on in-memory ledgers, some at session scope and some in child
processes that import ``tests.paper...`` without pytest, so no fixture reaches them; the import
does. Production code never imports this package. ``test_durable_ledger.py`` turns the refusal
back on, per test, to prove it.
"""

from ggTrader.paper import ledger as _ledger

_ledger.ALLOW_NON_DURABLE_LEDGER = True
