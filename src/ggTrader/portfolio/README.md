# portfolio

Multi-strategy combiners and risk overlay. Implementations land in **Phase 3+**:
- `simple.py` — SimpleAggregator (sum of sized signals)
- `hrp.py` — Hierarchical Risk Parity
- `risk_parity.py` — vol-targeted risk parity

Converts sized signals into Orders accounting for current positions, risk
limits, and available capital.
