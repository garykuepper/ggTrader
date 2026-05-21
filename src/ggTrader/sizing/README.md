# sizing

Position sizers: map (Signal, PortfolioState) → sized Signal. Implementations
land in **Phase 3**:
- `fixed_fraction.py`
- `vol_target.py`
- `kelly.py` (incl. fractional Kelly with confidence from meta-labels)
