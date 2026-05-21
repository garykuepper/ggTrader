# risk

Position limits, drawdown kill switches, and live-vs-backtest divergence
detection. Populated in **Phase 6**:

- `limits.py` — per-strategy and portfolio-level position/exposure limits
- `monitor.py` — runtime alerting when live P&L diverges materially from
  the backtest distribution (e.g., regime-shift detection)
