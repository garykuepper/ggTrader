# Archived legacy-engine scripts (P5c)

These scripts were moved out of `scripts/` on 2026-11-06 because they import code that no longer exists in the main tree (the hand-rolled legacy engine: `lab/simulate.py`, `sweep.py`, `harness.py`, `wfo.py`, `metrics.py`, the legacy strategy classes and `strategies/indicators.py`) or drive strategies that were not ported (decision OD11(b)).

**They run only at git tag `legacy-engine-vbt0285` via `scripts/legacy_env.sh`** (vectorbt 0.28.5, pandas 2.3.3). Rebuild that tree with `scripts/legacy_env.sh /path/to/legacy-engine`, then run the script from the extracted `src/` with the legacy interpreter. Scripts written during the engine rebuild (marked *branch-only* below) were never in the tag; their only copy is this one, and they additionally need the legacy modules alongside the current `ggTrader.engine`, so they are evidence of how the numbers in `research/programme/vectorbt-engine/` were produced, not runnable tools.

Research reports that cite these scripts by `scripts/<name>` still name the right file: look here, or in the tag. Do not edit the copies; the reports and JSON next to them are the record. `engine_parity_evidence_p2_full.py` is the original P2 script; its live successor (indicators and voters sections) stays at `scripts/engine_parity_evidence.py`.

| Script | Why it is here |
|---|---|
| `ablation_voters.py` | closed voter ablation on legacy EnsembleSignal/lab.wfo |
| `anchor_fix_reproduction_wfo.py` | legacy WFO driver (lab.wfo/sweep/ensemble); numbers live in its research report |
| `audit_reruns_wfo.py` | legacy WFO driver for xs/dual momentum and fomc_drift reruns |
| `congress_trades_correlation_check.py` | closed NO-GO diversification check on unported congress_trades |
| `core_spy_blend.py` | legacy WFO + simulate driver; closed core+SPY blend study |
| `cross_asset_sleeve_wfo.py` | legacy WFO driver of the closed cross-asset study (sleeve is ported; the study is not rerun here) |
| `deployment_audit.py` | one-off capital-deployment audit on the legacy simulator |
| `fomc_clustered_event_study.py` | closed FOMC drift event study on unported fomc_drift/indicators |
| `fx_hedge_overlay_static_baseline_check.py` | closed NO-GO check of unported fx_hedge_overlay |
| `gate_value_ablation.py` | closed ML-gate ablation on legacy EnsembleSignal |
| `ic_kelly_rebaseline_wfo.py` | closed ensemble_ic/kelly rebaseline; imports anchor_fix_reproduction_wfo |
| `insider_cluster_correlation_check.py` | closed NO-GO check of unported insider_cluster_buy |
| `lever_diagnostic.py` | closed lever diagnostic on legacy wfo/simulate |
| `leveraged_rotation_research.py` | closed NO-GO leveraged rotation study (unported) |
| `leveraged_trend_research.py` | closed NO-GO leveraged trend study (unported); also a raw-vectorbt importer |
| `live_indicator_warmup.py` | one-off live-vs-backtest drift study on legacy EnsembleSignal/indicators |
| `max_effect_correlation_check.py` | closed NO-GO check of unported max_effect |
| `midcap_bias_calibration.py` | closed midcap survivorship calibration on legacy lab.wfo |
| `midcap_research.py` | closed MidCap 400 study on legacy lab.wfo |
| `ml_signal_screen.py` | closed ML pre-screen on legacy signals/indicators |
| `month_end_treasury_wfo.py` | closed A10 study on unported month_end_treasury |
| `new_stack_parity_compare.py` *(branch-only)* | P1 evidence: compares new-stack legacy-code runs with P0 references |
| `new_stack_reference_runs.py` *(branch-only)* | P1 parity driver running the legacy WFO code on the new stack; raw-vectorbt importer |
| `p4_strategy_parity.py` *(branch-only)* | P4 evidence driver: ported strategies vs legacy implementations (needs both) |
| `p5_equity_parity.py` *(branch-only)* | P5 evidence driver: single-pass simulation vs legacy lab.wfo._sweep_fold |
| `p5_tie_audit.py` *(branch-only)* | P5 evidence: tie audit built on p4_strategy_parity (archived with it) |
| `pairs_stat_arb_research.py` | closed NO-GO pairs study (unported) |
| `pead_correlation_check.py` | closed NO-GO check of unported pead |
| `position_sizing_regimes.py` | closed position-sizing study on legacy simulate; raw-vectorbt importer |
| `position_sizing_wfo.py` | closed position-sizing WFO on legacy lab.wfo/simulate; raw-vectorbt importer |
| `run_position_sizing_wfo_scheduled.sh` | launcher for position_sizing_wfo.py (archived with it) |
| `wfo_conviction_comparison.sh` | docker launcher of `ggt lab --strategy ensemble_conviction` (unported) |
| `engine_parity_evidence_p2_full.py` *(branch-only)* | the original P2 evidence script (engine presets, voters and simulator against the legacy code) |
| `mcp_server.py` *(P10, 2026-11-06)* | stale MCP server for the lab: lists the deleted `STRATEGY_REGISTRY`, caches into the production checkout, shells out to the `ggtrader_live` Docker image, `mcp` is not a project dependency, and nothing registers it. Not runnable; do not start it |
| `headline_sentiment_backfill.py`, `short_interest_backfill.py`, `short_volume_backfill.py`, `house_ptr_backfill.py`, `fomc_calendar_backfill.py` *(P10, 2026-11-06)* | Backfill drivers of the loaders archived in `../legacy-src/` (strategies that now exist only at the legacy tag). They import modules that no longer exist in the main tree; run them at tag `legacy-engine-vbt0285`. |
