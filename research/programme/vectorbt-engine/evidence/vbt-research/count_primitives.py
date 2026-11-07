"""Read-only: measure line spans of functions classified as hand-rolled primitives."""
import ast, sys
from pathlib import Path

ROOT = Path("/home/flynn/ggTrader/src/ggTrader")
# (file relative to ROOT, qualified name) ; qualified "Class.method" or "func"
PRIMS = {
 "lab/metrics.py": ["curve_stats", "benchmark"],
 "lab/simulate.py": ["compute_vol_scalar", "compute_atr_stop"],
 "lab/harness.py": ["leak_check"],
 "lab/wfo.py": ["compute_wfe", "check_circuit_breaker", "check_shadow_reentry", "generate_folds",
                "_min_max_normalize", "composite_score", "_extract_grid_arrays", "_compute_dsr_inputs",
                "_pick_live_winner", "compute_anchor_set"],
 "lab/gates.py": ["_neighbor_flat_indices", "ndh_check", "_sr_standard_error", "psr", "expected_max_sr", "dsr_check"],
 "lab/sweep.py": ["build_grid", "_is_valid_combo", "pit_entry_mask"],
 "lab/data.py": ["collapse_daily_duplicates", "rebalance_dates", "eligible_at"],
 "lab/strategies/indicators.py": ["eligibility_mask", "bb_signals", "overnight_gap_signals", "rsi_signals",
     "ema_signals", "bb_strength", "rsi_strength", "ema_strength", "macd_signals", "macd_strength",
     "volume_bb_signals", "volume_bb_strength", "_weekly_rsi", "mtf_signals", "mtf_strength",
     "rsi_raw", "bb_raw", "ema_raw", "macd_raw", "vbb_raw"],
 "lab/strategies/signals.py": ["_vectorized_ema_tournament", "WfoTournamentSignal.to_targets"],
 "lab/strategies/ensemble.py": ["EnsembleSignal._generate_signals", "EnsembleSignal._apply_time_stop",
     "EnsembleConvictionSignal._generate_signals_with_sizes"],
 "lab/strategies/ensemble_ic.py": ["EnsembleICSignal._entries_exits_raw", "EnsembleICSignal._generate_signals"],
 "lab/strategies/ic_weights.py": ["forward_returns", "daily_cross_sectional_ic", "ic_weight_schedule"],
 "lab/kelly.py": ["_symbol_trades", "extract_trades", "expanding_kelly_fraction", "kelly_fraction_asof", "kelly_sizes"],
 "lab/strategies/ensemble_kelly.py": ["EnsembleKellySignal._generate_signals_with_sizes"],
 "lab/strategies/conviction.py": ["ConvictionBBSignal._compute_conviction_sizes"],
 "lab/strategies/momentum.py": ["CrossSectionalMomentum.select", "CrossSectionalMomentum.to_targets"],
 "lab/strategies/idio_vol.py": ["idiosyncratic_variance"],
 "lab/strategies/max_effect.py": ["trailing_max_return"],
 "lab/strategies/pairs_stat_arb.py": ["pairwise_correlations", "spread_zscore", "pair_signal"],
 "lab/strategies/commodity_trend.py": ["cross_sectional_momentum_scores", "regime_vol_zscore"],
 "lab/strategies/leveraged_trend.py": ["_trend_state", "_entries_exits"],
 "lab/strategies/leveraged_rotation.py": ["compute_breadth", "rotate_positions"],
 "lab/strategies/cross_asset_sleeve.py": ["leg_is_on"],
 "lab/strategies/treasury_curve.py": ["curve_slope_zscore", "curve_regime_weights"],
 "lab/strategies/fx_hedge_overlay.py": ["rolling_zscore", "real_fx_index", "trend_signal", "unhedged_weight"],
 "lab/strategies/month_end_treasury.py": ["month_end_signals"],
 "lab/strategies/fomc_drift.py": ["_event_signals"],
 "lab/strategies/index_deletion.py": ["deletion_events"],
 "lab/strategies/short_interest.py": ["ShortInterestStrategy._days_to_cover_trend"],
 "lab/strategies/short_volume_ratio.py": ["ShortVolumeRatioStrategy._avg_ratios"],
 "lab/strategies/retail_attention.py": ["RetailAttentionStrategy._spikes"],
 "lab/strategies/pead.py": ["PeadStrategy._most_recent_surprise"],
 "lab/strategies/insider_cluster.py": ["cluster_events"],
 "lab/allocation.py": ["trailing_realized_vol", "inverse_vol_weights", "target_vol_scale", "combine_sleeves"],
 "lab/blend.py": ["blend_curves"],
 "lab/regime.py": ["classify_regime", "compute_regime_scalar"],
 "lab/significance.py": ["newey_west_se", "excess_stats", "years_for_t", "passes_pretax_bar"],
 "lab/event_study.py": ["in_position_mask", "split_event_returns", "event_study_test"],
 "lab/momentum_screen.py": ["net_excess", "nw_t", "one_sided_p", "holm", "cagr_diff_pp", "capm",
     "_relative_wealth", "relative_drawdown", "worst_rolling_relative"],
 "lab/factor_screen.py": ["value_momentum_combo"],
 "lab/replica.py": ["run_replica"],
 "lab/a23_rebalance.py": ["Book", "_trade_to", "band_trigger", "check_rebalance"],
 "lab/century_leverage.py": ["build_legs", "simulate"],
 "lab/tax_lots.py": ["TaxLotBook"],
 "lab/train_gate.py": ["_build_dataset"],
 "paper/risk.py": ["RiskGuard"],
 "paper/overlay.py": ["compute_sleeve_curve", "compute_weights_and_scale"],
 "paper/catastrophe_stop.py": ["unrealized_pct", "find_catastrophe_stops"],
 "paper/cash_sweep.py": ["compute_sweep_buy", "compute_sweep_sell_for_funding", "estimate_prospective_buy_notional"],
 "paper/feature_gate.py": ["extract_features"],
 "paper/split_check.py": ["compute_split_corrections", "apply_corrections_to_positions"],
 "paper/dividend_check.py": ["compute_dividend_accruals"],
}

def spans(path):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = (node.lineno, node.end_lineno)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        out[f"{node.name}.{sub.name}"] = (sub.lineno, sub.end_lineno)
    return out

total = 0
missing = []
for rel, names in PRIMS.items():
    sp = spans(ROOT / rel)
    for n in names:
        if n not in sp:
            missing.append((rel, n)); continue
        a, b = sp[n]
        total += b - a + 1
print("TOTAL_PRIMITIVE_LINES", total)
print("MISSING", missing)
