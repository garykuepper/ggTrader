# Trading Strategy Pipeline Report

**Generated**: 2026-03-28 16:25:03

## Executive Summary

**WFO Training/Test period:** 2023-03-29 -> 2025-12-31  
**YTD performance window:** 2025-11-28 -> 2026-03-28  
**Coins:** 27

| | WFO Full Range | YTD |
|-|----------------|--------|
| Strategy CAGR | 13.17% | -13.40% |
| BTC buy & hold CAGR | 51.52% ❌ | -63.26% ✅ |
| S&P 500 CAGR | 22.61% ❌ | -17.86% ✅ |
| Strategy Sharpe | 1.08 | -0.80 |
| Max Drawdown | -6.93% | -9.90% |
| Total Trades | 129 | 36 |
| Win Rate | 44.96% | 19.44% |

### Full Range Portfolio

![Full Range Portfolio](plots/combined_portfolio_final_dashboard.png)

### YTD Portfolio

![YTD Portfolio](plots/combined_portfolio_ytd_dashboard.png)

## Result Validation (Training/Test Data)
**Period: 2023-03-29 -> 2025-12-31** — WFO-selected parameters replayed on the full training/test range.

### Combined Portfolio Performance

| Metric | Strategy | BTC (buy & hold) | S&P 500 (SPY) | Excess vs BTC |
|--------|----------|------------------|---------------|---------------|
| Total Return | 40.73% | 215.11% | 75.60% | - |
| CAGR | 13.17% | 51.52% | 22.61% | -38.35% |
| Sharpe Ratio | 1.0813 | 1.1401 | 1.2570 | - |
| Max Drawdown | -6.93% | -34.46% | -14.53% | - |
| Total Trades | 129 | 1 | 1 | - |
| Win Rate | 44.96% | - | - | - |

## YTD Performance
**Period: 2025-11-28 -> 2026-03-28** — WFO-selected parameters replayed on the YTD window, no re-optimization.

| Metric | Strategy | BTC (buy & hold) | S&P 500 (SPY) | Excess vs BTC |
|--------|----------|------------------|---------------|---------------|
| Total Return | -4.62% | -28.04% | -6.26% | - |
| CAGR | -13.40% | -63.26% | -17.86% | 49.86% |
| Sharpe Ratio | -0.8006 | -1.8338 | -2.8335 | - |
| Max Drawdown | -9.90% | -35.36% | -6.26% | - |
| Total Trades | 36 | 1 | 1 | - |
| Win Rate | 19.44% | - | - | - |

## Per-Coin Strategy Selection (WFO)

Best performing strategy per coin based on robustness scores (highest first).

| Symbol | Strategy | Selection | Robustness Score |
|--------|----------|-----------|------------------|
| KAS-USD | rsi_reversal+atr_trailing | wfo_robustness | 1.2260 |
| SPX-USD | supertrend_flip+fixed_sl_tp | wfo_robustness | 0.9987 |
| ZRO-USD | psar_adx+atr_trailing | wfo_robustness | 0.8891 |
| TAO-USD | psar_adx+atr_trailing | wfo_robustness | 0.6579 |
| ADA-USD | supertrend_flip+fixed_sl_tp | wfo_robustness | 0.6552 |
| PEPE-USD | supertrend_flip+fixed_sl_tp | wfo_robustness | 0.6306 |
| XLM-USD | rsi_reversal+trailing_stop | wfo_robustness | 0.6112 |
| ETH-USD | ema_cross+atr_trailing | wfo_robustness | 0.5938 |
| ZEC-USD | ema_cross+atr_trailing | wfo_robustness | 0.5694 |
| SOL-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.5396 |
| LINK-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.5019 |
| DOGE-USD | psar_adx+trailing_stop | wfo_robustness | 0.4960 |
| WIF-USD | psar_adx+trailing_stop | wfo_robustness | 0.4791 |
| AVAX-USD | supertrend_flip+atr_trailing | wfo_robustness | 0.4489 |
| UNI-USD | macd_cross+atr_trailing | wfo_robustness | 0.4412 |
| AKT-USD | ema_cross+atr_trailing | wfo_robustness | 0.4311 |
| NEAR-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.4242 |
| DOT-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.3758 |
| ONDO-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.3693 |
| CRV-USD | psar_adx+atr_trailing | wfo_robustness | 0.3538 |
| XRP-USD | ema_cross+atr_trailing | wfo_robustness | 0.3349 |
| XMR-USD | ema_cross+atr_trailing | wfo_robustness | 0.3312 |
| LTC-USD | psar_adx+atr_trailing | wfo_robustness | 0.2776 |
| BTC-USD | psar_adx+fixed_sl_tp | wfo_robustness | 0.2484 |
| SUI-USD | ema_cross+trailing_stop | wfo_robustness | 0.1928 |
| FET-USD | supertrend_flip+atr_trailing | wfo_robustness | 0.1509 |
| TRX-USD | donchian_breakout+atr_trailing | wfo_robustness | 0.1164 |

### WFO Fold Timeline

```mermaid
gantt
    title Walk-Forward Folds (Step = Test Length)
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m
    
    section Fold 4
    Train  :active, f4_tr, 2024-02-28, 2025-01-29
    Test   :crit, f4_ts, 2025-01-29, 2025-05-21
    
    section Fold 5
    Train  :active, f5_tr, 2024-06-19, 2025-05-21
    Test   :crit, f5_ts, 2025-05-21, 2025-09-10
    
    section Fold 6
    Train  :active, f6_tr, 2024-10-09, 2025-09-10
    Test   :crit, f6_ts, 2025-09-10, 2025-12-31
    
```

### WFO Out-of-Sample Sharpe — Per Fold

Per-fold OOS Sharpe for each coin's winning strategy (folds ordered as above). Negative = strategy did not generalise.

| Symbol | Strategy+Exit | IS Rob | OOS Rob | Consistency | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 |
|--------|---------------|--------|---------|-------------|--------|--------|--------|--------|--------|--------|
| KAS-USD | rsi_reversal+atr_trailing | 1.9212 | 0.8517 | 100% | n/a | n/a | n/a | 1.844 | 0.229 | n/a |
| SPX-USD | supertrend_flip+fixed_sl_tp | 2.6952 | 0.5974 | 67% | n/a | n/a | n/a | 0.851 | -0.075 | 1.039 |
| ZRO-USD | psar_adx+atr_trailing | 2.3297 | 0.5693 | 67% | n/a | n/a | 0.742 | n/a | 1.383 | -0.146 |
| ETH-USD | ema_cross+atr_trailing | 1.8079 | 0.4883 | 50% | -0.819 | -2.534 | 1.392 | 2.722 | 2.638 | -1.463 |
| ZEC-USD | ema_cross+atr_trailing | 1.9071 | 0.3746 | 50% | n/a | n/a | -0.818 | 0.649 | -1.712 | 2.880 |
| ADA-USD | supertrend_flip+fixed_sl_tp | 2.7607 | 0.3463 | 40% | -3.414 | -2.389 | 3.734 | -0.760 | 1.584 | n/a |
| TAO-USD | psar_adx+atr_trailing | 1.9510 | 0.2991 | 67% | n/a | n/a | 1.178 | -0.911 | 0.818 | n/a |
| WIF-USD | psar_adx+trailing_stop | 1.1682 | 0.2380 | 80% | 0.523 | 0.187 | -0.644 | 0.546 | n/a | 0.469 |
| PEPE-USD | supertrend_flip+fixed_sl_tp | 2.2259 | 0.1874 | 60% | 2.097 | n/a | 1.732 | 2.381 | -0.999 | -1.545 |
| XMR-USD | ema_cross+atr_trailing | 1.2585 | 0.1377 | 50% | -1.889 | -2.705 | 0.889 | 3.266 | -2.235 | 1.268 |
| XLM-USD | rsi_reversal+trailing_stop | 2.6898 | 0.0560 | 50% | -2.663 | -3.844 | 3.448 | 1.412 | 2.384 | -2.521 |
| AVAX-USD | supertrend_flip+atr_trailing | 1.7607 | 0.0385 | 60% | n/a | -1.459 | 1.404 | 1.019 | -2.048 | 1.006 |
| CRV-USD | psar_adx+atr_trailing | 1.6566 | -0.0212 | 50% | n/a | -2.876 | n/a | 0.201 | 1.994 | -0.750 |
| DOGE-USD | psar_adx+trailing_stop | 2.0660 | -0.0224 | 60% | 0.066 | n/a | 3.060 | -0.752 | 1.133 | -2.224 |
| XRP-USD | ema_cross+atr_trailing | 2.0033 | -0.0482 | 33% | -2.849 | -0.830 | 4.267 | -0.824 | 0.995 | -1.816 |
| DOT-USD | rsi_reversal+atr_trailing | 1.8885 | -0.0918 | 50% | -4.594 | 1.319 | 1.236 | 0.940 | -0.897 | -0.412 |
| UNI-USD | macd_cross+atr_trailing | 2.1374 | -0.1812 | 60% | n/a | -2.160 | 1.193 | 1.293 | 0.924 | -2.249 |
| LINK-USD | rsi_reversal+atr_trailing | 2.7172 | -0.2276 | 50% | -1.497 | -3.233 | 1.858 | 0.082 | 1.224 | -1.497 |
| BTC-USD | psar_adx+fixed_sl_tp | 1.9345 | -0.3469 | 40% | n/a | -1.563 | 0.904 | 0.508 | -0.582 | -1.098 |
| NEAR-USD | rsi_reversal+atr_trailing | 2.6114 | -0.3619 | 50% | n/a | n/a | 1.744 | 1.639 | -1.604 | -2.135 |
| SOL-USD | rsi_reversal+atr_trailing | 2.7486 | -0.3730 | 67% | 0.163 | -1.011 | 1.657 | 0.162 | 0.360 | -2.626 |
| AKT-USD | ema_cross+atr_trailing | 2.7405 | -0.4145 | 50% | n/a | n/a | 1.125 | 0.921 | -0.751 | -2.187 |
| ONDO-USD | rsi_reversal+atr_trailing | 2.5152 | -0.5428 | 60% | n/a | 1.773 | 1.036 | -2.726 | 0.638 | -2.237 |
| LTC-USD | psar_adx+atr_trailing | 2.2897 | -0.6228 | 60% | 0.271 | n/a | -3.326 | 0.096 | 0.530 | -1.232 |
| TRX-USD | donchian_breakout+atr_trailing | 1.7422 | -0.6515 | 50% | -3.290 | -2.329 | 1.108 | 0.904 | 0.752 | -3.147 |
| FET-USD | supertrend_flip+atr_trailing | 2.2711 | -0.8513 | 50% | n/a | n/a | 0.229 | 1.039 | -2.589 | -1.838 |
| SUI-USD | ema_cross+trailing_stop | 2.8251 | -0.9281 | 33% | -3.628 | -0.524 | 1.275 | 0.045 | -0.004 | -3.886 |

## Full-Period Performance (Per-Coin)

Performance metrics from running WFO-selected parameters on full-range data.

| Symbol | Strategy | Selection | Return % | Sharpe | Max DD % | Trades | Win Rate % |
|--------|----------|-----------|----------|--------|----------|--------|-----------|
| SOL-USD | rsi_reversal | wfo_robustness | 16.50% | 1.1631 | -3.91% | 21 | 57.14% |
| TAO-USD | psar_adx | wfo_robustness | 17.12% | 1.0199 | -4.72% | 15 | 73.33% |
| ADA-USD | supertrend_flip | wfo_robustness | 6.11% | 0.8004 | -2.25% | 8 | 37.50% |
| BTC-USD | psar_adx | wfo_robustness | 4.20% | 0.7622 | -2.33% | 15 | 40.00% |
| ZRO-USD | psar_adx | wfo_robustness | 6.96% | 0.5723 | -5.02% | 14 | 50.00% |
| DOGE-USD | psar_adx | wfo_robustness | 6.89% | 0.5413 | -6.30% | 19 | 63.16% |
| LINK-USD | rsi_reversal | wfo_robustness | 6.06% | 0.5234 | -4.99% | 20 | 45.00% |
| ETH-USD | ema_cross | wfo_robustness | 2.31% | 0.3540 | -4.70% | 16 | 37.50% |
| AVAX-USD | supertrend_flip | wfo_robustness | 1.10% | 0.3304 | -1.48% | 1 | 100.00% |
| XRP-USD | ema_cross | wfo_robustness | 1.32% | 0.1715 | -3.71% | 17 | 29.41% |
| TRX-USD | donchian_breakout | wfo_robustness | 0.34% | 0.0686 | -3.08% | 18 | 38.89% |
| LTC-USD | psar_adx | wfo_robustness | 0.13% | 0.0375 | -2.57% | 8 | 37.50% |
| FET-USD | supertrend_flip | wfo_robustness | 0.01% | 0.0116 | -3.15% | 3 | 33.33% |
| NEAR-USD | rsi_reversal | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| SUI-USD | ema_cross | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| ZEC-USD | ema_cross | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| DOT-USD | rsi_reversal | wfo_robustness | -1.34% | -0.1717 | -6.13% | 18 | 27.78% |
| XLM-USD | rsi_reversal | wfo_robustness | -1.60% | -0.4356 | -3.74% | 12 | 25.00% |
| PEPE-USD | supertrend_flip | wfo_robustness | -1.91% | -0.7322 | -2.30% | 2 | 0.00% |
| XMR-USD | ema_cross | wfo_robustness | -3.87% | -1.1709 | -3.87% | 15 | 6.67% |

---

*For pipeline methodology, fold structure, and benchmark definitions see [docs/UNIFIED_PIPELINE.md](../docs/UNIFIED_PIPELINE.md).*

---

*Report generated by ggTrader Pipeline*