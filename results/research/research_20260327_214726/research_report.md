# Trading Strategy Pipeline Report

**Generated**: 2026-03-27 22:01:27

## Executive Summary

**WFO Training/Test period:** 2023-01-01 -> 2025-12-30  
**YTD performance window:** 2025-03-28 -> 2026-03-28  
**Coins:** 34

| | WFO Full Range | YTD |
|-|----------------|--------|
| Strategy CAGR | 12.84% | 0.41% |
| BTC buy & hold CAGR | 74.01% ❌ | -28.15% ✅ |
| S&P 500 CAGR | 22.47% ❌ | 15.43% ❌ |
| Strategy Sharpe | 1.14 | 0.15 |
| Max Drawdown | -9.82% | -8.35% |
| Total Trades | 124 | 38 |
| Win Rate | 50.00% | 28.95% |

### Full Range Portfolio

![Full Range Portfolio](plots/combined_portfolio_final_dashboard.png)

### YTD Portfolio

![YTD Portfolio](plots/combined_portfolio_ytd_dashboard.png)

## Result Validation (Training/Test Data)
**Period: 2023-01-01 -> 2025-12-30** — WFO-selected parameters replayed on the full training/test range.

### Combined Portfolio Performance

| Metric | Strategy | BTC (buy & hold) | S&P 500 (SPY) | Excess vs BTC |
|--------|----------|------------------|---------------|---------------|
| Total Return | 43.64% | 426.04% | 83.58% | - |
| CAGR | 12.84% | 74.01% | 22.47% | -61.17% |
| Sharpe Ratio | 1.1367 | 1.4255 | 1.2852 | - |
| Max Drawdown | -9.82% | -34.46% | -14.53% | - |
| Total Trades | 124 | 1 | 1 | - |
| Win Rate | 50.00% | - | - | - |

## YTD Performance
**Period: 2025-03-28 -> 2026-03-28** — WFO-selected parameters replayed on the YTD window, no re-optimization.

| Metric | Strategy | BTC (buy & hold) | S&P 500 (SPY) | Excess vs BTC |
|--------|----------|------------------|---------------|---------------|
| Total Return | 0.41% | -28.15% | 15.42% | - |
| CAGR | 0.41% | -28.15% | 15.43% | 28.56% |
| Sharpe Ratio | 0.1530 | -1.7830 | 0.9163 | - |
| Max Drawdown | -8.35% | -35.36% | -10.49% | - |
| Total Trades | 38 | 1 | 1 | - |
| Win Rate | 28.95% | - | - | - |

## Per-Coin Strategy Selection (WFO)

Best performing strategy per coin based on robustness scores (highest first).

| Symbol | Strategy | Selection | Robustness Score |
|--------|----------|-----------|------------------|
| USELESS-USD | ema_cross+atr_trailing | wfo_robustness | 2.3418 |
| PENGU-USD | psar_adx+fixed_sl_tp | wfo_robustness | 1.6839 |
| PUMP-USD | rsi_reversal+atr_trailing | wfo_robustness | 1.3710 |
| ZEC-USD | ema_cross+atr_trailing | wfo_robustness | 0.9396 |
| TAO-USD | psar_adx+atr_trailing | wfo_robustness | 0.8003 |
| TRUMP-USD | ema_cross+trailing_stop | wfo_robustness | 0.7624 |
| BNB-USD | psar_adx+atr_trailing | wfo_robustness | 0.7170 |
| ETH-USD | supertrend_flip+atr_trailing | wfo_robustness | 0.7094 |
| HBAR-USD | macd_cross+fixed_sl_tp | wfo_robustness | 0.6234 |
| LINK-USD | supertrend_flip+atr_trailing | wfo_robustness | 0.5949 |
| XRP-USD | psar_adx+atr_trailing | wfo_robustness | 0.5760 |
| ADA-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.5583 |
| VIRTUAL-USD | psar_adx+atr_trailing | wfo_robustness | 0.5368 |
| SPX-USD | psar_adx+fixed_sl_tp | wfo_robustness | 0.5333 |
| SOL-USD | ema_cross+atr_trailing | wfo_robustness | 0.5235 |
| TRX-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.4910 |
| FARTCOIN-USD | supertrend_flip+trailing_stop | wfo_robustness | 0.4545 |
| WIF-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.4507 |
| XMR-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.3821 |
| NEAR-USD | ema_cross+fixed_sl_tp | wfo_robustness | 0.3611 |
| XLM-USD | psar_adx+trailing_stop | wfo_robustness | 0.3593 |
| DOGE-USD | macd_cross+atr_trailing | wfo_robustness | 0.3451 |
| AVAX-USD | donchian_breakout+atr_trailing | wfo_robustness | 0.3196 |
| SUI-USD | ema_cross+fixed_sl_tp | wfo_robustness | 0.3128 |
| BTC-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.2560 |
| PEPE-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.2540 |
| ONDO-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.2437 |
| AAVE-USD | ema_cross+trailing_stop | wfo_robustness | 0.1900 |
| CRV-USD | donchian_breakout+atr_trailing | wfo_robustness | 0.1854 |
| KAS-USD | supertrend_flip+atr_trailing | wfo_robustness | 0.1844 |
| UNI-USD | psar_adx+fixed_sl_tp | wfo_robustness | 0.1763 |
| XCN-USD | rsi_reversal+atr_trailing | wfo_robustness | 0.1616 |
| ICP-USD | psar_adx+fixed_sl_tp | wfo_robustness | 0.1511 |
| FET-USD | psar_adx+trailing_stop | wfo_robustness | 0.1241 |

### WFO Fold Timeline

```mermaid
gantt
    title Walk-Forward Folds (Step = Test Length)
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m
    
    section Fold 6
    Train  :active, f6_tr, 2024-09-01, 2025-08-30
    Test   :crit, f6_ts, 2025-08-30, 2025-12-30
    
```

### WFO Out-of-Sample Sharpe — Per Fold

Per-fold OOS Sharpe for each coin's winning strategy (folds ordered as above). Negative = strategy did not generalise.

| Symbol | Strategy+Exit | IS Rob | OOS Rob | Consistency | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 |
|--------|---------------|--------|---------|-------------|--------|--------|--------|--------|--------|--------|
| USELESS-USD | ema_cross+atr_trailing | n/a | 2.3418 | 100% | n/a | n/a | n/a | n/a | n/a | 2.342 |
| PENGU-USD | psar_adx+fixed_sl_tp | 1.3703 | 1.8528 | 100% | n/a | n/a | n/a | 1.490 | 2.040 | n/a |
| PUMP-USD | rsi_reversal+atr_trailing | n/a | 1.3710 | 100% | n/a | n/a | n/a | n/a | n/a | 1.371 |
| ETH-USD | supertrend_flip+atr_trailing | 1.0647 | 0.6739 | 83% | 0.010 | 0.770 | 1.769 | -3.405 | 4.079 | 0.759 |
| TAO-USD | psar_adx+atr_trailing | 1.8878 | 0.6252 | 67% | n/a | n/a | 1.326 | -0.164 | 0.918 | n/a |
| ZEC-USD | ema_cross+atr_trailing | 2.5053 | 0.5783 | 67% | n/a | n/a | n/a | 0.212 | -1.058 | 2.544 |
| BNB-USD | psar_adx+atr_trailing | 2.4254 | 0.4590 | 50% | n/a | n/a | n/a | n/a | -0.830 | 1.740 |
| XCN-USD | rsi_reversal+atr_trailing | n/a | 0.3232 | 33% | n/a | n/a | n/a | 3.424 | -1.510 | -0.085 |
| ADA-USD | rsi_reversal+atr_trailing | 2.3223 | 0.3113 | 40% | -1.631 | -1.581 | 2.722 | n/a | 1.234 | -0.368 |
| XLM-USD | psar_adx+trailing_stop | 1.0835 | 0.3010 | 50% | -0.219 | -2.689 | 3.013 | n/a | 0.230 | n/a |
| SPX-USD | psar_adx+fixed_sl_tp | 1.4893 | 0.2079 | 75% | n/a | n/a | 1.640 | -1.898 | 0.452 | 0.801 |
| LINK-USD | supertrend_flip+atr_trailing | 1.9018 | 0.1964 | 67% | -0.622 | 0.396 | 0.952 | -0.818 | 0.681 | 0.277 |
| HBAR-USD | macd_cross+fixed_sl_tp | 2.4983 | 0.1892 | 50% | n/a | n/a | n/a | n/a | 1.019 | -0.487 |
| SOL-USD | ema_cross+atr_trailing | 1.7893 | 0.1871 | 60% | n/a | -1.182 | 0.651 | -1.260 | 0.894 | 1.031 |
| VIRTUAL-USD | psar_adx+atr_trailing | 2.1283 | 0.1754 | 50% | n/a | n/a | n/a | n/a | -0.494 | 0.789 |
| XMR-USD | rsi_reversal+atr_trailing | 1.5663 | 0.0972 | 50% | -1.551 | 0.110 | -0.474 | 0.244 | -0.678 | 1.488 |
| AVAX-USD | donchian_breakout+atr_trailing | 1.5681 | 0.0495 | 40% | -0.886 | -1.165 | 1.051 | n/a | -0.859 | 1.049 |
| TRX-USD | rsi_reversal+atr_trailing | 2.1535 | 0.0490 | 50% | 1.271 | -0.914 | 1.069 | -0.520 | 0.765 | -0.660 |
| TRUMP-USD | ema_cross+trailing_stop | 2.8240 | 0.0432 | 67% | n/a | n/a | n/a | 0.208 | -0.256 | 0.171 |
| XRP-USD | psar_adx+atr_trailing | 1.8801 | 0.0302 | 80% | 0.072 | 0.151 | 4.252 | -3.428 | 0.151 | n/a |
| WIF-USD | rsi_reversal+atr_trailing | 1.9062 | -0.0358 | 60% | n/a | 1.824 | -0.415 | 0.118 | 0.348 | -1.082 |
| ICP-USD | psar_adx+fixed_sl_tp | 0.8896 | -0.1071 | 50% | 1.177 | -3.507 | -0.448 | -2.127 | 1.492 | 1.067 |
| BTC-USD | rsi_reversal+atr_trailing | 1.7165 | -0.1367 | 33% | 2.868 | -3.269 | 1.564 | -0.253 | -0.157 | -0.571 |
| DOGE-USD | macd_cross+atr_trailing | 1.6173 | -0.1630 | 67% | 0.929 | -1.901 | 2.515 | -4.575 | 0.853 | 0.854 |
| FARTCOIN-USD | supertrend_flip+trailing_stop | 2.4172 | -0.1827 | 50% | n/a | n/a | n/a | n/a | -1.388 | 0.793 |
| UNI-USD | psar_adx+fixed_sl_tp | 1.3899 | -0.2058 | 33% | -1.921 | 0.527 | -0.344 | -0.852 | 1.323 | -0.916 |
| CRV-USD | donchian_breakout+atr_trailing | 1.4516 | -0.2631 | 40% | n/a | -1.790 | 1.440 | -2.080 | 0.586 | -0.189 |
| ONDO-USD | rsi_reversal+atr_trailing | 1.8688 | -0.3246 | 40% | n/a | -0.285 | 2.564 | -3.925 | -0.094 | 0.126 |
| NEAR-USD | ema_cross+fixed_sl_tp | 2.1254 | -0.4037 | 67% | 0.947 | 0.909 | 0.445 | -1.847 | -1.742 | 0.143 |
| SUI-USD | ema_cross+fixed_sl_tp | 2.3463 | -0.4933 | 50% | 2.535 | -2.557 | 2.003 | 0.605 | -1.294 | -2.272 |
| FET-USD | psar_adx+trailing_stop | 1.6977 | -0.5322 | 33% | 2.028 | -0.974 | -0.803 | -2.769 | 1.444 | -1.366 |
| PEPE-USD | rsi_reversal+atr_trailing | 2.3996 | -0.5817 | 40% | 2.552 | -1.091 | n/a | -3.886 | 0.495 | -0.186 |
| AAVE-USD | ema_cross+trailing_stop | 2.5863 | -0.8610 | 40% | -3.434 | -1.310 | 2.149 | n/a | 0.040 | -3.096 |
| KAS-USD | supertrend_flip+atr_trailing | 2.6738 | -0.8724 | 33% | n/a | n/a | n/a | -0.811 | 0.785 | -2.791 |

## Full-Period Performance (Per-Coin)

Performance metrics from running WFO-selected parameters on full-range data.

| Symbol | Strategy | Selection | Return % | Sharpe | Max DD % | Trades | Win Rate % |
|--------|----------|-----------|----------|--------|----------|--------|-----------|
| BTC-USD | rsi_reversal | wfo_robustness | 10.75% | 1.2506 | -4.00% | 22 | 36.36% |
| SOL-USD | ema_cross | wfo_robustness | 19.57% | 1.1576 | -5.32% | 27 | 48.15% |
| TAO-USD | psar_adx | wfo_robustness | 17.25% | 1.0020 | -4.72% | 15 | 73.33% |
| AVAX-USD | donchian_breakout | wfo_robustness | 6.12% | 0.7052 | -2.52% | 5 | 60.00% |
| ADA-USD | rsi_reversal | wfo_robustness | 3.79% | 0.5413 | -2.88% | 12 | 41.67% |
| LINK-USD | supertrend_flip | wfo_robustness | 6.95% | 0.5387 | -4.99% | 22 | 45.45% |
| TRX-USD | rsi_reversal | wfo_robustness | 2.49% | 0.3799 | -2.35% | 19 | 57.89% |
| ETH-USD | supertrend_flip | wfo_robustness | 2.39% | 0.3253 | -4.90% | 21 | 38.10% |
| FET-USD | psar_adx | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| NEAR-USD | ema_cross | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| SUI-USD | ema_cross | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| ZEC-USD | ema_cross | wfo_robustness | 0.00% | 0.0000 | 0.00% | 0 | 0.00% |
| DOGE-USD | macd_cross | wfo_robustness | -2.98% | -0.0886 | -12.67% | 63 | 26.98% |
| XLM-USD | psar_adx | wfo_robustness | -0.21% | -0.1050 | -0.69% | 3 | 33.33% |
| XMR-USD | rsi_reversal | wfo_robustness | -2.37% | -0.3355 | -4.22% | 14 | 28.57% |
| HBAR-USD | macd_cross | wfo_robustness | -2.59% | -0.4425 | -5.08% | 15 | 20.00% |
| PEPE-USD | rsi_reversal | wfo_robustness | -1.75% | -0.7106 | -2.05% | 1 | 0.00% |
| XRP-USD | psar_adx | wfo_robustness | -1.58% | -0.7862 | -1.58% | 3 | 33.33% |
| FARTCOIN-USD | supertrend_flip | wfo_robustness | -18.80% | -1.1184 | -21.48% | 51 | 15.69% |

---

*For pipeline methodology, fold structure, and benchmark definitions see [docs/UNIFIED_PIPELINE.md](../docs/UNIFIED_PIPELINE.md).*

---

*Report generated by ggTrader Pipeline*