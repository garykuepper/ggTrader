# Crypto Execution Venue Comparison: Alpaca vs. Kraken Pro vs. Binance.US

**Classification:** Internal Quantitative Research & Broker/Exchange Infrastructure Strategy
**Date:** 2026-07-06
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Overview & Objective

This document outlines the operational and fee structures of **Alpaca Crypto**, **Kraken Pro**, and **Binance.US**. The objective is to establish guidelines for how transaction costs (commissions and spreads) and liquidity should be modeled in the walk-forward optimization (WFO) backtester (`vectorbt` pipeline) and to evaluate potential live-execution venues.

## 2. Infrastructure & Fee Comparison

| Platform | Fee Structure Model | Maker / Taker Rates (Low Volume) | Spread Markup / Slippage Profile | Key Regulatory & Insurance Status | Interface & Developer Tools (CLI/SDK) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alpaca Crypto** | Tiered Commission | 0.15% / 0.25% (Tier 1) down to 0.00% / 0.10% | Moderate (Relies on external order routing) | Alpaca Crypto LLC; **Not** SIPC or FINRA protected. | Official Alpaca CLI (`alpacas` / `alpaca-cli`) & Unified SDK. |
| **Kraken (Kraken Pro)** | Tiered Commission | 0.25% / 0.40% (low volume) down to 0.00% / 0.05% | Low (Very tight bid-ask spread; deep books) | High-compliance dedicated exchange. | Official Kraken CLI (`kraken-cli`), WebSocket/REST APIs & CCXT. |
| **Binance.US** | Flat Commission | 0% / 0.02% | Moderate (Decent books for USD majors; thin altcoins) | Dedicated US exchange. | Basic REST API & CCXT support (No official CLI). |

---

## 3. Impact on Research & Backtesting (vectorbt Modeling)

When simulating strategies using `simulate.py` or writing custom weight/signal scripts, researchers must calibrate their fee/slippage parameters to reflect the target execution venue. Using standard stock parameters will result in massive out-of-sample underperformance.

### 1. Alpaca Crypto (Exchange-like Broker Model)
*   **Fees:** Alpaca has updated its model to a volume-tiered maker/taker commission. For baseline WFO backtests, model a taker fee of at least **0.0025 (0.25%)** or maker fee of **0.0015 (0.15%)** per side.
*   **Liquidity:** While commission fees are lower now, Alpaca still routes to external liquidity venues. Keep a slippage penalty of at least **0.001 to 0.002 (0.1% to 0.2%)** to account for routing latencies and potential spread wider than native exchange order books.

### 2. Kraken Pro (Exchange Model)
*   **Fees:** Model using `fees=0.0025` (maker) or `fees=0.0040` (taker) for low-volume baselines. If simulating institutional-scale volumes, use the sliding tier defaults (e.g., `fees=0.001` or lower).
*   **Liquidity:** Deepest books. Spreads are highly competitive. This is the gold standard for backtest realism.
*   **API Integration:** Highly robust, reliable WebSocket/REST endpoints, making it the preferred venue for high-frequency or multi-asset execution systems.

### 3. Binance.US (Low-Fee Exchange Model)
*   **Fees:** Model using `fees=0.0002` (taker) or `fees=0.0` (maker) as a best-case scenario.
*   **Liquidity:** Altcoin books can be thin. If trading beyond BTC/ETH, add a slippage penalty of at least **0.002 (0.2%)** to account for order book depth.

## 4. Key Recommendations

1.  **Strict Simulation Penalties:** For all future crypto strategy sweeps, do not default to standard equity fees (e.g., 0.1% or similar). Apply a minimum of **0.4% taker fee** (modeled as `fees=0.004`) to reflect Kraken Pro baseline, or **0.25% fee + 0.15% slippage** (modeled as `fees=0.0025`, `slippage=0.0015`) to reflect Alpaca execution.
2.  **Order Routing Selection:** 
    *   For testing and development of API brokerages or multi-asset integrations (where stocks and crypto sit side-by-side): Use **Alpaca**.
    *   For alpha generation, quantitative strategy research, and high-frequency trading: Deploy **Kraken Pro** for deep books and standard maker/taker trading.
