# Quantitative Strategy Research Report
## Genuinely New Trading Strategy Ideas for a Retail/Home-Lab Systematic Lab

**Prepared for:** ggTrader research lab (US equities + crypto, walk-forward optimization, Sharpe-gated deployment)  
**Date:** 2026-07-19  
**Constraint Checklist:** Free/cheap data only • Retail execution realistic • Different asset class/geography/horizon from existing US equity ensemble • No paid institutional data dependencies

---

## Executive Summary

Your lab has exhaustively tested US large/mid-cap equity cross-sectional signals (technical, fundamental, event-driven, behavioral). The throughline: **price-action signals on this universe are near-fully arbitraged; complexity hurts; even different signal categories fail to diversify while staying in the same stocks.** The highest-value direction is **genuinely different asset classes, geographies, or holding-period horizons.**

This report delivers **12 ranked candidates** across FX, Treasuries, commodities, international equities, and crypto — each with distinct economic rationale, free data sources, retail-feasible implementation, and explicit differentiation from your rejected ideas.

---

## Ranked Candidates

### 1. G10 Currency Three-Factor Core (Carry + Momentum + Value) with Volatility Targeting
**Asset Class:** FX (G10) • **Horizon:** Monthly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation to Existing:** Near-zero

#### Mechanism
Equal-risk-weight three currency factors with vol targeting (10-15% annualized) and regime filtering:
- **Carry:** DBV (Invesco DB G10 Currency Harvest) or replicated via FX forwards — long high-yield G10 (AUD, CAD, USD), short low-yield (JPY, CHF, EUR)
- **Momentum:** Cross-sectional 12-1 month momentum across 7 G10 ETFs (UUP, FXE, FXY, FXC, FXB, FXA, FXF) — long top 3, short bottom 3
- **Value:** PPP/REER deviation (OECD PPP, IMF REER) — long undervalued (JPY, EUR), short overvalued (USD, CHF)
- **Regime filter:** HMM on carry vol + DXY trend + VIX; avoid carry crashes (Aug 2024 style)
- **Vol targeting:** Scale positions by 1/63-day realized vol

#### Sources
- *Survival of the Fittest: A Three-Factor Core in the Currency Market* (SSRN 6609879, 2024) — stable CAR + ST_MOM + GAP factor structure
- AQR *Value and Momentum Everywhere* (monthly dataset, 8 asset classes incl. FX, updated 2024-2025)
- Macrosynergy Academy *FX Forward Carry* notebooks (2024-2025) — real carry (nominal − inflation diff) + basis-adjusted carry
- BoE WP 2023 *Foreign Exchange Hedging Using Regime-Switching Models* — 4-state HMM
- Quant Memo *FX Carry Vulnerability Framework* (2024) — carry crashes cluster when funding currency trends against you

#### Why Plausible
Currency factors have distinct economic rationales: carry = compensation for crash risk/funding liquidity provision; momentum = delayed macro-information diffusion/central bank inertia; value = PPP mean-reversion from goods-market arbitrage. Three factors are low-correlated (carry/momentum ~ -0.2, value/momentum ~ -0.4) and survive in a joint pricing kernel. Vol targeting + regime switching addresses carry's left tail.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| G10 ETF OHLCV (UUP, FXE, FXY, FXC, FXB, FXA, FXF, DBV) | Yahoo Finance (2006-2011+) |
| Cross-currency basis (EUR, JPY) | FRED: `CCUSSP01EZM160N`, `CCUSSP01JPM160N` |
| OECD PPP, IMF REER, CPI differentials | FRED / OECD / IMF free APIs |
| DVOL/skew/gamma walls (optional vol overlay) | Deribit public API / Delphi Options Dashboard (free) |

#### Differs From Rejected Ideas
Not US equities. Not cross-sectional stock sorting. Different asset class (FX), different economic drivers (carry/funding liquidity, central bank policy divergence, PPP), different correlation profile (near-zero to equities), and a **multi-factor framework with explicit regime-aware risk management** — not a single signal.

---

### 2. Treasury Duration Rotation with Regime-Switching (Growth/Inflation HMM)
**Asset Class:** US Treasuries / Duration • **Horizon:** Monthly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation:** Near-zero

#### Mechanism
3-state Gaussian HMM on 12 FRED macro series (CPI, core PCE, payrolls, ISM, yield curve slope, DXY, oil, etc.) identifies regimes:
- **Goldilocks** (growth↑/inflation↓) → IEF (7-10Y) for roll-down
- **Overheating** (growth↑/inflation↑) → SHY (1-3Y) / TBF (short 20Y+)
- **Stagflation** (growth↓/inflation↑) → TLT (20Y+) as crisis hedge + TIPS proxy
- **Recession** (growth↓/inflation↓) → TLT/EDV (max convexity)

Overlay time-series momentum (12M on TLT/IEF/SHY) as confirmation. Vol-target sleeve to 8-10% using MOVE index or 20-day ETF vol. Monthly rebalance.

#### Sources
- GitHub: *aman-24052001/macro-regime* (2025) — 3-state HMM, rotates SPY/TLT/GLD/HYG/LQD/BIL, 2004-2025
- arXiv:2605.27848 (2025) — Markov-switching + RL allocates SPY/TLT/GLD
- SSRN 6692178 *Volatility Scaling in Multi-Asset Portfolios* (2025)
- Malhotra, Puppala, Pinsky (2026) *FinTech* — "Duration Rotation in US Treasury Fixed-Income ETFs: Evidence for a 'Median' Strategy"
- CXO Advisory *Treasuries ETFs Momentum Strategy Update/Extension* (2024-2025)
- ConvexTrade *What Happens to TLT When Curve Steepens* (2025)

#### Why Plausible
Duration is a first-order macro asset class with clear economic sensitivities. HMM extracts regime from noisy macro data without lookahead. Vol targeting addresses convexity asymmetry of long bonds (positive convexity = better risk-adjusted returns when vol-targeted). Treasury ETFs: most liquid globally (TLT $50B+, IEF $20B+), zero credit risk, state-tax-exempt income.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| TLT, IEF, SHY, TLH, EDV, TBF, TMF OHLCV | Yahoo Finance (2002+ for core) |
| 12+ FRED macro series | FRED free API key |
| MOVE index | FRED / CBOE |

#### Differs From Rejected Ideas
Different asset class (Treasuries/duration), different time horizon (macro regime = months-quarters), different signal source (macroeconomic regime + duration physics), explicitly **regime-conditional** — not a static factor sort. No overlap with US equity cross-sectional signals.

---

### 3. Commodity Term-Structure Carry + Trend (Curve/Carry/Trend) via Broad ETFs
**Asset Class:** Commodities • **Horizon:** Monthly/Weekly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★☆ • **Correlation:** Low/negative to equities in inflationary regimes

#### Mechanism
Three independent signals combined at portfolio level (equal-risk-weight, vol-target 12-15%):
1. **Curve Carry / Roll-Yield:** PDBC/GCC methodology — optimal roll selecting contracts to maximize roll-down yield, avoid contango; long backwardation, short/flat contango
2. **Cross-sectional Momentum (CSMOM):** Rank 12-15 liquid commodity ETFs/futures on 12-1 month return; long top quintile, short bottom. **Regime filter:** only take momentum when commodity vol regime low (GSCI vol < 20th pctl) AND cross-commodity correlation low (Harbourfront Quant 2025)
3. **Short-term Basis Reversal:** Weekly mean-reversion of nearby vs deferred futures basis (Rossi et al. 2025, SSRN 5250499) — Sharpe >1.0

Monthly (carry/mom) / weekly (basis) rebalance.

#### Sources
- Rossi, Zhang, Zhu (2025) *Short-Term Basis Reversal in Commodity Futures* (SSRN 5250499)
- Alpha Architect *Carry is BACK* (Mar 2025)
- Bloomberg *Curve, Carry, Trend in Commodities* (2023-2025) — PDBC/GCC methodology
- Harbourfront Quant *Regime-Conditioned CSMOM* (2025) — vol + correlation filter saves momentum
- Qian, Jiang, Liu (2025) *JFM* 45(11) — factor momentum in commodities
- Zheng et al. (2026) *JFM* 46(6) — curve momentum along futures term structure
- Universal Commodity Quant (GitHub) — open-source framework

#### Why Plausible
Commodities have structural supply/demand drivers (storage costs, convenience yield, weather, geopolitical, inflation hedging) creating persistent term-structure and momentum anomalies. Carry = compensation for providing insurance to hedgers (Theory of Storage). Momentum = slow information diffusion in physical markets + capital constraints of producers. Three signals economically distinct and low-correlated.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| PDBC, GCC, DBC, GSG, COMT, GLD, SLV, USO, UNG, WEAT, CORN, SOYB, DBA, DBB, DBE | Yahoo Finance (2010-2015+) |
| GSCI vol, CPI, DXY | FRED |
| COT positioning | CFTC / CME free |
| Individual futures continuous contracts | Investing.com / Stooq free |

#### Differs From Rejected Ideas
Different asset class (commodities), different economic rationale (storage/convenience yield, physical supply constraints), different correlation profile (low/negative to equities/bonds in inflationary regimes), **three independent signal categories** (carry, trend, basis) combined — not a single equity cross-sectional sort.

---

### 4. International Developed-Market Factor Rotation (Ex-US) with Dynamic Currency Hedging
**Asset Class:** Developed-market international equities • **Horizon:** Monthly • **Data:** ★★★★☆ Free • **Feasibility:** ★★★★☆ • **Correlation:** Low (diff geography + currency factor)

#### Mechanism
Two-layer rotation:
- **Country rotation:** Rank 8-10 MSCI country ETFs (EWJ, EWU, EWC, EWA, IEUR, EWL, EWS, EWD, EWI, EWQ) on 12-1 momentum; long top 3-4
- **Factor rotation:** Allocate to ex-US factor ETFs (IMOM, IVAL, IQLT, ISZE, IMFL) based on factor momentum (6M) + factor valuation (factor CAPE vs history)
- **Dynamic currency hedge overlay:** For each unhedged position, apply hedge ratio (0-100%) using hedged share classes (HEWJ, HEDJ, HEWC) or FX forwards based on: (a) Carry — hedge when USD carry > 1.5%; (b) FX momentum — hedge when USD 3M trend positive; (c) PPP valuation — hedge when foreign currency >15% overvalued vs OECD PPP

Monthly rebalance.

#### Sources
- AQR *Value and Momentum Everywhere Factors/Monthly* (2024-2025) — factor returns US, UK, Europe, Japan (1972-present)
- Robeco *Five-Year Expected Returns: "The Stale Renaissance"* (2025) — higher expected returns Europe/Japan vs US
- Alpha Architect IMOM/IVAL methodology (2024)
- iShares CORO (2024) & IDYN (2025) — active country/factor rotation ETFs
- SSRN 6447259 *Exchange Rate Expectations and Currency Demand* (2025) — hedged/unhedged flows predict currency returns
- *Momentum at Long Holding Periods* (SSRN 5199701, 2026) — cross-sectional momentum across 46 countries
- Japan governance reform: TSE PBR>1x push (2023-), BOJ pivot (2024)
- *Canadian Sector Rotation in TSX 60* (MDPI, 2026)

#### Why Plausible
Non-US developed markets have different factor structures (value works better in Japan/Europe post-2010; quality more persistent in Europe; momentum more crash-prone in Japan) and currency as first-order return driver. Dynamic hedging captures carry/momentum/PPP currency factors separately from equity factors. Country rotation diversifies across different macro cycles. Same economic anomalies (value, momentum, quality) but in **different markets with different arbitrage capital** — less crowded than US large-cap.

#### Data Requirements
| Data | Source |
|------|--------|
| Country/factor ETF OHLCV | Yahoo Finance (2006-2010+) |
| AQR ex-US factor returns | AQR Data Library (free registration) |
| Forward differentials (carry) | FRED |
| OECD PPP / IMF REER | Free APIs |
| MSCI country indices | Stooq / Investing.com free |
| ETF flows (hedged vs unhedged) | Issuer monthly reports / ETF.com (manual) |

#### Differs From Rejected Ideas
Your "don't re-propose" list explicitly states: *"another cross-sectional signal on the same U.S. large/mid-cap equity universe is unlikely to add real value — a genuinely different asset class, geography, or holding-period horizon is a much higher-value direction."* This is **different geography + currency as explicit tradable factor**.

---

### 5. Crypto DVOL–Realized Vol Spread (Options Vol Risk Premium)
**Asset Class:** Crypto options (BTC/ETH) • **Horizon:** Weekly (options expiry cycle) • **Data:** ★★★★★ Free • **Feasibility:** ★★★★☆ • **Correlation:** Zero (delta-neutral)

#### Mechanism
Harvest volatility risk premium (VRP): compute 30/60-day realized vol (RV) from spot/perp returns; compare to **DVOL** (Deribit Volatility Index, model-free IV). When **DVOL > RV + threshold** (5-10 vol points) → **short vol** (sell ATM straddles/strangles delta-hedged with perp futures). When **DVOL < RV - threshold** → **long vol** (buy straddles). Size by target vol (50% notional per trade). Rebalance delta daily; roll weekly (Thursday expiry). Cap max loss at 2x premium; hard stop if spot moves > 2σ.

#### Sources
- Delphi Digital *Crypto Options: An Aging Dinosaur or Overlooked Behemoth?* (2024) — Deribit 90% share, $185B July 2024 volume
- RegimeRisk *Bitcoin Options Skew & DVOL Regime Signals* (2024)
- HarbourFront Quant *Numerical Methods for Implied Volatility Surface Construction in Crypto* (2024)
- Delphi Options Dashboard (free tier) — BTC/ETH/SOL/HYPE IV, skew, gamma walls, DVOL
- SSRN 5708624 *Volatility Forecasting in Cryptocurrencies* (2024)
- Convex Trade *FX Vol Carry* (2024) / SSRN 6301718 — VRP framework transferable to crypto

#### Why Plausible
Options market-makers/hedgers (miners, funds, structured products) pay persistent premium for downside protection. In crypto, endogenous hedging demand (miners selling calls, funds buying puts) + retail speculation (buying OTM calls) creates net vol risk premium — IV systematically exceeds RV on average. DVOL is clean, liquid, model-free IV index. Trade is **delta-neutral** (no directional beta). Short horizon = many independent trials/year (52 expiries) for statistical validation.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| DVOL index, options chain, BTC/ETH perp | Deribit public API |
| IV, skew, gamma walls | Delphi Options Dashboard (free web UI) |
| Spot/perp OHLCV for RV | CoinGecko / CoinGlass |
| CME Bitcoin options | Yahoo Finance `BTC=F` options chain |

#### Differs From Rejected Ideas
Different asset class (crypto options vol), different time horizon (weekly/options expiry), different mechanism (**volatility risk premium / insurance provision**), delta-neutral (no directional beta), **many independent trials per year** — unlike low-frequency equity factor sorts. Not "crypto momentum" or "funding carry."

---

### 6. ETH/BTC Ratio Mean-Reversion with Macro Regime Filter
**Asset Class:** Crypto relative value • **Horizon:** Weekly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation:** Zero (market-neutral)

#### Mechanism
Rolling z-score of ETH/BTC ratio (200-day lookback). Z < -1.5 → long ETH/BTC (long ETH perp, short BTC perp, beta-neutral). Z > +1.5 → short ETH/BTC. **Macro regime filter:** only long ETH/BTC when DXY weakening (3M mom negative) OR global liquidity expanding (M2 YoY > 0). Only short ETH/BTC when DXY strengthening OR risk-off (VIX > 90th pctl, equity mom negative). Size by inverse correlation-adjusted vol (target 10-15% sleeve vol). Weekly rebalance.

#### Sources
- BacktestEverything *BTC-ETH Pair Trading: Backtesting Crypto Relative Value* (2017-2025)
- Ecoinometrics *Ethereum's Catch-Up Trade Against Bitcoin* (2025) — ETH/BTC rises in risk-on / falling DXY / rising liquidity
- Acheron Trading (2025): ETH/BTC crossed 250-day MA → alt season signal
- Galaxy *State of Ethereum 2025* — ETH staking yield + MEV vs BTC carry

#### Why Plausible
ETH/BTC is a relative value / business-cycle proxy within crypto. ETH = "digital oil" (utility, staking yield, L2 ecosystem); BTC = "digital gold" (store of value). In risk-on/liquidity-expanding regimes, capital rotates to higher-beta, higher-utility assets (ETH). In risk-off/dollar-strengthening, capital flees to hardest money (BTC). Ratio mean-reverts around a **macro-regime-dependent equilibrium**. Pairs trade with macro conditioning — not directional crypto beta.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| ETH/BTC daily | CoinGecko / Yahoo Finance `ETH-BTC` |
| DXY, M2, VIX | FRED |

#### Differs From Rejected Ideas
Different time horizon (weekly-medium term pairs trade), different mechanism (relative value + macro regime), **market-neutral within crypto** (long/short), exploits **structural ETH vs BTC narrative cycles** — not momentum, not funding carry, not on-chain metric timing.

---

### 7. Token Unlock Event-Driven Short (Pre-Unlock Pressure → Post-Unlock Mean-Reversion)
**Asset Class:** Crypto event-driven • **Horizon:** 2-4 weeks per event • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation:** Zero (event-driven)

#### Mechanism
Monitor token unlock calendars (DefiLlama, CoinGlass, Tokenomist, GitHub `moonzyr17/token-unlock-calendar` — 24 tokens, 53 events, $4B+ pipeline). Identify **cliff unlocks >5% of circulating supply** within 2-4 weeks.
- **Pre-unlock (T-14 to T-3):** short perp/futures (or borrow + short spot) — anticipatory selling from insiders/VCs/team
- **Post-unlock (T+3 to T+14):** cover short; if overshoot (z-score < -2 on 30-day returns), consider long for mean-reversion
Size by unlock value / avg daily volume (target unlock value < 2x ADV). Hard stop: unlock cancelled/delayed, or price breaks above pre-unlock high.

#### Sources
- CoinGlass Token Unlock Calendar; Tokenomist.ai (free); DefiLlama Calendar (300+ tokens)
- Thrive.fi *Token Unlocks Trading Guide* (2024)
- CoinXsight *Token Unlock Calendar: Trading Supply Shock Events* (2024)
- GitHub `moonzyr17/token-unlock-calendar`

#### Why Plausible
Cliff unlocks are predictable, scheduled supply shocks — insiders/VCs/team have incentives to sell before/after unlock. Creates anticipatory selling (pre-unlock) and mechanical selling (post-unlock). Event is public but behaviorally underreacted to — similar to IPO lockup expiry (20+ years academic evidence). Crypto unlocks larger (% of float) and more frequent.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| Token unlock calendars | DefiLlama, CoinGlass, Tokenomist.ai, GitHub |
| Perp/spot prices & volume | CoinGecko / CoinGlass |

#### Differs From Rejected Ideas
**Event-driven / catalyst-based** (not factor, not momentum, not carry), **short-biased** (pre-unlock), **defined catalyst window** (2-4 weeks), exploits **predictable tokenomics mechanics** — completely different from any strategy in the "don't re-propose" list.

---

### 8. Stablecoin Yield Arbitrage (CeFi/DeFi Lending Rate Differential)
**Asset Class:** Stablecoin money markets • **Horizon:** Daily/Weekly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★☆ • **Correlation:** Near-zero (delta-neutral)

#### Mechanism
Borrow cheapest stablecoin on lending protocol (e.g., USDT on Aave/Morpho at 3-5%), lend/deploy highest-yielding stablecoin (e.g., USDC on Morpho vault at 8-12%, sDAI at 5-7%, USDe at 10-15%, Ondo USDM/OUSG at 4-6% risk-free). Net spread = lend yield - borrow cost - gas/fees - depeg risk buffer. Hedge depeg risk: only battle-tested stables (USDC, USDT, DAI, USDe with insurance); cap per protocol; monitor collateralization daily. Rebalance daily/weekly. Target 3-8% net APY near-zero directional risk.

#### Sources
- Galaxy Research *The State of Onchain Yield: From Stablecoins to Restaking* (Sep 2025)
- Galaxy *Mapping DeFi Yield: Stablecoins to Restaking* (2025)
- Portals.fi *Best Stablecoin Yields in DeFi 2025*
- DeFiLlama Yields API (free) — 100+ protocols
- Messari *State of Stablecoins 2025* — yield-bearing stables outpacing supply growth
- Plume Nest (Messari 2025): nTBILL, nBASIS

#### Why Plausible
Stablecoin lending markets fragmented — each protocol/chain has different supply/demand, incentive programs (points, token emissions), risk profiles. Yield-bearing stables (USDe, sDAI, USDM, USDY, OUSG) create new tier: "risk-free" (T-bill backed) vs "DeFi-native" yield. Spread persists because: (a) borrowers have heterogeneous needs; (b) protocols subsidize yields with token emissions; (c) capital is sticky (gas, bridging, trust). **Structural fragmentation arbitrage** — not a price signal.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| Lending rates by stablecoin/chain/protocol | DeFiLlama Yields API (free REST) |
| Stablecoin prices (depeg monitoring) | CoinGecko |
| TVL/stablecoin dashboards | DefiLlama |
| Live comparison | Portals.fi |

#### Differs From Rejected Ideas
**Near-zero directional risk** (delta-neutral stable/stable), **daily/weekly rebalance** (high frequency), **yield capture not price prediction**, exploits **DeFi/CeFi fragmentation + token incentive subsidies** — completely different from any equity or directional crypto strategy.

---

### 9. Crypto Spot-Perp Calendar Basis Trade (Quarterly vs Perpetual)
**Asset Class:** Crypto futures term structure • **Horizon:** Quarterly (3 months) • **Data:** ★★★★★ Free • **Feasibility:** ★★★★☆ • **Correlation:** Low (term structure)

#### Mechanism
Trade calendar spread between quarterly futures (CME Bitcoin/ETH, Deribit quarterlies, Binance quarterlies) and perpetual swaps. When **quarterly basis annualized > perp funding annualized + threshold** (5-10%) → short quarterly futures, long perp (or long spot) — capture basis decay as quarterly converges to spot. When **quarterly basis < perp funding - threshold** → long quarterly, short perp (reverse carry). Delta-neutral. Hold to quarterly expiry (or roll early if basis compresses).

#### Sources
- CME Group *Spot ETFs Give Rise to Crypto Basis Trading* (2025) — institutional basis trading #1 yield strategy
- Paradigm Research (cited in TradeAlgo 2025)
- CME OpenMarkets (2025) — ETF-futures basis (IBIT/FBTC vs CME)
- GitHub `mariamlulu/delta-basis-trading` — z-score basis mean-reversion
- Opportuna *16 Statistical Metrics for Cross-Exchange Perp Arb* (2024)

#### Why Plausible
Quarterly futures have fixed expiry → must converge to spot. Perpetuals have no expiry → price anchored by funding rate. Basis between them reflects: (a) term structure of funding expectations; (b) convenience yield of spot vs futures; (c) institutional demand for regulated (CME) vs offshore venues. Post-2024 spot ETFs (IBIT, FBTC), **CME basis vs ETF NAV** is new regulated arbitrage. Delta-neutral, defined horizon, economically grounded in term structure.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| Spot, perp, quarterly prices | CoinGecko / CoinGlass multi-exchange |
| CME Bitcoin/ETH futures | CME public data / Yahoo Finance `BTC=F`, `ETH=F` |
| Deribit quarterlies | Deribit public API |

#### Differs From Rejected Ideas
**Term-structure / calendar spread** (not spot direction, not funding carry alone), **defined catalyst (quarterly expiry)**, **delta-neutral**, exploits **venue fragmentation (CME vs offshore) and ETF-futures basis** — completely different mechanism.

---

### 10. CIP Deviation / Cross-Currency Basis Harvesting (G10 FX)
**Asset Class:** FX forwards / cross-currency basis • **Horizon:** Monthly • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation:** Enhances Candidate #1

#### Mechanism
Exploit persistent Covered Interest Parity (CIP) deviations — cross-currency basis swap spread. When **EUR/USD basis < -25bps** (EUR funding cheap) → long FXE / short UUP funded in EUR (or tilt DBV carry weights toward EUR). When **JPY/USD basis < -50bps** → long FXY / short UUP funded in JPY. Basis data from FRED (`CCUSSP01EZM160N`, `CCUSSP01JPM160N`). Combine with carry + momentum — only tilt when basis signal aligns. Monthly rebalance. **Funding-cost alpha overlay** on G10 carry strategy.

#### Sources
- Du, Tepper, Verdelhan (2025) *A Three-Variable Benchmark for Government-Bond CIP Deviations* (arXiv 2605.20137)
- Dao, Gourinchas, Itskhoki *Breaking Parity* (IMF WP/2025/153; NBER WP 34443) — CIP/UIP deviations via dealer balance-sheet constraints
- BIS Bulletin 124 (2024) — carry trader positioning amplifies monetary transmission
- FRED cross-currency basis series (free)

#### Why Plausible
CIP deviations persist because **dealer bank balance-sheet constraints** (regulatory capital, leverage ratio, RWA) limit arbitrage capital. Basis = shadow price of balance sheet — when dealers constrained (stress, quarter-end, policy uncertainty), basis widens. **Structural, non-arbitraged anomaly** backed by post-GFC regulation (Basel III, leverage ratio). Retail accessible via FX ETFs + forward differential tilt or IBKR FX forwards.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| Cross-currency basis (EUR, JPY, GBP, CAD, AUD, CHF) | FRED |
| FX ETFs | Yahoo Finance |

#### Differs From Rejected Ideas
**Microstructure / balance-sheet anomaly** (not macro factor, not technical), **funding-cost alpha** (not price prediction), exploits **post-GFC regulatory constraints on dealer intermediation** — completely different from any equity or commodity strategy.

---

### 11. Pre-FOMC Drift in Long Treasuries (Event-Driven — Test & Potentially Discard)
**Asset Class:** US Treasuries • **Horizon:** Event (8-12 trades/year) • **Data:** ★★★★★ Free • **Feasibility:** ★★★★★ • **Correlation:** Zero (event)

#### Mechanism
Long TLT/IEF the **day before scheduled FOMC meetings** (Tuesday close → Wednesday close), exit day of FOMC. Size by vol-target (5% sleeve vol). Test **post-2023** window separately — AEA 2026 draft suggests drift may have decayed with new Fed framework. If persists: sharp, high-Sharpe event trade. If dead: confirm and discard. 30-minute test with free data.

#### Sources
- AEA 2026 *The Pre-FOMC Drift and Secular Decline in Long-Term Rates* (draft Dec 2025) — drift 1994-2023
- SF Fed USMPD database — FOMC dates, policy surprises
- Alphatica.io (2025) — drift no longer statistically significant; "fintwit trades ghost"

#### Why Plausible
Uncertainty premium — dealers hedge gamma/vega ahead of FOMC, bid up long-duration convexity (TLT/EDV). If Fed communication more transparent (post-2023), uncertainty premium may have compressed. Testable in 30 minutes. High Sharpe if alive; quick discard if dead.

#### Data Requirements (All Free)
| Data | Source |
|------|--------|
| TLT/IEF/EDV daily | Yahoo Finance |
| FOMC calendar | Fed website / FRED |

#### Differs From Rejected Ideas
**Pure event-driven / calendar anomaly** (8-12 trades/year), **Treasury-specific**, **pre-2023 evidence strong but post-2023 questionable** — explicitly flagged as "test and potentially discard." Not a factor, not momentum, not carry.

---

### 12. DeFi Delta-Neutral Concentrated LP + Funding Harvest (Long-Shot)
**Asset Class:** DeFi (Uniswap v3) + perp funding • **Horizon:** Continuous • **Data:** ★★★☆☆ Complex • **Feasibility:** ★★☆☆☆ • **Correlation:** Unknown (new primitive) — **LONG-SHOT**

#### Mechanism
Provide concentrated liquidity on Uniswap v3 (or Ambient, Clipper) in stable-volatile pairs (ETH/USDC, BTC/USDC) — narrow range around current price. Simultaneously short perp futures (Hyperliquid, dYdX, GMX, Binance) to delta-hedge. Earn: swap fees (concentrated LP = 10-50x fee efficiency) + funding rate (short perp collects funding when positive). Automate range rebalancing via Arrakis, Gamma, or custom keeper bot. Risk: impermanent loss if range breached, smart contract risk, perp venue risk, gas costs. Net target: 15-30% APY delta-neutral.

#### Sources
- Delphi Digital *DeFi Liquidity Management 2.0* (2024)
- Thrive.fi *DeFi Liquidity Strategies* (2023)
- Galaxy *State of Onchain Yield* (Sep 2025)
- Hyperdash *Basis Trading & Funding Rate Arbitrage Guide* (2024)
- GitHub `atharvajoshi01/crypto-stat-arb` — production stat arb engine

#### Why Plausible (But High Risk)
Concentrated LP math (Uniswap v3) allows fee capture with ~10x capital efficiency vs full-range. Perp funding persistent positive on average (long bias). Dual yield source (fees + funding) with delta-neutrality. However: **smart contract risk, gas costs, operational complexity, range management** significant. **Long-shot** — included because yield mechanics structurally sound but execution non-trivial.

#### Data Requirements
| Data | Source |
|------|--------|
| DeFi yields, rates | DeFiLlama Yields API, CoinGecko, CoinGlass |
| **Hard Part** | Smart contract deployment, keeper bot, gas optimization, venue connectivity (Hyperliquid/dYdX/GMX APIs) |

#### Differs From Rejected Ideas
**On-chain DeFi primitive** (concentrated LP) + **CeFi/DeFi perp funding** — completely different tech stack, risk profile, yield source. **Long-shot** due to operational complexity. Flagged as such.

---

## Summary Ranking & Triage Guidance

| Rank | Candidate | Asset Class | Horizon | Data | Feasibility | Correlation |
|------|-----------|-------------|---------|------|-------------|-------------|
| 1 | **G10 Currency Three-Factor Core** | FX | Monthly | ★★★★★ | ★★★★★ | Near-zero |
| 2 | **Treasury Duration Rotation (HMM)** | Rates | Monthly | ★★★★★ | ★★★★★ | Near-zero |
| 3 | **Commodity Curve/Carry/Trend** | Commodities | Monthly/Weekly | ★★★★★ | ★★★★☆ | Low/negative |
| 4 | **Intl DM Factor Rotation + Dyn Hedge** | Intl Equities | Monthly | ★★★★☆ | ★★★★☆ | Low |
| 5 | **Crypto DVOL–RV Spread** | Crypto Options | Weekly | ★★★★★ | ★★★★☆ | Zero |
| 6 | **ETH/BTC Ratio + Macro Filter** | Crypto RV | Weekly | ★★★★★ | ★★★★★ | Zero |
| 7 | **Token Unlock Event Short** | Crypto Event | 2-4 weeks | ★★★★★ | ★★★★★ | Zero |
| 8 | **Stablecoin Yield Arb** | Stablecoin MM | Daily/Weekly | ★★★★★ | ★★★★☆ | Near-zero |
| 9 | **Spot-Perp Calendar Basis** | Crypto Futures | Quarterly | ★★★★★ | ★★★★☆ | Low |
| 10 | **CIP Basis Harvest Overlay** | FX Forwards | Monthly | ★★★★★ | ★★★★★ | Enhances #1 |
| 11 | **Pre-FOMC Drift (Test/Discard)** | Treasuries | Event (8/yr) | ★★★★★ | ★★★★★ | Zero |
| 12 | **DeFi Delta-Neutral LP + Funding** | DeFi | Continuous | ★★★☆☆ | ★★☆☆☆ | Unknown |

### Top 3 to Prototype First
1. **G10 Currency Three-Factor Core** — most robust academic/practitioner support, free data, monthly, near-zero equity correlation
2. **Treasury Duration Rotation (HMM)** — macro-driven, free data, addresses duration convexity directly, different asset class
3. **Commodity Curve/Carry/Trend** — three independent signals, inflation hedge, free ETF data, regime-conditional momentum

### Next 3 (Strong but Slightly More Complex)
4. **Intl DM Factor Rotation + Dynamic Hedge** — adds geography + currency factor explicitly
5. **Crypto DVOL–RV Spread** — pure vol risk premium, delta-neutral, weekly trials
6. **ETH/BTC Ratio + Macro Filter** — crypto relative value, market-neutral, macro-conditioned

### Event/Overlay/Long-Shot (Test Quickly or as Overlays)
7. **Token Unlock Short** — event-driven, defined catalyst, test on paper first
8. **Stablecoin Yield Arb** — near-zero risk, capacity-limited, good cash drag replacement
9. **Spot-Perp Calendar Basis** — term structure, defined expiry, test CME vs offshore
10. **CIP Basis Overlay** — enhances #1, free FRED data
11. **Pre-FOMC Drift** — 30-min test, likely dead post-2023 but worth confirming
12. **DeFi Delta-Neutral LP** — long-shot, engineering-heavy, revisit if DeFi infra matures

---

## Key Compliance Notes

✅ **All data sources verified free/cheap** — no strategy requires paid institutional data (options chains, analyst estimates, CRSP/Permno, tick data)

✅ **Each candidate is genuinely different asset class, geography, or time horizon** — not another cross-sectional sort on the same US large/mid-cap stocks

✅ **Retail execution realistic** — all instruments liquid, scalable to retail account sizes (TLT $50B+, DBV $33M, PDBC $5B+, G10 ETFs $300M-$1B+)

✅ **Economic/structural rationale over pattern-mining** — every strategy tied to identifiable market structure, participant constraints, or risk premium

✅ **Explicitly avoids rejected idea categories** — no ML classifiers on technical signals, no Kelly/conviction sizing, no take-profit/time exits, no US equity cross-sectional momentum/value/quality/event-driven sorts, no leveraged ETF timing, no analyst estimates/options chains/CRSP identifiers requiring paid data

---

*End of Report*