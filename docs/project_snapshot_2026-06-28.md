# ggTrader Project Snapshot — 2026-06-28

> **Classification:** Internal engineering / research reference.
> **Scope:** State of the program, validated results, research post-mortem, and recommended next steps.
> **Predecessor:** [project_snapshot_2026-06-23.md](project_snapshot_2026-06-23.md) (architecture & WFO mechanics deep-dive — still current; not repeated here).

---

## 1. Executive Summary

ggTrader is a **quantitative research lab + paper-trading system** for US equities. It asks one question: *can a systematic strategy reliably beat buy-and-hold S&P 500, after costs, in honest out-of-sample testing?*

**Current answer: yes, marginally and robustly — via a multi-indicator mean-reversion ensemble sized to deploy idle cash.** That config is validated and live on paper. As of this snapshot, **the research has converged**: the two big "improve the signal" levers explored over the past week (machine-learning entry filtering and exit-rule optimization) were both tested in honest walk-forward and **rejected**. The evidence now points clearly to where the edge does and doesn't live.

**One-line state:** validated config is ready for go-live; per-trade selection levers are exhausted; remaining upside is in sizing, ensemble structure, or a new market.

---

## 2. The Validated System (what's deployed)

| Element | Setting | Why |
|---|---|---|
| **Strategy** | 5-voter ensemble: BB + RSI + EMA + MACD-divergence + Volume-BB | Independent signal failures cancel; validated by 11-config voting ablation (2026-06-24) |
| **Entry** | ≥2 of 5 voters agree (oversold) | Majority vote filters single-signal false alarms |
| **Exit** | RSI cross back above 50 (fires independently) + vote-based exit | Captures the reversion bounce; **confirmed best** vs TP/time-stop (2026-06-28) |
| **Sizing** | Flat 3% of equity per signal | Deploys ~61% otherwise-idle cash — *this* is what beat SPY (2026-06-25) |
| **Universe** | S&P 500, point-in-time membership | Avoids survivorship inflation (+1–3%/yr) |
| **ML entry gate** | **OFF** | Anti-predictive; falsified three ways (2026-06-28) |
| **Safety** | NDH + Deflated-Sharpe gates, circuit breaker, anchor fallback | Rejects overfit params; halts on regime breakdown |

**Honest OOS performance (SP500 full-grid WFO, the validated baseline):**
Sharpe **1.12** vs SPY 0.58 · CAGR **16.3%** · MaxDD **−11%** · 16/17 gates pass.

**Live:** Alpaca paper account, daily cron 12:45 PT (fills before close), Telegram alerts, honest fill-logging reconciled against the broker (2026-06-27).

---

## 3. What Worked

- **The 5-voter ensemble.** Combining negatively-correlated trend + reversion signals is the core edge. MACD-divergence and Volume-BB each *add* (Sharpe 0.68 → 0.89); the validated default drops only the harmful weekly MTF voter.
- **Exposure sizing (3%), not signal-hunting.** The single highest-impact change of the whole project. Raising per-trade size from 2%→3% deployed idle cash and is what tipped the strategy past SPY. The lesson that reframed everything: **edge came from sizing, not from a better signal.**
- **Honest walk-forward + robustness gates.** Point-in-time universes, NDH/DSR gates, and the circuit breaker repeatedly caught overfit/noise-fit configs *before* deployment — including both June-28 rejections. The methodology is the most valuable asset in the repo.
- **Vectorbt-first lab.** One `from_signals` call simulates the whole sweep; 2.07× profiling speedup; DB-backed offline research.

## 4. What Failed (post-mortem)

| Rejected idea | Why it failed |
|---|---|
| **ML / feature entry gate** (June 28) | Falsified 3 ways: P(up) classifier is *anti*-predictive (blocks the best entries); EV-regression redesign is *worse* OOS; the one robust axis (volatility) is a 2020 artifact with zero out-of-sample value. 10 daily features hold **no stable entry-selection signal**. |
| **Exit-rule changes — TP & time-stop** (June 28) | Honest WFO recommended the *unmodified* RSI exit; exit params fit noise (winner flips fold-to-fold, gates caught it); replacement arm (no indicator exit) drew down −39.7%. |
| **Momentum strategies** | Large-cap momentum is well-arbitraged; negative OOS. |
| **Trailing / hard stops** | Exit on normal fluctuation; destroy reversion P&L. |
| **Weekly MTF voter** | Too slow for daily reversion; Sharpe 0.68 → 0.49. |
| **6-voter ensemble** | Only "better" because it included harmful MTF. |
| **Conviction-weighted sizing** | No extra profit over flat sizing. |
| **Mid-cap as standalone** | Genuinely noisier plateaus (NDH gate correct); demoted to a possible diversification sleeve. |

**The meta-finding (June 2026):** every attempt to *second-guess individual trades* — which name to buy (entry gate), or when to sell beyond the indicator (exit tuning) — has failed honest OOS. The strategy's edge is **structural** (the voter ensemble) and **scale-based** (sizing), not selection-based.

---

## 5. Current Performance Context

| Metric | 5-voter ensemble (OOS) | SPY |
|---|---|---|
| Sharpe | 1.12 | 0.58 |
| CAGR | 16.3% | ~13% |
| MaxDD | −11% | −22% |

Beats SPY on both risk-adjusted return and drawdown. The margin is real but modest — consistent with the finding that easy, cheap edges are exhausted.

---

## 6. Recommended Next Research Steps

Ordered by expected payoff × independence-from-what's-already-been-tried. The first recommendation is **not** research.

### A. Proceed to go-live (the highest-value action) 🔵
The validated config is ready and the two most-tempting "improve it first" levers are now closed. Continuing to tune the same reversion book has low expected payoff. **Recommendation: monitor paper 5–10 days, fund the $1,000 live account, and go live.** Further research runs in parallel and must clear the same gates to change anything.

### B. Sizing & regime levers (on-thesis, untested the *right* way) 🧪
The edge is in sizing — so push *there*, not on selection:
- **Kelly / volatility-scaled sizing.** Size each trade by a principled risk estimate rather than flat 3%. This is the natural successor to the 3% win and is genuinely distinct from the rejected *conviction* sizing (which used signal strength, not risk/EV).
- **Portfolio-level regime exposure.** Not an entry filter (those failed) — a *book-level* throttle that scales total deployed capital up/down by market regime. Note the prior caution: reversion likes turbulence, so a naive "risk-off in volatility" filter cuts the very environment it profits from. Test as a sizing dial, not an on/off gate.

### C. Weighted voting (incremental) 🧪
Give better-track-record voters more weight instead of equal votes. Lower expected payoff (it's still tuning the same ensemble) but cheap and untested. Worth one honest WFO pass.

### D. A genuinely orthogonal direction (highest ceiling, highest cost) 🧪
The reversion-on-SP500 book is near its measurable ceiling. The largest remaining upside is **diversification into an uncorrelated return stream**, not more tuning:
- **A different asset class / venue** (the crypto book on Kraken, or futures) where the same honest-WFO machinery applies but the return driver is independent.
- **Low-correlation universe sleeves** (the MidCap↔Nasdaq 0.35 correlation finding) combined at the *portfolio* level and scaled — explicitly as a diversification play, judged on blended risk-adjusted return, not standalone gate-pass.

### What NOT to do
- Don't revisit entry-side ML or feature gating on these features — falsified three ways.
- Don't re-sweep exits on this book — tested and rejected.
- Don't add voters or relitigate trailing stops — both closed.

**Bottom line:** the research phase on the SP500 reversion signal is effectively complete. The decision in front of the project is operational (go live) plus, in parallel, a deliberate pivot toward *sizing* and *diversification* rather than further per-trade refinement.
