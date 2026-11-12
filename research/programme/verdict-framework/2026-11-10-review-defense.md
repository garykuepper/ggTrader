# Defensive studies: proposed retrospective annotations

**Date:** 2026-11-10  
**Scope:** Five pending inventory records selected from `inventory.json`: the cross-asset static sleeve, its registered 2007–2011 holdout, its slow-trend arm, commodity trend, and the dynamic FX hedge overlay. The reviewed annotations preserve original verdicts and do not change production or study artifacts.

## Method and boundaries

I applied the draft verdict framework dated 2026-11-10, especially its evidence/action distinction and defensive path. Original study conclusions and their scopes remain unchanged. The proposed `protocol_result` records the original registered gate when it can be checked; the evidence label assesses what the existing material supports under the new framework. Missing power alone does not establish invalidity or a negative result. `UNRESOLVED` is used only where audit findings prevent a valid independent inference or protocol determination.

The selected studies do not share one defensive claim. The sleeve asks whether a fixed SPY-funded 20% mix improves risk-adjusted results and drawdown versus SPY; its distinct holdout asks whether the frozen rule passes a weak, crisis-heavy historical check. Trend timing asks whether the 252-day arm improves over the static mix. The commodity report tests an isolated commodity momentum/filter strategy against SPY. The FX report's primary claim is dynamic versus static hedging on the same ETF pairs; SPY is reference-only. None of these is silently reinterpreted as the other's question.

## Proposed annotations

### Static 80/20 SPY + TLT/GLD/PDBC sleeve

The original brief pre-registered five policy checks. The report records all passed, including Sharpe and MaxDD comparisons across pinned and long windows, 2022 performance, 3 bp sensitivity, and leave-one-leg-out checks. It also reports lower CAGR than SPY and labels a post-hoc Sharpe-gap bootstrap. Lens 1 calls the pinned static headline clean. Lens 2/Lens 9 cannot compute uncertainty or power from the saved artifact. The evidence label is therefore **INCONCLUSIVE** for the new defensive framework: there is no registered material downside inference paired with a wealth-noninferiority margin, and the bootstrap is post-hoc. This does not erase the original `GO (shadow only)` or claim the policy checks failed. Action is **HOLD / PARK THIS VERSION** because no owner-approved next-stage scope is evidenced in the reviewed sources.

### Static sleeve holdout, 2007-06 through 2011-02

The holdout had its own frozen restricted rule and passed the stated Sharpe, MaxDD, and 3 bp checks. Its own brief calls this weak support: the short span is dominated by 2008, a regime likely to flatter the sleeve. Lens 1 calls the data clean; Lens 7 spot-checked the recomputation. Lens 6 is unverifiable and Lens 9 cannot compute a t-statistic or MDE from the artifact. Record **MET / INCONCLUSIVE / HOLD**. The protocol passed; this narrow result does not establish a material defensive benefit or noninferior wealth. Preserve `PASSED (weak)` exactly.

### Slow-trend arm on the cross-asset sleeve

The pinned trend comparison improved Sharpe by only 0.007 and MaxDD by 1.6 points against registered bars of 0.05 Sharpe or 3 points of drawdown improvement; long-window trend Sharpe was below static, and the WFO halted in folds 2–8. The registered trend-preference rule is **NOT MET**. The original report concludes trend adds nothing over static; this is a closure of that timing specification, not a judgment against the static sleeve. Evidence remains **INCONCLUSIVE** for a broad inferential claim: the audit marks WFO validity unknown (it predates an engine fix), DSR units unknown, code unverifiable, and power unavailable. Preserve the original narrow closure.

### Commodity trend (A3)

The original report states OOS Sharpe 0.13 vs SPY 0.74, CAGR 1.0%, MaxDD −37.0% vs −33.7%, 27/50 gates, and an active regime halt for a 14-ETF 12–1 momentum plus volatility-filter rule. Preserve the original `NO-GO` and isolated scope. The retrospective audit found no raw artifact or brief, code unverifiable, mixed contamination (anchor leak may favor the candidate; duplicate-SPY direction unclear; DSR units unknown), and no MDE. As these faults prevent checking the underlying inference and protocol, annotate **UNRESOLVED / UNRESOLVED / CLOSE**. This is not evidence that commodity trend has no edge; carry and basis-reversal legs were outside the tested scope.

### Dynamic FX hedge overlay

The original report's primary claim is that a dynamic carry/value/trend overlay beats static FX hedge policies on two ETF pairs, not that hedging beats SPY. It reports dynamic results below each static baseline and preserves a specific retail-proxy `NO-GO`. The audit found no raw artifact or brief, unverifiable code, mixed contamination, unknown DSR units, and no MDE. The annotation is **UNRESOLVED / UNRESOLVED / CLOSE** because the audited record cannot independently validate or re-determine the result. Preserve the original verdict for this specification; do not generalize it to dynamic FX hedging as a field.

## Coverage limitations

This is a five-record batch, not a complete defensive-study reconciliation. It omits other pending allocation overlays and leveraged/timing studies in the inventory. The source material does not provide candidate-specific power for these records; for the two reports whose raw artifacts were not located, the summary metrics are prose claims rather than independently recomputed results. I did not run return calculations or tests. These annotations must not be treated as approved policy, new research authorization, or changes to any historical verdict.

An independent Sol reviewer checked the numeric claims against the source and identified missing comparator context, previously inspected windows and two source closures; the lead verified the source passages and adopted these corrections. Review is same-family; no returns were recalculated.
