# Stock allocation annotations — 2026-11-10

This batch contains three reviewed `stocks-*` inventory entries. It preserves
each source verdict and its documented scope/action. **INCONCLUSIVE** describes
the evidence review where Lens 2 found code or power limitations; it does not
rewrite the underlying protocol result. No result authorizes a trade or an
intra-year allocation change.

| Inventory row | Original verdict | Evidence | Protocol / action |
| --- | --- | --- | --- |
| `lens2-stocks-a23-static-leverage` | GO for owner decision | INCONCLUSIVE: clean data and recomputation, but code unexamined and MDE unavailable. Report distinguishes historical pass from the 2026-09-30 +2.5 pp funding-cost sensitivity (+0.80 pp/year, below the +1.0 bar). | Historical owner-decision bars MET; preserve owner-adopted allocation and scope. |
| `lens2-stocks-a24-sleeve-trend-rider` | NO-GO as a return improvement | UNRESOLVED: the report's displayed 70/30 comparator is not established as matching A24's average exposure; code review and power are also unavailable. | Protocol UNRESOLVED; preserve the narrow NO-GO and close this specification. |
| `lens2-stocks-century-leverage` | keep 30% | INCONCLUSIVE: clean data and recomputation, but code unexamined and MDE unavailable. The synthetic/proxy study finds 30% trips the −90% ruin flag; no alternative satisfies all registered conditions. | Rule MET; advice is only an input to the 2027-09-30 review. |

## Registered dates, data vintages, and criteria

- **A23:** Primary real-SSO window 2006-06-21–2026-09-29; synthetic holdout
  1993-02-01–2006-06-20. The registered bar was after-tax CAGR ≥ SPY +1.0
  pp/year on the primary window and not below SPY on the synthetic holdout.
  The 70/30 book uses a fixed ±8 pp month-end band. These windows were
  already seen, and the earlier synthetic SSO path uses a spread calibrated
  on later real returns, so it is not untouched OOS. The funding sensitivity
  is a dated 2026-09-30 historical scenario, not a current market reading.
- **A24:** Primary real-SSO window 2006-06-21–2026-09-29. The frozen rider is
  60% SPY core plus a 40% sleeve in SSO when month-end SPY is ≥2% above its
  10-month SMA, returning to SPY when ≥2% below; trades occur next bar. Its
  registered comparator is static exposure matched to A24's average exposure,
  after tax. The source does not establish that the displayed static 70/30
  comparator meets that condition, so protocol determination remains
  unresolved despite the report's documented narrow NO-GO.
- **Century leverage:** Full window 1926-07-01–2026-06-30, with registered
  eras 1926–1945, 1946–1972, 1973–1981, 1982–1999, and 2000–2026-06;
  calibration window 2006-06-21–2026-04-30. The French daily archive uses
  its 202608 database vintage (labels to 2026-08-31), but ruling r2 truncates
  the result at 2026-06-30 because the frozen Shiller dividend input ends in
  June. The fixed arms are 0/15/30/45/60% SSO. For a dated 2027-09-30 review,
  an alternative must beat 30% by ≥0.5 pp/year full-window after-tax CAGR,
  not trail 30% in 1973–1981, have max drawdown no more than 10 pp deeper,
  and avoid the −90% ruin flag. A below-30% recommendation also requires
  30% itself to trip ruin or trail 0% in full-window CAGR. The 30% arm trips
  ruin; no alternative passes all criteria. The source flags that this rule
  can retain a ruin-flagged incumbent. No automatic allocation change is
  authorized.

The JSON records per-record evaluation dates, data vintage, registered
criteria, source paths for each brief/report/raw result/ruling, and Lens 2/Lens
9 evidence. For A23 and century leverage, unavailable power is recorded as an
uncertainty and does not invalidate their deterministic registered decisions.
A24's equal-average-exposure comparator is not established, so its protocol
result is UNRESOLVED. The leverage studies are allocation trade-off analyses,
not stock-selection edge tests. No private account records, database, broker,
or return systems were accessed.

Independent Sol reviewed and rechecked this batch after corrections; final FOMC WFE is attributed to the separate audit reruns artifact. Review was same-family; no returns were computed.
