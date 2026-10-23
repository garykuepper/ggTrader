# Ruling r2: century leverage end date (senior quant, 2026-10-23)

This ruling amends the [approved brief](2026-10-23-century-leverage.md) §2 and §5 on one point,
forced by the data. It was made before any study result was computed. No other number or rule
changes.

**Finding.** In the frozen Shiller workbook (`data/external/shiller/`, see its MANIFEST), the S&P
dividend field ends at **2026-06**; July and August 2026 are blank. The brief's registered
end, 2026-08-31, therefore lacks the taxable-dividend input that the approved resolution
requires for the 1× leg.

**Ruling.** The study ends at **2026-06-30**, the last month in which every input (French daily
`Mkt-RF`/`RF`, Shiller dividends, A23 frozen inputs) is present.
- No filling, carry-forward or imputation of any kind.
- The full window becomes 1926-07-01 → 2026-06-30, and the last subperiod becomes
  2000 → 2026-06-30.
- The calibration window (2006-06-21 → 2026-04-30) is unchanged.
- The two dropped months are reported as excluded, with the reason.

Truncating removes two months out of about 1,200. Imputing them would introduce assumed data into
a tax calculation. Truncating is the smaller deviation.
