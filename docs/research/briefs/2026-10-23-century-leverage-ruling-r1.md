# Ruling r1: century leverage A23 reproduction gate (senior quant, 2026-10-23)

This ruling interprets the [approved brief](2026-10-23-century-leverage.md) §8. It changes no
number or rule.

**Finding.** The unmodified A23 driver no longer reproduces
`_static_leverage_aftertax_results.json` exactly. Its free upstream inputs (yfinance SPY, SSO and
BIL prices and distributions, and FRED's DTB3 T-bill series) have been revised since 2026-09-30,
and its calibration window was open-ended.
- With the window pinned to the committed 2026-09-30 end (`--calib-end`, added in this commit),
  the pre-2006 haircut moves from −0.149%/yr to −0.147%/yr.
- The 75/25 synthetic-holdout terminal value moves by about $17 on $336,748.
- The fixed 2006–2026 real-SSO results are unaffected.

**Ruling.** The gate's purpose is to show that new code does not change A23's *computation*. It
is satisfied this way:
1. Freeze the A23 inputs once, as fetched today, to `data/external/a23_inputs/` with a SHA-256
   manifest.
2. Run the A23 logic as of commit `53b739c` (this commit, before any study refactor) on those frozen inputs. The only change allowed is a loader that reads the frozen files in place of the live fetch, with the same columns and values.
3. Run it again with the study's code.
4. The two outputs must be numerically identical.

The upstream-vintage difference above is recorded as a known limit of free data. It does not
justify a code change, and the committed A23 JSON is left as it is.
