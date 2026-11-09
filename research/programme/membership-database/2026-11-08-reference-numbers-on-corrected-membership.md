# Reference numbers on the corrected S&P 500 membership list

**Date:** 2026-11-08. **Status:** evidence note. It changes no reference run, no saved evidence file and no code default.

## Summary

- The two reference backtests (core17 and its dsr0 variant) were re-run on the corrected membership list, once on vintage 2 and once on vintage 3 of the database. Vintage 1 is the original frozen file that all reference runs keep using.
- core17 moves slightly: Sharpe **0.3905 to 0.3958**, CAGR **3.50% to 3.56%**, gates **1 of 17** (unchanged). dsr0 is the same to 14 decimal places (Sharpe 0.6374, CAGR 3.41%, gates 17 of 17).
- Vintage 2 and vintage 3 give **bit-identical** results. This is expected: the three index events added by vintage 3 happen after the backtests' evaluation window ends (2026-04-30).
- The live-signal replay (120 sessions, pinned to vintage 1) is unchanged at 120 of 120. On vintages 2 and 3 it matches on only **27 of 120** sessions; the 93 differing sessions all fall between 2026-06-22 and 2026-10-30, after the list's corrections and the 2026 index changes.
- **No verdict changes.** The strategy is still far below SPY after this correction. The owner chose (2026-11-08) to keep the reference runs on vintage 1 and record these numbers here instead of re-baselining.

## Numbers

| Vintage | Source | core17 Sharpe | core17 CAGR | core17 gates | dsr0 Sharpe | dsr0 CAGR | dsr0 gates | Live replay |
|---|---|---|---|---|---|---|---|---|
| 1 | frozen file, to 2026-06-02 | 0.3905 | 3.501% | 1/17 | 0.6374 | 3.406% | 17/17 | 120/120 equal |
| 2 | upstream download, to 2026-08-18 | 0.3958 | 3.559% | 1/17 | 0.6374 | 3.406% | 17/17 | 27/120 equal |
| 3 | curated gap-fill, to 2026-11-04 | 0.3958 | 3.559% | 1/17 | 0.6374 | 3.406% | 17/17 | 27/120 equal |

- core17 Sharpe on vintages 2 and 3: 0.39581258576544154 (identical bits). On vintage 1: 0.39050398505373224.
- dsr0 Sharpe on vintage 1: 0.6373813330260034; on vintages 2 and 3: 0.6373813330260019. The difference is floating-point noise (about 1.5e-15), but the equity files are not byte-identical to vintage 1's, so do not describe dsr0 as bit-identical across that step.
- Equity-curve file hashes (first 16 hex characters of sha256): vintage 1 core17 `7dcb77d71658501d`, dsr0 `d3266f2da74d04c1`; vintages 2 and 3 core17 `6fa8c6d730eb3929`, dsr0 `121239f21ffd2a12`.

## What causes the core17 change

Vintage 2 is the upstream dataset's corrected history. The largest correction moves BLDR, JBL and UBER from 2023-10-18 to 2023-12-18 (the index rebalance was effective 2023-12-18; the original file placed them two months early, which gave a small lookahead). There are 10 historical rewrites in all between vintages 1 and 2.

## Where the raw results are

`evidence/` holds the raw result files, named `<vintage>__p5_<run>.json` and `<vintage>__p7_live_golden.json`. Each records which membership list it used (`db`, load id, content checksum) in its provenance block.

To reproduce: the database must hold the vintages (`ggt db membership status`), then run `scripts/p5_wfo_validation.py --membership db:3 --out-dir <scratch folder> --force` and `scripts/p7_live_golden.py --membership db:3 --out <file>`. The frozen reference evidence under `research/programme/vectorbt-engine/p5-evidence/guarded/` will report `DIFFERS` against these runs; that is expected.

## Caveats

- These runs use the frozen price tape of 2026-11-06. Prices for BE, VYLR and SKYD, which joined the index in September and October 2026, are not in the tape or the price table; they do not affect these windows.
- The vintage 2 and vintage 3 lists come from different sources (an upstream dataset, and a hand-assembled list confirmed against S&P press releases and the SPY fund's holdings). See `docs/superpowers/specs/2026-11-07-membership-database-design.md`, Appendix C.
- Nothing here re-baselines the oracle. If a future study needs the corrected list as its reference, that is a separate, signed-off step.
