# Pre-registration: cross-asset sleeve, unseen-data holdout (2007-06 → 2011-02)

**Frozen:** 2026-09-28, before any run on this window. Parent study:
`docs/research/2026-09-28-cross-asset-sleeve.md` (static arm GO, shadow only).

## Why this window, and why it is weak evidence

The 2026-09-28 sanity check (`artifacts-2026-09-28-web-batch/sanity_checks.py`)
already scored the static 80/20 with DBC on 2011-03-01 → 2026-04-30 (0.89 vs
SPY 0.84), so everything from 2011-03 on has been seen. The only unseen data
runs from BIL's first month (inception 2007-05-30) to 2011-02-28, about 3.75
years.

That span is dominated by 2008, when Treasuries and gold rallied while equities
fell. Those are the conditions most likely to flatter this sleeve.

- **A pass is weak evidence.** It shows the rule didn't break, not that the edge is real.
- **A fail counts heavily against it.** If the sleeve can't beat SPY in its most
  favourable regime, the 2021–26 result is very likely a gold/commodity-run
  artifact.

## Frozen rule (unchanged from the parent brief)

- Static arm only: 80% SPY + 6.67% each TLT / GLD / **DBC** (PDBC starts
  2014-11; DBC tracked it identically where both exist). Keep all three legs;
  no leg changes in response to the parent study's leave-one-out.
- Month-end close decision, applied next bar, 1 bp per side. Driver path:
  `select() → to_targets() → simulate_weights()`, as in the parent study.
- Window: **2007-06-01 → 2011-02-28**.

## Pass bar (unchanged criteria, restricted to what this window can test)

1. Book Sharpe > SPY Sharpe.
2. Book MaxDD no worse than SPY's.
3. 1 and 2 still hold at 3 bp per side.
4. Reported, not gated: leave-one-leg-out and the calendar-2008 return.

The parent's 2022 criterion does not apply here.

**Reading the result.** Pass → the shadow continues unchanged, and the pass is
recorded as weak support. Fail → the static GO is withdrawn and the shadow stops.
