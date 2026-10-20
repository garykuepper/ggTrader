# Changelog

## 2026-10-20

- **Research: record fixes and momentum re-test pre-registration (Phase 0).**
  - **Correction section appended:** `docs/research/2026-10-19-production-replica-audit-nogo.md`
    updated with a dated correction section documenting the `dual_momentum` duplicate
    finding (identical to `xs_momentum` on the 2026-10-19 replay; negative filter never
    removed a pick; not an independent test), Deflated Sharpe Ratio (DSR) gate power
    limits under 1-year folds and 12-combo grid (in-sample Sharpe ~2.5 required to pass),
    and clustered Federal Open Market Committee (FOMC) event study statistics.
  - **Clustered FOMC event study:** recorded day-clustered test across 113 FOMC meeting
    dates (`docs/research/_fomc_clustered_event_study_20261020.json`), finding gross
    t = 2.258 and net t = 0.981 (fails statistical significance after transaction costs;
    the earlier unclustered reference of 3.619 gross / 1.572 net counted each meeting
    day three times across TLT/IEF/EDV).
  - **Research snapshot & prompt templates updated:** `docs/research/RESEARCH_SNAPSHOT.md`,
    `docs/research/prompts/web-strategy-research-prompt.md`, and
    `docs/research/prompts/local-implementation-prompt-TEMPLATE.md` synchronized.
    Explicitly labeled historical KEEP 0.80 vs 0.78 and historical "gates 17/17" as void
    and superseded by the 2026-10-19 production-replica audit NO-GO.
  - **Next steps updated:** `docs/next_steps.md` updated with DSR power limits note and
    made pre-registered momentum re-test the active research step linking to
    `docs/research/briefs/2026-10-20-momentum-retest.md`.

## 2026-10-19

- **Research: production-rule daily-bar audit — NO-GO.** New event-loop replica
  and Newey-West excess-return helpers reproduce the specified sizing, risk,
  prior-bar signals and SPY sweep rules. Primary CAGR 14.19% versus SPY 14.83%,
  excess −0.64 pt/year and t −0.69; all sensitivities fail. The holdout is
  INCONCLUSIVE after a tainted-trade stop, while exact historical live parity
  remains UNVERIFIED. Owner-approved amendments preserve those failures and
  permit negative closure on the primary result. Corrected momentum and FOMC
  reruns remain NO-GO; A23 distribution-tax sensitivity is separate owner
  information. See `research/2026-10-19-production-replica-audit-nogo.md`, its raw
  JSON/curves and the regenerated snapshot. No live code, configuration or
  deployment changed.

- **Lab event-study timing correction:** close-to-close event samples now exclude
  the entry session's return, which was earned before buying, and include the
  exit session's return. The registered FOMC semantics test exposed this error;
  a non-flat price regression also checks sample counts and means. Prior event
  statistics require recomputation. The owner authorized the correction in
  `research/briefs/2026-10-19-production-replica-fomc-timing-amendment.md`;
  no live code or configuration changed.

- **Lab DSR correction:** the gate compared annualized Sharpe against per-day
  thresholds and used the wrong kurtosis term, so it passed almost everything.
  An annual Sharpe of 1.0 over 252 days with 48 trials scored 1.0 instead of
  about 0.10. Sharpe is now converted to per-period units for the calculation;
  the reported expected maximum remains annualized. Every past "x/17 gates"
  count reflects the density gate only. Corrected walk-forward folds will
  fall back to the anchor much more often. No live code or config changed.

## 2026-09-30

- **Research: static fractional leverage (A23) — GO for owner decision;
  A24 sleeve trend rider — NO-GO.** New driver
  `scripts/static_leverage_aftertax.py` (lot-level tax engine, band
  rebalancing, synthetic SSO 1993–2006 calibrated on the real overlap, A24
  rider); raw results `docs/research/_static_leverage_aftertax_results.json`;
  report `docs/research/2026-09-30-static-fractional-leverage.md`. 70/30
  SPY/SSO beats SPY by +1.6 pt/yr after tax on 2006–2026 at the cost of −66%
  vs −55% drawdown; Sharpe below SPY. No live change. **Funding-cost check**
  (same day, `--rf-offset`): margin +0.80 pt at today's ~4% bills, below the
  +1.0 bar — GO is conditional; owner decision re-opened.
- **Research intake: leveraged S&P ETF web batch** (`WEB_RESEARCH_CANDIDATES.md`
  2026-09-30 batch): A23/A24/B13/C8 added, 31 sources verified, Gemini
  backtest figures unsourced and excluded. `ohlcv` SSO/UPRO re-fetched in
  full; FRED `DTB3` cached.

## 2026-09-29 (built 2026-09-28 night, deployed 2026-09-29 ~17:15 PT)

- **Deployed** image `31fcff28` (built from `291e6dd`); rollback image
  `10e1e073`. Next run (09-30) predicted: AVB excluded (one alert), no stock
  buys (AIZ/ALL/BALL/GL fill 4 slots vs cap 3), no sells signalled as of
  09-28, no sweep buy (cash 5% < new 8% trigger).
- **Tape repair:** APH and MNST full history re-fetched; every bar before
  07-20 had been 2x too high (mixed split adjustment). MRNA's +177% on 08-19
  is real (Alpaca adjusted bars, ~30x volume), left as-is. Backup in
  `results/tape_repair_20260929/`.
- **`.env`:** `SWEEP_BUY_TRIGGER_PCT` removed, so the 8% dead-band default
