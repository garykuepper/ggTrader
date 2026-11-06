# Intraday Stage 0A: sampled SIP coverage observed; start input unresolved

Stage 0A execution is complete. Its registered bar conclusion is **unresolved**,
and the auction branch is **auction_branch_unresolved**. This is a data-preflight
finding, not a strategy GO/NO-GO. No strategy returns were computed.

The [fixed contract](briefs/2026-10-25-intraday-stage0a-contract.md) authorized
metadata only. The [raw artifact](_intraday_probe.json) records the calendar,
frozen sample/request manifest, whitelisted metadata, response hashes and
per-request outcomes. It was generated at `2026-10-25T18:34:14.285190+00:00`
using code revision `c20be22713b3d277052b03ad8ff0ff451c17139e`.
Artifact SHA256: `10703eb0bc71c435577941740014b0cbeeaf2276696205a5b2ae08b6ab972bd2`.

## Observations

- All 20 official-calendar selections succeeded. The deterministic diagnostic
  months span January 2016–September 2026; they do not approve a strategy window.
- All 91 HTTP attempts returned 200: 20 calendar and 71 market-data logical
  requests. No retries were needed. Outcomes were 72 nonempty successes,
  18 empty successes and one schema error. HTTP success alone is insufficient
  evidence of complete or usable data.
- All 20 sampled **SIP** sessions contained every one of their 390 expected
  regular minutes, with no duplicate or misaligned regular timestamps. Each
  also returned a 16:00 bar outside that grid. It was retained and flagged,
  without assigning it an execution-price meaning.
- The January 4, 2016 SIP opening sentinel contained all five expected minutes.
  The required **December 31, 2015 previous-close input was empty** on both
  feeds: neither expected 15:59 nor 16:00 minute was available. Consequently
  only 21 of the 22 required primary bar checks passed. The window or previous
  close convention was not changed after this observation.
- The 2014 and 2015 opening sentinels were empty on both feeds. These bounded
  observations do not establish an archive inception date.
- **IEX:** nine of the 20 sampled sessions were empty; 14 had at least one
  missing regular minute, totaling 3,607 missing minutes. Six had complete
  regular grids. No interpolation or halt explanation was applied. This
  sensitivity feed is not a substitute for SIP in this sample.
- All 20 full-session auction requests returned opening and closing metadata,
  including multiple venues. The 2014/2015 closing-only sentinel requests were
  empty. The January 4, 2016 closing-only request returned HTTP 200 but failed
  the documented metadata schema; it is unresolved, not proof of no auction.
  Its body hash is retained, while raw payload and prices were discarded.
  The full-session request for that same date succeeded; the discrepancy
  remains unexplained.

## Interpretation and next work

The sampled SIP regular-minute coverage is encouraging for feasibility, but
it does not satisfy the fixed Stage 0A gate because the start's previous-close
input is unavailable. Do not claim a gap-free 2016–2026 archive, choose a new
start from this sample, or run the proposed strategy yet.

The next bounded metadata investigation should specify a supplemental request
manifest before fetching: diagnose the closing-only auction schema discrepancy
and establish which previous-close input/date convention is supportable.
Preserve this artifact; any new vendor response needs its own hash and result.
A later Stage 0B would check prices, adjustments and endpoint reconciliation
only under its separate authorization. Official auction venue/execution mapping,
full-window coverage and the strategy protocol remain open before backtesting.

## Verification and implementation rulings

Codex independently reviewed the implementation for specification and quality;
both passed without findings. The 38 focused mocked tests passed. Repository
Ruff check and format-check passed. The parent independently checked all 91
request records, the manifest/document hashes, planned request set, successful
pagination, and deep price/token/credential exclusion in the actual artifact.
The full capped suite passed: **1,458 passed, 30 deselected**, in 313.49s.
One `websockets.legacy` dependency deprecation warning arose in the existing
headline-sentiment test; it was not changed for this study. The final independent
branch audit verified all 48 bar grids, the 91-request plan, document/manifest/artifact hashes and the deep metadata whitelist.
Minimum observed request-start spacing was 0.650023 seconds, meeting the
0.65-second floor. Both independent reviews found no Critical, Important or
Minor issues.

Three routine interpretations were fixed before the live run: the previous-close
bar grid expects 15:59 and 16:00, with the inclusive 16:01 endpoint flagged
separately; only the three explicitly named coverage sentinels get auction
requests; and valid empty symbol maps represent empty observations. A fourth
rule discards every price/unknown field even from schema-error evidence. If
those interpretations prove wrong, the cost is a revised bounded probe or
an additional request, not a retrospective strategy change.

The standalone CLI's hard timeout uses Linux main-thread signals. No production
paper-trading path, database, dependency, Docker configuration or cron changed.
Stage 0B, full-window data and strategy evaluation were not performed.
