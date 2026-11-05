# Accruals source feasibility (metadata only)

**Date:** 2026-11-02. **Verdict: unresolved.** The published portfolio history is promising, but terms for the intended use, revision/point-in-time behavior and a compatible benchmark/cost design have not all been established. No return series was downloaded or examined for this review; this is not a strategy result.

The [French accrual portfolio documentation](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_port_form_AC.html) reports monthly returns starting July 1963, before the required 1994 start, and quintile and decile portfolios. It forms portfolios at each June end on NYSE breakpoints, using changes in operating working capital per split-adjusted share from fiscal year ends t−2 to t−1 divided by t−1 book equity per share. Eligibility includes NYSE, AMEX and NASDAQ stocks with the stated market-equity, positive book-equity and working-capital data. The [size/accruals documentation](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_AC.html) describes 25 size/accrual intersections held July through June.

The [data library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html) offers downloadable portfolio files and notes changes in underlying CRSP data. The source identifies copyright holders; specific permission for this project's use and redistribution of raw files has not been established. A stated annual formation convention is not proof that historical accounting values were available to a trader on every formation date or that subsequent restatements were excluded. The [library explains](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) that it reconstructs the full return history monthly and that historical returns can change when CRSP revises its database; [CRSP changes](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/changes_crsp.html) also affect historical returns. Accrual-specific accounting-data revision behavior and historical file vintages remain unverified.

A long-only low-accrual portfolio can be specified before a test, but its size sort and exact weighting must be registered. A compatible benchmark, common observation windows, costs and treatment of turnover must also be specified and supported; public monthly portfolio returns do not by themselves establish implementable stock-level turnover, transaction costs or taxable lots. Verify the proposed benchmark's provenance and comparability before registration.

**Stock-level reconstruction is separate and unproven.** Published portfolio returns do not provide a point-in-time constituent and fundamentals history suitable for B0/B1. The [SEC financial statement data sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html) begin in 2009 and are described as as-filed, but the SEC cautions that they do not include all filing metadata; they do not by themselves establish the required point-in-time pipeline. The [feasibility inventory](2026-11-02-free-data-stock-feasibility.md) and [family-scope review](2026-11-02-family-scope-review.md) identify the missing pipeline and the tentative distinct-mechanism classification. Resolve the terms and revision questions, and document a defensible benchmark/cost protocol before calling this screen feasible. A feasible screen or a future pass would authorize research, not capital.

## Follow-up metadata and proposed registration (no returns examined)

The [December 2024 library archive](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library_202412_archive.html)
includes download links for both accrual portfolio datasets. This single FIZ
release snapshot is not a series of contemporaneous formation-date accounting
vintages. The initial text-extraction search missed these links; direct HTML
inspection confirmed them without opening the return files. The provider's
[variable definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html)
do not establish accrual-specific original-versus-restated handling. Searches
of the provider's library, detail and variable pages for usage, redistribution,
license and permission terms found copyright notices but no authoritative
explicit grant or prohibition for the intended local analysis or raw-file
redistribution. This is a documented search, not a legal conclusion. A proposed
workflow keeps raw downloads out of the public repo and records URLs, retrieval
dates, content hashes, code and permitted derived summaries; check applicable
terms before distribution. The detail-page HTML also contains a restrictive
notice about site images and code; its application to the research data itself
is not established. Earlier project briefs specified freezing raw downloads,
including a prior raw-file commit instruction; the proposed local-only cache
changes that publication practice, not the need for provenance and hashes.

A candidate **proposal**, not a registered or tested rule: one primary
hypothesis using the value-weighted largest NYSE size quintile / lowest accrual
quintile of the French 5×5 size/accrual portfolios, compared with the French
market total return (`Mkt-RF + RF`). This is a broad-market comparison, not SPY
or an S&P 500 stock-level backtest. The top-size and bottom-accrual quintiles
are narrower than the earlier BIG 2×3 sorts, and the detail page does not
confirm value-weighted section labels. Verify the exact cell and weighting from
metadata before freezing a brief; do not select another cell after viewing
returns. Use common retrieval and source-vintage documentation.

Proposed fixed windows are July 1963–April 2026 for A1 and January 1994–April
2026 for A2 (754 and 388 months respectively). Align all months exactly;
missing, duplicate, sentinel, non-finite or non-positive post-cost monthly
wealth factors stop the run rather than silently shortening or filling it. The
previous factor screen's 1.0% annual base and 2.0% stress haircuts, applied as
`c/12` per month to the candidate, are assumptions rather than measured
turnover, execution costs or taxes. A1 would use 12-lag Bartlett Newey–West monthly excess returns `e_t=R_p,t−R_m,t−c/12`,
normal-tail `p=Φ(−t)`, and one-sided Holm family-wise α=0.00135; a single registered
hypothesis has a single Holm threshold. A2 compares `100 × [(∏(1+R_p,t−c/12))^(12/388) −
(∏(1+R_m,t))^(12/388)]` on the same 388 months, requiring ≥ +1.0
percentage point/year, not after tax. There is no error-control claim across successive studies.

**Verdict remains unresolved.** A revised-history portfolio screen is
technically plausible, but accounting availability/restatements, the intended
usage workflow and metadata-only cell mapping still require resolution or an
explicitly documented limitation before a registered brief. Stock-level B0/B1
needs a separate point-in-time fundamentals, identity and execution pipeline.
No return observations were viewed, and this proposal authorizes no test.
