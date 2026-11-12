# Expected-investment-growth portfolio screen

**The proposed overall hurdle was not met.** An independent Claude Opus reviewer recomputed the figures from the frozen inputs. The owner approved the separate historical screen in conversation. Its [fixed specification](2026-11-11-expected-growth-proposal.md) and [raw output](_expected_growth_screen_20261111.json) preserve the rules and figures.

The candidate is the provider's value-weighted large-stock, high-expected-investment-growth portfolio. The comparator is the provider's broad-market total return, **not SPY**. The returns below deduct an assumed 1% annual cost from the candidate only; they are pretax.

| Window | Candidate annualized growth | Market annualized growth | Proposed hurdle |
| --- | ---: | ---: | --- |
| 1967–2025 | 13.02% | 10.94% | Newey–West t = 2.728, one-sided p = 0.00318; misses t ≥ 3 and p ≤ 0.00135 |
| 1994–2025 | 13.41% | 10.78% | +2.63 percentage points a year; clears the proposed +1 point hurdle |

Both hurdles were required. The 2% cost sensitivity is descriptive: annualized growth was 11.90% versus 10.94% over the full window and 12.30% versus 10.78% since 1994. The candidate's worst month-end drawdown at the 1% assumption was −44.43%, versus −50.30% for the market; these occurred in different crashes (2000–2002 and 2007–2009 respectively). This was not a registered protection test.

The data and idea were already published. The one-test threshold does not provide Holm correction across previous studies. The candidate excludes financial firms and firms with negative book equity, while the market includes them. The provider's 2026 reconstruction uses revised source data; original accounting vintages, real turnover, executable fills and tax are unverified. The result depends on the assumed, unmeasured cost: the independent review found t = 4.17 with no cost, a post-hoc observation that does not change the registered 1% rule. A published portfolio is not a directly tradable stock-selection system, and this result neither identifies current winners nor establishes a smaller effect is absent. The historical momentum and other NO-GOs and the production freeze remain unchanged.

The driver and input hashes are in the raw output. Its recorded git commit is `73d4a4f45d34d66fac18102cc95692cdcf36efbf`; the wider worktree was dirty, but the reviewer confirmed that the hashed inputs and code match the committed versions. Local commit and file timestamps are not independently witnessed. The driver output's `status` field is inherited proposal wording; the owner subsequently approved this calculation, as recorded in the specification. No paper or live trading is authorized.
