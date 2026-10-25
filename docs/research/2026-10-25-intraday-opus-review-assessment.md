# Parent assessment of the Opus plan review

Claude Opus 5.5 reviewed the overall research plan at revision `8fbea0a`.
Its [full review](2026-10-25-intraday-overall-plan-opus-review.md) is advisory;
none of its proposed protocol, gate or authorization changes is approved by
receipt of the review. No new requests, prices, code or returns were needed.

## Findings verified against local evidence

- Freeze the protocol before price inspection. The seven-step outline put
  Stage 0B ahead of the final protocol freeze, leaving avoidable room for
  informed choices. Move the freeze ahead of further price work.
- The forward path is inconsistent: the intraday brief's standalone +3%
  after-tax-policy hurdle routes a pass to a SPY-relative standard with different
  timing, cash and no-order terms. The forward standard also retains conflicting
  approved/draft wording. Resolve its application explicitly; do not silently
  weaken an approved standard or change the intraday gate.
- The r7 brief already includes a missing-prior-close fallback at line 69.
  Therefore the missing 2015-12-31 input need not block the candidate forever.
  A revised approved protocol can use the next full session, 2016-01-05, with
  2016-01-04 as its input. This corrects the parent's earlier blanket statement
  that the missing input must be resolved by another probe before backtesting.
  It does not relabel the fixed Stage 0A result, approve the strategy window or
  establish price quality.
- The saved metadata confirms multiple `P` closing records in later samples.
  A venue alone is insufficient to select an official execution close. Decide
  the executable exit and a documented auction-selection rule before prices.
- Production's documented 15:45 ET schedule overlaps the proposed holding
  period, and its sweep defaults to SPY. An intraday forward book must be
  isolated from that production reconciliation.

## Qualifications to the review

- The explanation for the closing-only schema error is a hypothesis. The
  actual payload was discarded, so omitted opening arrays are not established
  as its cause. Do not change the parser based solely on that conjecture.
- Opus's numerical power example assumes volatility and approximate session
  count. It estimates a threshold-crossing mean, not a minimum detectable
  effect at a specified power. A proper planning note must state the desired
  power, covariance assumptions and published/out-of-window inputs. Those
  figures are not empirical strategy findings or new gate recommendations.
- A 15:59 bar close remains a proxy; it cannot automatically replace auction
  execution in an executable design. Freeze how a failed auction mapping
  affects the study before collection.
- Endpoint-focused validation, eliminating the IEX rerun and combining price
  validation with a bulk pull are reasonable simplifications to propose in
  r8. They do not override the current stricter brief or authorize a pull.
  Any eligibility/no-trade policy must report missingness and retain the full
  calendar rather than silently removing difficult sessions.
- Call the historical test post-original-sample and single-shot. It is not an
  untouched holdout simply because the original paper ended earlier; related
  public results already exist, and future hypotheses reuse the same window.

## Recommended immediate task

Prepare a documents-only r8 decision memo covering the window fallback,
executable exit, meaning of A1, economic/forward-standard alignment, missing-data
policy, simplified collection sequence and shadow isolation. State proposed
defaults and alternatives clearly for owner approval. Freeze and hash the
scoring specification before any new price inspection. Do not run the next
probe or implement the reviewer suggestions implicitly.
