# Amendment: continue the historical daily-bar study with the live parity gap disclosed

**Status: approved by Flynn in this thread on 2026-10-19, before study runs.**

**Original brief:** [2026-10-19 production-replica audit](2026-10-19-production-replica-audit.md).
The original file remains unchanged.

## 1. What the investigation established

The original parity run matched 10 of 32 orders to the specified historical signal
rules, or 31.25%. The registered denominator includes 19 owner-directed manual
liquidations. A further order, AEE's September 29 sale, reproduces using the old
120-day signal lookback: that day's log reports 83 input bars, before the longer
lookback image was deployed that afternoon.

ALL's October 13 sale and ARES's October 15 purchase cannot be reconstructed from
the exact original yfinance inputs. The current tape does not reproduce those
orders under the prior-completed-bar rule. Alternative prices within the sessions'
recorded trading ranges can produce the signals, but those substitutions are
diagnostics, not proof of what live saw. ALL's prior-day broker mark also produces
its exit; a broker mark is not an archived yfinance input.

Evidence and raw replay inputs are in
[`_replica_parity_20261019.json`](../_replica_parity_20261019.json), keys
`orders`, `explained_pct`, `notes`, and `investigation`. Backup checks are recorded
under `investigation.backup_check`. The available backup configuration and local
dumps do not supply these decision-time inputs.

The loader can include a same-session daily bar or retain a previous partial bar.
It does not guarantee the original brief's assumption that signals always use the
prior completed close. An end-of-day replay therefore cannot establish exact
replication of those historical live decisions.

## 2. Approved change

Permit the remaining historical studies to answer this narrower question:

> Does the fully specified ensemble portfolio model, using prior completed daily
> bars and the original brief's portfolio rules, beat SPY on the registered windows?

Continue with the original design and pass thresholds. Do not adjust parameters,
windows, costs, signal lag, cash assumptions, or thresholds to match the two orders.
The change is permission to evaluate that daily-bar model while historical live
parity remains unverified.

The report must distinguish two conclusions:

- **Historical daily-bar model:** GO or NO-GO using the original brief's statistical,
  pre-tax and conditional after-tax thresholds on both windows.
- **Exact deployed-trader replication:** UNVERIFIED. The original audit's live-parity
  gate has not passed. Never report 100% parity, exclude the two orders to improve
  the percentage, or count hypothetical price substitutions as actual inputs.

A model GO supports further forward validation. It does not establish that the
deployed trader has an edge. A model NO-GO rejects the specified daily-bar model;
it does not supply a measured performance verdict for the unreconstructed live path.
Any freeze-lift action remains the owner's decision, with this limitation explicit.

## 3. Gates and controls retained

- Keep the price-artifact/tainted-trade stop, the 70% holdout coverage floor, and
  the FOMC session-semantics stop exactly as registered.
- Run after-tax replay only when both windows pass both pre-tax thresholds.
- Retain all task tests, review gates, checkpointed raw artifacts, pinned windows,
  sensitivity arms, and per-task commits from the implementation plan.
- The independent corrected-harness momentum/FOMC studies and A23 tax bounds may
  proceed after this amendment is approved; their original tests and bars apply.
- Keep the live freeze, restricted imports, SELECT-only database access, permitted
  loader cache writes, and no-deploy/no-push rules unchanged.

Collecting immutable signal inputs and enforcing completed-bar semantics in the
live path are follow-ups for a separately authorized change after the freeze.
They are not part of this amendment or permission to modify production now.

## 4. Approval

Flynn approved this amendment explicitly in this thread on 2026-10-19. The
historical daily-bar study and independent reruns may proceed under these terms.
