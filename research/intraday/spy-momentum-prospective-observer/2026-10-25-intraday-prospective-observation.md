# Prospective SPY intraday observation protocol

**Status: owner authorized offline build with "ok build it" after the plain-language explanation. Activation remains unapproved.**
Offline implementation is authorized; activation has not begun. The [execution plan](../../superpowers/plans/2026-10-25-intraday-prospective-study.md)
requires approval of this concrete brief before building and a separate activation decision
after tested code, source preflight and exact timer definitions are reviewable.

## 1. Question and limits

Can our isolated observer reliably record the fixed SPY intraday rule prospectively,
and what hypothetical outcomes occur over the registered observation period?
This is operational and descriptive research, not a confirmatory profitability test.
Historical r8 remains UNRESOLVED (data). There is no historical pass to carry forward.
The existing stock-candidate forward standard is not amended or satisfied here.

No broker orders, borrowing, account changes, paid data or capital allocation.
No changes to production paper code, cron, Docker, registry or database. The
15:45 ET production SPY sweep is irrelevant to positions in this file-only ledger;
the observer must never share a paper account's cash or positions.

## 2. Registered dates and fixed milestones

Proposed inception rule: choose the **third full NYSE session strictly after
activation's New York calendar date**, leaving at least two complete full sessions
for preparation. Before activation completes, hash the exact first 252 scheduled
full-session dates, checkpoint date (slot 20), last date (slot 252), preceding
official-session anchor and intervening early closes. Compare offline NYSE calendar
5.5.0 with Alpaca's calendar; disagreement prevents activation. First observation
cannot occur until this dated manifest exists. Dates may not be selected using
outcomes. No fixed November 2 deadline or performance-dependent start is retained.

Freeze the endpoint at manifest slot 252. If an unexpected closure cancels a listed
day, append `exchange_closed`, retain its slot and do not extend the endpoint.
If a listed full day becomes an early close, retain the slot, record `early_close`,
place no hypothetical trade and archive its closing-minute anchor. Both count
toward the 20/252 observation slots, but neither is a full-session return observation.
Calendar amendments preserve original and revised official schedules. Previously
scheduled early closes are outside the 252 slots and are still archived as anchors.

Count every slot whether flat, faulty or halted. Operational rates use actual
eligible full sessions as denominators and separately disclose closures/early closes;
return statistics never invent market observations for cancelled days. No restart
or replacement sessions erase faults. Missing activation of the registered first
session requires a new future registration, not retrospective decisions.

## 3. Source, timing and decision

Use only free delayed SIP bars from Alpaca, `SPY`, `feed=sip`, `timeframe=1Min`, `adjustment=raw`, `asof=-`,
ascending minute timestamps, inherited read-only credentials. Verify actual
entitlement and delay before activation; no switch to IEX. Never assume SIP
access is real-time or that today's revised archive equals an earlier observation.
A minute bar is timestamped at the start of its interval. Each request's inclusive
`end` must be no later than current time minus the verified source delay.

All wall-clock schedules are America/New_York; artifacts use UTC and identify
the exchange session. Tentative standalone job stages, subject to capability preflight:

1. **15:20 ET input capture.** Use the immediately preceding official session's
   closing minute (15:59 normal, 12:59 after a 13:00 early close) from its canonical
   first valid archive, identical to the record used to value its hypothetical
   exit. Never silently re-fetch a different anchor. If that archive is missing,
   its recovery must finish strictly before this 15:20 stage; otherwise today's
   signal input is unavailable. A conditional **15:15 ET recovery job** may fetch
   an absent/incomplete/invalid prior-session archive within its valuation grace,
   retaining all original faults; it may not replace an existing valid canonical
   value. A differing later value is a revision, never a replacement.
   Bootstrap alone may fetch the preceding anchor during 15:20 input capture,
   completed before 15:25, because no earlier prospective position needs valuing.
   Fetch today's 09:59 minute close with inclusive `end=09:59:00 ET`. Do not request
   or expose entry/exit values to the decision process. Freeze input references.
2. **Input cutoff strictly before 15:25:00 ET.** Required inputs and their receipt times must be
   persisted by then. Decision uses these frozen snapshots only; later corrections
   cannot change the original signal or whether its evidence was timely.
3. **Decision window 15:25:00 inclusive to 15:26:00 exclusive ET.** Write exactly one immutable
   side and input references. The decision artifact must be completed before
   the window ends, not merely a process started. A late/missing stage is a fault.
4. **Hypothetical entry at 15:30 minute open, exit at 15:59 minute close.** These
   are bar proxies valued later, never claimed executable fills. Once the side is
   written, later prices cannot choose or cancel the hypothetical position.
5. **16:20 ET archive.** Attempt that day's regular-session snapshot by 16:25, with inclusive end at the
   official closing minute (15:59 normal, 12:59 after a 13:00 early close).
   Its first valid values are canonical for valuation and the following anchor.
   If that deadline is missed or an initial response is absent/incomplete/invalid,
   a capture finished **strictly before the next official session's 15:20 ET** may
   supply the first valid valuation, marked `late_capture`. Retain every earlier
   payload/fault. This grace applies to valuation only, never to signal inputs or
   decisions. It fails the operational deadline and cannot cure that checkpoint.
   After the grace deadline, missing valuations stay unresolved; later observations
   are revisions for a separate appendix, never primary values. A later valid
   snapshot cannot replace an already valid canonical price. Conflicting valid
   records are logged, not resolved by selecting a preferred outcome.

On early-close sessions, skip input/decision/trade stages, archive the official
session after the close using the same 16:20 job, and preserve its anchor.
No cash-market observation jobs run on a known non-session.

Proposed request bounds: at most 12 GET attempts/session across stages, retries
and pagination; >=0.65 seconds between starts; 30-second total request deadline;
at most three attempts/page (the 30-second deadline is per request, not per stage);
no redirects/proxies or secret logging. The budget is per New York calendar date
for all recovery and current-session calls, so midnight or a restart cannot reset
a stage allowance. No stage may retry past its cutoff. Bounds and measured payload/storage estimates must
be verified in readiness evidence before activation, not increased silently.

## 4. Fixed rule, accounting and diagnostics

`r_first = P_0959 / P_prior_close - 1`. Side is 1 only when `r_first > 0`,
otherwise 0. No filters, other symbols, threshold search or outcome-dependent tuning.
Use current ledger equity, hypothetically fractional, no leverage, starting at
$100,000 at the bootstrap close. Idle cash earns zero.

For a timely valid decision, `s_t = side * (P_1559 / P_1530 - 1) - side * 2 * c`,
where `c=0.0001` base and `c=0.0002` stress. Costs are charged on hypothetical
traded days; genuine flat decisions are zero. Base and stress have separate wealth
paths. Costs do not certify spread, auction availability or market impact.

Preserve r8's simplified tax policy for comparability: at year-end debit 24% of
positive calendar-year net realized P&L **after costs**, separately for base and
stress, with no loss benefit or carryforward;
accrue terminal partial-year tax at final valuation. The first tax year is the
inception-year stub; debit on that year's last official-session close if it is
inside the observation window. Use the same rule each year, and the fixed final
observation close for terminal accrual. Never hardcode the former proposed dates. It is a policy proxy, not
validated taxable lot accounting or a personal tax return. Tax-debited equity
determines subsequent sizing. Annualize using actual elapsed days from the
anchor close divided by 365.25. No historical wealth is appended to this series.

For SPY's non-gating comparison, use the complete issuer distribution schedule,
not just patches for missing vendor rows. Freeze source URL, sheet, columns,
fund identity/CUSIP and original source bytes/hash; append subsequent dated
publications. Respect ex-date versus record/pay dates. Require unique official
session ex-dates and positive declared dividend amounts. Preserve capital-gain
legs separately; blank is unknown until field semantics are documented, not zero.
USD must be documented by the source convention; unknown currency is unresolved.
Compare overlapping vendor amounts rounded to the issuer's published decimal
places using decimal ROUND_HALF_UP; do not pick the more favorable value.
Any difference remaining after this rounding stays unresolved. Reinvest each known distribution at
that session's closing-minute proxy, with no interim/terminal tax, explicitly
different from the sleeve tax policy. Include early-close sessions for this path.
Unknown source completeness, declaration/classification or prices keep that
diagnostic provisional. A complete terminal comparison requires an issuer publication
obtained after the final session and confirmed to cover the full observation
window, with field semantics verified. Initial and revised versions stay separate.

Report counts, gross/base/stress outcomes, simplified tax drag, break-even costs,
drawdown, worst month, losing streak, signal counts, SPY comparison and the
ex-distribution-flat sensitivity when their inputs are valid: zero the hypothetical
return on every issuer cash-distribution ex-date, without changing the primary path.
Display NW-5 t and CAGR as plain numbers only: no threshold comparisons, pass/fail
booleans or GO/NO-GO fields. NW uses intercept-only OLS, Bartlett HAC, maxlags=5,
use_correction=True; zero or nonfinite variance is unresolved. Complete fixed-rule
statistics require complete valid rule inputs; an observer no-position fault is not
silently treated as a fixed-rule zero.
No replacing the old historical subperiods with favorable prospective windows.

## 5. Evidence and fault policy

Every record binds protocol/code/runtime/calendar hashes, session/stage, source
parameters, requested/received/completed UTC times, original lossless payload
and byte hash, prior-record hash and decision references. Store original evidence
separately from corrections. Reject duplicate stages; after a crash reconcile
the existing claim before any action, never create a second side or trade. A
claimed stage is completed by atomic rename. Restart may finish a still-incomplete
decision only inside its window and if no completed side exists; after the window
it is a fault. Record host clock-sync status with each stage; absent/unsynchronized
clock status is an evidence fault. Local timestamp evidence is never independent.

Proposed timing evidence is **local only**: hashes and host timestamps support
integrity checks, not independent proof of when a decision existed. No external
receipt service or publication is authorized. Reports must say locally timestamped
and may not claim independently witnessed decisions. A stronger evidence route
requires a separately approved mechanism before the observations it would witness.

Retain every scheduled session and its evidence status:

- **One-session faults:** missing/stale/late signal inputs, missing/late decision,
  clock-evidence fault or late archive. Record no new position if no timely valid
  decision exists; retain the fault. These alone do not halt future sessions when
  account equity remains known. Never insert a timely decision retrospectively.
- **Required data:** validate all endpoints on flat sessions too. Missing/invalid
  flat-session prices are unresolved rule/diagnostic data, not a reason to invent
  returns; they do not by themselves make the observer's known cash equity unknown.
- **Halting faults:** unknown entry/exit valuation on a hypothetical traded day
  beyond its allowed grace period, duplicate/conflicting decisions, or ledger/hash
  integrity failure. Freeze subsequent new positions and preserve the last known
  accounting state; do not invent an exit, wealth or tax liability. Pending valuation
  must resolve before equity-dependent sizing can proceed.
- **During a halt:** continue data capture and timely signal decisions where inputs
  permit, marked `halted_no_position` with separate signal-side and observer-side
  fields. If inputs are missing record that fact; do not fabricate a signal side.
- **Resume:** only by an owner-approved append-only amendment completed before a
  future session's 15:20 stage. Never restart inception, erase faults or backfill trades.
- **Operational versus valuation evidence:** a permitted late valuation may complete
  P&L accounting, but it remains late for the operational checkpoint. Required values
  outside the grace period stay unresolved in the primary path.
- **Aggregate claims:** preserve elapsed time, all slots and every fault. Show observer
  policy outcomes separately from fixed-rule outcomes; do not report a complete
  fixed-rule return if its inputs are missing. Accounting uncertainty prevents
  complete observer-wealth claims too.

If observer no-order behavior becomes relevant, distinguish realized hypothetical
observer policy from the fixed trading rule; do not claim they are the same strategy.
This study's unresolved evidence can never become a profitability pass.

## 6. Operational and descriptive evaluation

At registered slot 20, operational acceptance requires every scheduled full-session decision
to have completed on time, with archives by 16:25, complete required valuations,
original evidence and no unresolved fault; approved exchange-closed/early-close
slots require correct no-trade classification and their applicable archives instead.
A genuine late/missing decision fails this checkpoint
even if later data arrive. It is an operational failure, not strategy NO-GO.
Halt new hypothetical trades pending review; capture may continue. Local-only
timestamp status is disclosed, not silently upgraded to independently witnessed.

Monthly reports retain all sessions and faults. There is no interim performance
pass, optimization, performance-triggered extension or conditional start-date reset.
At registered slot 252 publish an immutable descriptive report/JSON. If required
data remain unavailable, report UNRESOLVED instead of a complete performance result.
Here and in the plan, "publish" means save an immutable **local repository file**,
not a push, message or external publication. Save original and amended evidence
separately; do not hold the initial report
until an attractive result or invent a guaranteed vendor publication lag.

Planning arithmetic only: assuming IID whole-calendar daily SD 25 bp, N=252 gives
mean standard error 1.574852 bp/session. Expected t=3 requires mean 4.724556 bp;
approximately 80% power at that cutoff requires 6.049984 bp under a normal
approximation. Actual SD, autocorrelation, trade frequency and joint economic
performance are unknown. These are not measured returns or an actual HAC power
calculation. A one-year record cannot be promised to establish a modest edge.

This phase ends at its registered endpoint; there is no statistical pass label.
Any later confirmatory test needs its own predeclared sample and monitoring rule.
No observer outcome authorizes broker paper, live money or a standard exemption.
If closures reduce actual full-session observations below 252, report the actual N;
the planning arithmetic is not recalculated into an approval threshold.

## 7. Delivery and authorization sequence

1. Parent drafts this brief and records proposed defaults; Opus reviews it.
2. Resolve objections, present this exact brief and obtain protocol approval.
3. Sonnet implements offline in the three isolated files in the approved plan;
   parent verifies tests and integration, with Flash only on exact mechanical checks.
4. Parent provides a bounded source preflight, synthetic rehearsals, exact timer,
   paths/permissions/request/storage bounds, code/protocol hashes and readiness review.
5. Owner authorizes activation of the concrete observer. Until then no scheduler,
   production change, live observation or hypothetical decision is enabled.

The owner's instruction to execute starts this sequence. It does not retroactively
approve this newly specified brief or the future timer configuration.


## Build authorization record

After the explanation that this records hypothetical buys/sells, sends no orders,
checks operation after about a month and observes about a year, the owner said
"ok build it". This authorizes building and synthetic testing of the isolated
recorder/practice journal. It does not activate collection or a timer. Stage 7.2
is satisfied for offline implementation; activation remains reserved at Stage 7.5.
