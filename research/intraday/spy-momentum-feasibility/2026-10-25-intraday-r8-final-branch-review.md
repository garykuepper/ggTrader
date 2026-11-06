> The sole Important finding was fixed in `6e89281`. The [final scoped review](2026-10-25-intraday-r8-final-fix-review.md) passes with no new findings. This earlier report is retained as review history.

Reviewer: Codex GPT-6-Astra, used after the requested Opus final review failed its API rate limit. Code reviewed: 7b0c088 / branch head 9042da2. Approval 2 is pending.

**Verdict: NEEDS FIXES.** One Important finding; no Critical or new Minor findings.

Mandatory benchmark failure can silently retain GO. Diagnostic failures are recorded only when computation raises. benchmark() can return infinite wealth and cagr=None; displacement accepts those values and _json_safe() converts infinity to null without changing the verdict.

The reviewer reproduced this in memory on the registered offline calendar with invented endpoints around 100, alternating gross afternoon returns of 3.6/5.6 bp and positive morning signals, plus one finite positive synthetic payout of 1e10 each quarter. Validation passed for 2,679 scored sessions; A1 and A2 passed; serialized result was GO with null benchmark wealth/CAGR/displacement and no diagnostic error. These are deliberate synthetic corruption inputs, not claims about SPY.

Required fix: record non-finite mandatory diagnostic calculations as named failures before serialization; require valid benchmark wealth/CAGR. An otherwise passing result must become UNRESOLVED_DIAGNOSTICS. Preserve legitimately undefined attribution and zero-variance diagnostic t-statistics. No numeric cap or new performance threshold. Refresh the code-bound preflight after fixing.

The reviewer confirmed the remaining frozen protocol, endpoint/half-day/DST handling, NW-5 correction, annual tax ledger, base/stress separation, corporate-action corrections, runtime/vendor-review binding, pre-claim revalidation, collection-directory claim and bounded GET collection. Production, loaders, DB, cron and Stage 0A were unchanged.

It independently compared the generated preflight to offline build_manifest(): exact manifest, runtime, protocol/code/vendor hashes and limits matched; 132 logical requests (1 calendar, 1 daily, 1 corporate-action, 129 minute chunks).

No repeated pytest/Ruff, network, credentials, market data or file mutation by the reviewer. The parent supplied the 1,734-passing-test evidence. Benchmark failure was the sole new blocker.
