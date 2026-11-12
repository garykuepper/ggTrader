# Where we left it (2026-11-12)

Development and research are paused. Everything is committed. Nothing was merged, pushed or deployed during the pause.

## What is where

| Work | Where it lives | Status |
|---|---|---|
| Membership database, weekly check, press-release source, price backfill | `main` | Merged, CI green, running |
| Engine changes (intraday and time-stop inputs, safe hashing) | Branch `feature/verdict-register`, commit `acb4eac5` | Committed, not merged |
| Paper-trader changes and the proposed halt policy | Same branch, commit `47d8a3c9`, plus review fixes `f3d39310`, `c66b18e0`, `b41cf239` | Committed, not merged, not deployed |
| Verdict register, eight stock screens, input pilot, drafts | Same branch, commit `51b55492` | Committed, not merged |
| [Strategy category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md) | Same branch, commit `0ae8739f` | Committed |
| Refreshed documents and this note | Same branch | Committed |
| Paper-order recovery fix | `main`, commit `783c7267` | Local only, not pushed, not deployed |

## Checks done during the pause

- **Tests on the branch tip:** 2,785 passed in the full tier and 1,144 in the study tier, with no real failures. 18 tests were skipped, most because they need the price tape.
- **Reference results:** the core backtests (Sharpe 0.3905 and 0.6374) and the live replay (120 of 120 sessions) are unchanged. The equity curves are byte-identical to `main`'s.
- **Opus review of the engine and paper commits:** no blocking findings. The default engine path is unchanged and the halt policy fails closed.

## Waiting on you

- Approve or decline the proposed paper-trader halt policy.
- Decide whether to send the two drafts: a Nasdaq data-permission request and an AVB broker inquiry. Neither was sent.
- Rule on the factor-scope wording.

## Review results for the engine and paper commits

- **Opus (2026-11-12):** no blocking findings in either commit.
- **Sol, merge-gate review (2026-11-12):** the engine commit has no blocking findings. The paper commit had two, both now fixed (commits `f3d39310`, `c66b18e0`, `b41cf239`):
  - **Sweep buy hole:** an invalid exit quantity (for example negative) blocked the strategy's buys but not the cash-sweep buy. The sweep buy is now blocked too. The reserve-restoring sweep sell is unchanged on purpose.
  - **Split halt:** an unapplied broker split can look like a 5% daily loss and halt buys. A first fix added the split value back to equity, but Sol showed that a stale stored correction could then hide a real loss. That fix was removed. The conservative halt stays, the message now says a known unapplied split may be the cause, and exits remain allowed.
  - **Pending exits:** a requested exit that was still pending skipped the quantity check, so an invalid quantity there did not block buys. Both exit loops now check the quantity first.
- **Sol final re-check (2026-11-12):** no blocking findings. Sol tried 40 combinations of exit type, live or dry run, pending or not, and bad quantities. None produced a buy.
- **Mutation checks:** each new test fails when its fix is removed.
- **Still open before any merge:** your explicit approval of the halt policy.
- **Accepted trade-off:** an unapplied split can still cause a false one-day halt of buys and sweeps.
- **Minor items not fixed:**
  - A failed Telegram send can hide a rejected buy in routine output (already so before these commits).
  - A slow-filling exit aborts the run and leaves no snapshot.
  - The paper commit message overstates "durable order intents".
  - One mutation script targets code that no longer exists.
  - No test pins the engine's default stop settings, and one stop-chunking test would pass without intraday stops.
- **Same-family note:** Sol is from the same family as the original author (Codex). The fixes were written by Claude, so Sol's re-checks were cross-family. Opus has not seen the fixes.
- **Not caused by these commits:** `tests/lab/test_data.py::test_cached_loader_interval_to_timedelta` fails when run together with the paper, engine, lab and utils test folders. It fails the same way on `main`, and it passes alone and in the full tier.

## Things to know

- **Merge conflicts later:** `docs/next_steps.md` differs between this branch and `main`, so merging will conflict there. The main checkout also holds an untracked `research/programme/benchmark-objectives/` folder that this branch tracks. Move it aside before merging.
- **Large file:** the branch includes a 2.4 MB IRS PDF source document.
- **Intraday supervision:** the Hermes job has been paused since 2026-11-06 09:18 Pacific. The reason is unknown and there is no alert. Manual checks are allowed until it is restored. The observer's four timers still record.
- **Data caveats:** FERG has zero volume before March 2021. SKYD and VMRK carry other tickers' price history on a different adjustment basis.
- **Weekly check:** it runs from the main checkout, so keep that checkout on `main`.

## Old branches

- **Deleted on 2026-11-12 (fully merged, no worktree):** `deps/drop-ccxt`, `deps/update-all`, `refactor/study-isolation`, `refactor/study-moves-1`, `refresh/sp500-membership`, `vbt12/engine`.
- **Kept:** the `backup/*` branches and `research/momentum-retest`, which is unmerged.
- **Worktrees kept:** four old worktrees hold untracked work, a virtual environment or intraday data. The vbt12 worktree holds the only copy of the frozen price tape. One agent worktree has uncommitted edits.

## First three things to do on return

1. Read [next steps](next_steps.md) and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md).
2. Decide on the halt policy. The review fixes are done and Sol's final re-check is clean. Optionally have Opus look at the fixes before a merge.
3. Decide on the two unsent drafts, and look for Saturday's weekly heartbeat message.
