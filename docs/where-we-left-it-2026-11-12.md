# Where we left it (2026-11-12)

Development and research are paused. Everything is committed. Nothing was merged, pushed or deployed during the pause.

## What is where

| Work | Where it lives | Status |
|---|---|---|
| Membership database, weekly check, press-release source, price backfill | `main` | Merged, CI green, running |
| Engine changes (intraday and time-stop inputs, safe hashing) | Branch `feature/verdict-register`, commit `acb4eac5` | Committed, not merged |
| Paper-trader changes and the proposed halt policy | Same branch, commit `47d8a3c9` | Committed, not merged, not deployed |
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

- **Opus (2026-11-12):** no blocking findings.
- **Sol (2026-11-12, merge-gate review):** the engine commit has no blocking findings. The paper commit has two.
  - **Blocking 1:** a requested exit with an invalid broker quantity (for example negative) blocks the strategy's buys but not the cash-sweep buy. The sweep guard at `src/ggTrader/paper/trader.py:1353` only checks loss halts.
  - **Blocking 2:** a split the broker has not applied can make booked equity look 5% lower than the previous close, which triggers a false daily-loss halt (`trader.py:1011`). Opus raised the same scenario as medium.
  - **Minor:** a failed Telegram send can hide a rejected buy in routine output. This was already the case before these commits.
  - **Weak tests:** one invalid-exit test uses cash too small to trigger the sweep buy, and one stop-chunking test would still pass without intraday stops.
- **Opus minor items still open:** a slow-filling exit aborts the run and leaves no snapshot, the paper commit message overstates "durable order intents", one mutation script targets code that no longer exists, and no test pins the engine's default stop settings.
- **What it means:** do not merge the paper commit. Fix the two blocking findings, add tests that fail without the fixes, and get a re-review. The halt policy also needs your explicit approval.
- **Same-family note:** Sol is from the same model family as the author of this code. Opus is the independent check.
- **Not caused by these commits:** the test `tests/lab/test_data.py::test_cached_loader_interval_to_timedelta` fails when run together with the paper, engine, lab and utils test folders. It fails the same way on `main`, and it passes alone and in the full tier.

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
2. Fix Sol's two blocking findings in the paper commit, get a re-review, then decide on the halt policy.
3. Decide on the two unsent drafts, and look for Saturday's weekly heartbeat message.
