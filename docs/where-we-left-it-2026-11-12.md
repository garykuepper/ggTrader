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
- Say which old branches to delete (list below).

## Waiting on review

- A Sol review of the engine and paper commits, before any merge.
- Minor Opus findings to fix before merging: a phantom daily-loss halt can follow an unapplied split, a slow-filling exit aborts the run and leaves no snapshot, the paper commit message overstates "durable order intents", one mutation script targets code that no longer exists, and no test pins the engine's default stop settings.

## Things to know

- **Merge conflicts later:** `docs/next_steps.md` differs between this branch and `main`, so merging will conflict there. The main checkout also holds an untracked `research/programme/benchmark-objectives/` folder that this branch tracks. Move it aside before merging.
- **Large file:** the branch includes a 2.4 MB IRS PDF source document.
- **Intraday supervision:** the Hermes job has been paused since 2026-11-06 09:18 Pacific. The reason is unknown and there is no alert. Manual checks are allowed until it is restored. The observer's four timers still record.
- **Data caveats:** FERG has zero volume before March 2021. SKYD and VMRK carry other tickers' price history on a different adjustment basis.
- **Weekly check:** it runs from the main checkout, so keep that checkout on `main`.

## Old branches (not removed)

- **No worktree, fully merged (safe to delete):** `deps/drop-ccxt`, `deps/update-all`, `refactor/study-isolation`, `refactor/study-moves-1`, `refresh/sp500-membership`, `vbt12/engine`.
- **Kept:** the `backup/*` branches and `research/momentum-retest`, which is unmerged.
- **Worktrees kept:** four old worktrees hold untracked work, a virtual environment or intraday data. The vbt12 worktree holds the only copy of the frozen price tape. One agent worktree has uncommitted edits.

## First three things to do on return

1. Read [next steps](next_steps.md) and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md).
2. Start the Sol review of the engine and paper commits, then decide on the halt policy.
3. Decide on the two unsent drafts, and look for Saturday's weekly heartbeat message.
