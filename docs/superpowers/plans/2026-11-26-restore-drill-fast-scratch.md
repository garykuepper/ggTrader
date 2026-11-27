# Close-out and Next Steps Plan (revised 2026-11-27)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (small, doc-only tasks). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Record that the 2026-11-26 backup is proven, bring the docs and memory up to date, and put the owner's open decisions in front of them so the research can start.

**Revision note:** The first version of this plan moved the restore test's temporary copy to the fast drive. The test ran overnight with the old script and passed (report `ok: true`, 78 of 78 table counts and 6 of 6 checksums matched, temporary copy cleaned up). The script change is no longer needed to finish this work. It is kept as an optional item (Task 5) for the owner to decide.

## Global Constraints

- Plain words; define any technical term in the same sentence. Do not cite the retired old evidence rules.
- No GO/NO-GO labels. Paper trader stays stopped. No orders.
- Nothing pushed or deployed without the owner's say. The owner has already approved merge and push for the cleanup work, not for new changes.
- Keep database backup copies and old reports; never rescore them.
- No heavy disk jobs in daytime without the owner's say.

## Tasks

### Task 1: Record the backup result
**Files:** `docs/database-inventory-2026-11-26.md`, `docs/kaggle_import_runbook.md` (dump section), `docs/changelog.md`
- [ ] State: dump `ggtrader_20261126.dump` (12,019,307,097 bytes) restored in a disposable copy; 78 of 78 counts and 6 of 6 checksums matched; report `drill_report_20261126.json`.
- [ ] Note the old `attempts` folder was deleted, and that the 11-25 dump also passed.
- [ ] Verify: `grep -n "20261126" docs/database-inventory-2026-11-26.md docs/changelog.md`.

### Task 2: Add the SPY 1993–1999 table everywhere it belongs
**Files:** `docs/database-inventory-2026-11-26.md`, `research/programme/next-research-agenda-2026-11-26.md` (lines ~9, 11, 156), `research/RESEARCH_SNAPSHOT.md` (~361, ~442), `docs/next_steps.md`
- [ ] Describe table `market.spy_daily_yahoo_1993_1999` (1993-01-29 to 1999-12-31, free Yahoo, fetched 2026-11-26, frozen, manifest `spy_1993_1999_20261126.json`).
- [ ] Replace "SPY did not exist / no comparator before 2000" wording with: SPY covers 1993 onward; only 1969–1992 lacks a fund to compare against.
- [ ] Verify by grep that no stale "before SPY existed" text remains.

### Task 3: Update next_steps and memory
**Files:** `docs/next_steps.md`, memory `project_cleanup_and_next_research_2026-11-26.md`, `project_database_consolidation_2026-11-26.md`, `MEMORY.md`
- [ ] Mark answered owner questions (merge and push done at b295e821, attempts deleted, fresh dump taken, Yahoo SPY added).
- [ ] Fix the stale memory lines ("on branch, not merged", "not pushed", "no fresh dump").
- [ ] Commit docs only; push only if the owner says so.

### Task 4: Tidy-up (small, no data touched)
- [ ] Delete the git-ignored workspace `.superpowers/sdd/2026-11-26-cleanup-and-next-research` (the plan is merged). Ask first if the owner wants the ledger kept.
- [ ] Tell the owner the crontab holds a plaintext token; offer to move it to a file the cron line reads.

### Task 5: Optional — fast-drive restore tests
Add `--scratch-root PATH` to `scripts/db_restore_drill.py` (default stays `thesix`) so future tests can build their copy on the main drive and leave the RAID alone. Only if the owner wants it; the earlier six-test design stands (nearest-existing-parent free-space check, refusal of `/`, resolved-path safety check).

## What is left that needs the owner

1. **Which tests to run first.** Recommended: the three early-decades arms (early decades, slowing the trading down, a rule that accounts for stocks about to die) as one run on 1969-02-03 to 1998-07-31.
2. **Comparison for 1969–1992.** May the CRSP market return series (the all-stock average in the data we hold) stand in for SPY, or start the early test in 1993?
3. **Downloads vs the RAID:** pause Deluge or move its active folder to the main drive? (Only matters if more heavy jobs run on `thesix`.)
4. **Optional Task 5** above.
5. Three older open items: two unsent drafts, the AVB question, the factor-scope wording.

## Self-Review

- Each remaining item above maps to a task or an owner decision; nothing builds or runs without approval.
- No task touches the live database, the paper trader or backup files.
