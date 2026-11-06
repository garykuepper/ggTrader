# Investigation ledger (copy)

Copy of the working ledger kept during the root-cause investigation (2026-11-04 to 2026-11-05), committed
2026-11-05 so that the citations in the synthesis and the checkpoint stay verifiable after the scratch workspace
was removed. It records task completions, review rounds, deferred minor findings, the owner's decisions
and the lead's rulings (R0 to R27). It is a working log, not a deliverable: wording is terse and some
entries are superseded by later ones. The synthesis (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`)
is the committed record of conclusions.

---

# SDD ledger — plan: docs/superpowers/plans/2026-11-04-root-cause-investigation.md

Spec: docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md (approved 2026-11-04, reachable).
Branch: investigation/root-cause-2026-11-04 (cut from main 1e419c27). Models: Claude only (Sonnet orchestrates/implements code, Opus judgment + lens work, Haiku extraction).

## Pre-flight scan (tasks sharing a file or interface; then per-task self-consistency)

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 -> T6,T7,T8,T9,T10,T11,T13,T14 | check_report / CLI used as the gate | Headings list consistent. Banned phrases would trip legit quotes of owner instructions. -> Ruling R3 |
| T1 -> T14 | `--paths-only` | Defined in T1 CLI and tested. OK |
| T2 -> T13 | run_capped.sh | OK. T2 test allocation via `bytearray(n)` is calloc/lazy and may never hit the cap. -> Ruling R2 |
| T3 -> T12,T14 | guard snapshot baseline | `docker ps` Status text ("Up 3 hours") and observer-worktree `git status` change legitimately day to day, so diffs would false-alarm. -> Ruling R4 |
| T4 -> T6..T12 | inventory JSON rows/ids | OK; every consumer keys on `id` and `lens_owners` |
| T5 -> T9 | sensitivity spec | OK; ordering T5 before T9 stated |
| T6 -> T7,T9 | per-id spec_matches_code / tape_state | OK |
| T7 -> T9 | recompute json | OK |
| T8,T11 DB reads | SELECT only | OK (read-only constraint holds) |
| T9 -> T10 (Lens 4 ceiling) | Lens 2 classifications | Plan lists T10 Lens 9 first, then Lens 4; Lens 4 needs T9 done. -> Ruling R5 (run T10 after T9) |
| T12 -> T13 | owner-approved re-run list | Spec section 6 mandates the owner gate; skill's "no stalls" yields to the spec. -> Ruling R6 |
| All tasks | SDD no-subagents contract for implementers vs plan steps that say "dispatch Haiku/Opus agents" inside a task | -> Ruling R1 |

Per-task self-consistency: T1 tests match CLI (OK). T2 OK after R2. T3 OK after R4. T4-T14 steps agree with their Files/Interfaces blocks (OK). T6 `_lens6_replays.json` created and referenced consistently.

## Rulings
Ruling R0: Work on branch investigation/root-cause-2026-11-04 in the main checkout, not a separate worktree — .venv, data/ and results/ are git-ignored and live only in the main checkout, and the checker verifies cited results/ paths. Main branch untouched; nothing pushed. — Costs if wrong: switch branches back; no live effect (production runs from the Docker image).
Ruling R1: Implementers never dispatch subagents (SDD contract). Where a plan task says Haiku/Opus extract-then-judge, I (controller) dispatch the Haiku extraction passes first, writing files under the SDD workspace, then dispatch one Opus implementer per task that reads those files and does the judgment. — Costs if wrong: extra controller dispatches; no quality loss.
Ruling R2: T2 allocation test must touch pages (e.g. `b"\x01" * (512*1024*1024)`), not `bytearray(n)`. — Costs if wrong: a test that can't fail.
Ruling R3: T1 banned-phrase scan skips markdown blockquote lines (starting with `>`) so reports can quote owner instructions; add a test. — Costs if wrong: a banned phrase hidden in a quote.
Ruling R4: T3 guard records: observer worktree HEAD hash; `systemctl --user cat` hash and `is-enabled` for the four timers; container name+image+State (no uptime); main checkout `git status --short data/`. A changed observer HEAD is flagged for human inspection (observer may legitimately commit), not auto-failed. — Costs if wrong: a missed observer write; mitigated by the HEAD check.
Ruling R5: Run T10 Lens 9 step, then Lens 4 only after T9 completes; Lens 5 can run any time after T4. — Costs if wrong: none.
Ruling R6: Stop at the end of T12 for owner approval of the re-run list (spec section 6 and section 10). If no unknowns can change a verdict, T12 states Phase 2 is skipped and I continue to T14 without stopping. — Costs if wrong: a pause or an unapproved run; no re-run is started without approval.
Ruling R7: Implementers run sequentially (no parallel implementers) to avoid git index and BASE/HEAD races; Haiku extraction dispatches may run in parallel because they write only to the workspace. — Costs if wrong: slower wall clock.

Task 1: complete (commits 1e419c27..74606550, review clean)
Task 1: minor (deferred): URL-skip test doesn't exercise "://" (dead code at check_lens_report.py:80)
Task 1: minor (deferred): missing report file gives a raw FileNotFoundError traceback (exit 1) instead of a problem line
Task 1: minor (deferred): duplicate "## " headings merged; headings inside code fences counted; backticked paths with spaces skipped
Task 2: review 1 — Spec ✅ Quality approved; 1 Important (unrequested PYTHONPATH=src override at run_capped.sh:74); commits 74606550..80b80ac6
Task 2: minor (deferred): no .venv existence check in run_capped.sh (silently falls back to system python)
Task 2: minor (deferred): no tests for arg quoting, exit-code passthrough other than 0/1, or absence of the 137 message on other failures
Task 2: fix round 1/5 (1 addressed, 0 open — PYTHONPATH override removed; commits 80b80ac6..5c73fe27)
Task 2: complete (commits 74606550..5c73fe27, review clean)
Task 3: review 1 — Spec ✅ Quality approved (0 Critical/Important); commits 5c73fe27..8eeda009
Task 3: minor (deferred): guard_snapshot.sh:93 git status can refresh .git/index; prefix GIT_OPTIONAL_LOCKS=0; use refs/heads/main at :136
Task 3: complete (commits 5c73fe27..8eeda009, review clean)
Note: guard baseline records main=1e419c27; any later merge of this branch into main legitimately changes that line only.
Task 4: Haiku extraction wave dispatched (20 passes) -> /home/flynn/ggTrader/.superpowers/sdd/2026-11-04-root-cause-investigation/extract/out-*.json (ruling R1)
Task 4: extraction wave complete; merged_raw.json written (R1 step 1 done)
Task 4: implementer DONE_WITH_CONCERNS commit 823071c0 (140 rows); review pending
Ruling R8: Task 4 fix round 1 also includes minors 7-11 and 13 (same JSON rows, one re-render) — cost if wrong: slightly larger re-review.
Task 4: review 1 — Spec ✅ Quality NOT approved; 6 Important (replica conflict missing; midcap400 live flag; crypto kraken snapshot merge; void headline superseded_by; 5 uncovered files; momentum counter-evidence), minors 7-14; commits 8eeda009..823071c0
Task 4: minor (deferred): 12 ten false live flags w/o stated basis (asked implementer to null where unsupported); 14 six rows rest only on private memory notes
Task 4: fix round 1/5 (12 addressed incl. R8 minors, 0 open; commits 823071c0..02a97b9e)
Task 4: minor (deferred): stocks-ml-entry-gate-falsified live_deployed true though gate disabled 2026-06-28 (true-with-note?); report says 66 flips vs 65 in outputs; header says "one row added" but 1 removed/5 added
Task 4: complete (commits 8eeda009..02a97b9e, review clean) -> inventory: 144 rows (108 stocks / 11 cross-asset / 25 crypto), 103 null raw_artifact
Ruling R9: Task 5 concerns accepted — (1) CI/t/power are on annualized mean arithmetic excess return (what every artifact's t tests), labeled as such in each row, while the raw CAGR gap is still shown as the point estimate the owner's goal needs; the two quantities are never conflated. (2) Long-history screens vs a Fama-French market comparator stay as rows labeled "comparator: <name>, not SPY". — Costs if wrong: Lens 2 table is less directly comparable; no verdict affected.
Task 5: implementer DONE_WITH_CONCERNS commit c67f93e1; review pending
Ruling R10: Task 5 fix round 1 includes minors 5-8 (same document) — cost if wrong: slightly larger re-review.
Task 5: review 1 — Spec ❌ Quality not approved; 4 Important (arm/bracket choice open; gap/t window ambiguity; second SE method inside a row; window choice when verdict used one of several); commits 02a97b9e..c67f93e1
Task 5: fix round 1/5 (findings 1-3,5-8 addressed; 1+4 open for static-leverage rows A23/A24, plus ensemble-mapping rule and disclosure) commits c67f93e1..799b5a7d
Task 5: fix round 2/5 (4 addressed, 0 open; commits 799b5a7d..a5fe547b)
Task 5: complete (commits 02a97b9e..a5fe547b, review clean)
Task 5: minor (deferred): stray blank line before _static_leverage_aftertax_results.json heading; mapping rule wording "class named in the inventory row" only fits the live-defaults row
Ruling R11: Task 6 is split into 6a (Lens 6 report) then 6b (Lens 1 report), each its own implementer + review; adds two output files not in the plan: docs/research/_lens6_verdicts.json ({id, spec_matches_code: yes|no|partly|unverifiable}) and docs/research/_lens1_tape_state.json ({id, tape_state_at_run: clean|contaminated|unknown}) — these are the "per-inventory-id verdicts" T6 produces for Lens 2/7. Cost if wrong: none.
Task 6a (Lens 6): implementer DONE_WITH_CONCERNS commit 36acba57; 90 rows verdicted (14 yes/28 partly/15 no/33 unverifiable); review pending. Lens 1 Haiku run-metadata passes complete: extract/lens1-out-{0..3}.json
Ruling R12: Task 6a fix round 1 includes minors 4-7 (same docs) — cost if wrong: slightly larger re-review.
Task 6a: review 1 — Spec ✅ Quality NOT approved; 3 Important (brief-contradiction framing omitted parity amendment; dividend add-back also feeds daily-loss halt start value, not "reported value only"; crypto trailing-stop verdict "no" -> "partly"), minors 4-8; commits a5fe547b..36acba57
Owner notified in chat of 7 confirmed live-path facts (spec 11.4) after review 1.
Task 6a: fix round 1/5 (7 addressed, 0 open; commits 36acba57..659e3417)
Task 6a: complete (commits a5fe547b..659e3417, review clean) -> lens6 report + _lens6_verdicts.json (90 rows: 14 yes / 29 partly / 14 no / 33 unverifiable) + _lens6_replays.json
Task 6a: minor (deferred): 17 rows not examined by Lens 6 (cheapest tests listed in report); return impact of live-vs-backtest differences unquantified
Task 6b (Lens 1): implementer DONE_WITH_CONCERNS commit 77cecdab; 55 contaminated/17 clean/72 unknown; review pending
Ruling R13: Task 6b fix round 1 includes minors 2-7 (same documents).
Task 6b: review 1 — Spec ✅ Quality NOT approved; 1 Important (BRK.B/BF.B have tape via symbol normalisation; "16 members no tape" overstated -> <=14), minors 2-7; commits 659e3417..77cecdab
Task 6b: review 1 — Spec ✅ Quality NOT approved (1 Important: BRK.B/BF.B tape recount; minors 2-7); fix round 1/5 addressed all (14 members lack in-window tape; counts 55 contaminated / 18 clean / 71 unknown); commits 659e3417..6576811e
Task 6b: minor (deferred): PARA "about 42 (2025 Q3)" is post-membership quarter minimum (label it); long lines in lens1 report
Task 6: complete (6a+6b; commits a5fe547b..6576811e, review clean)
Task 7 (Lens 7): implementer DONE_WITH_CONCERNS commit debcfaaa; 5 recomputed/27 checked/1 mismatch; prereg: 18 pre-registered/4 amended/1 late/48 no brief/43 n.a.; review pending
Ruling R14: Task 7 fix round 1 includes minors 4-5.
Task 7: review 1 — Spec ✅ Quality NOT approved; 3 Important (false "no per-study pass bar before 2026-09-24" -> 2026-09-10 audit doc has success criteria; SPY baseline stored as text in PIT/ic-kelly JSON; "no holdout arm completed" mis-attributed to Phase 0); minors 4-6; commits 6576811e..debcfaaa
Lead correction: spec section 4 hypothesis 1 holdout wording corrected (holdout PRIMARY arm completed; closure block NO-GO/pretax_fail) — committed in-branch (lead edit to my own spec text, not an implementer fix).
Task 7: fix round 1/5 (5 addressed, 0 open; commits debcfaaa..cc6cb579 incl. lead spec correction 5cf88940)
Task 7: complete (commits 6576811e..cc6cb579, review clean) -> lens7 report + _lens7_recompute.json (41 rows; 1 mismatch: position sizing; prereg: 18 pre-registered/4 amended/1 late/48 no brief/43 n.a.; academic 11 verified/1 partly/2 not examined)
Task 7: minor (deferred): stray ". ," at lens7 report ~line 323; task-7 report summary keeps pre-correction wording (fix section corrects it); `match` string-enum documentation not found by reviewer
Task 8 (Lens 3): implementer DONE_WITH_CONCERNS commit c145d1ed; gap confirmed (accrual not paid by Alpaca); eras -3.02/-2.28/+1.60 (broker vs SPY TR) vs Phase0 -1.9/-2.6/+1.4; MNST split not applied by Alpaca paper (-0.85pp); review pending
Ruling R15: Task 8 fix round 1 includes minors 2-6.
Task 8: review 1 — Spec ✅ Quality NOT approved; 1 Important (end-of-day offset paragraph misdated from 08-11 and wrongly blames MNST; correct: from 08-18, AVB alone fits), minors 2-6; commits cc6cb579..c145d1ed
Owner notified in chat of 7 confirmed live-path facts from Lens 3 (spec 11.4) after review 1.
Task 8: fix round 1/5 (6 addressed, 0 open; commits c145d1ed..a825594b)
Task 8: complete (commits cc6cb579..a825594b, review clean) -> lens3 report + _lens3_snapshot_gap.json
Task 8: minor (deferred): lens3 Q4 (~line 404) still says "two independent sources agree" contradicting new independence note; line 225 attributes whole 1.77 pp EOD move to AVB (1.10 pp is MNST via broker value); summary line 49 "+1.6 pp stock sleeve" excludes MNST only in the body
Ruling R16: Run Lens 9 (task 10 step 1) BEFORE Lens 2 (task 9): Lens 2 classification needs detectable-effect sizes (spec Lens 2 says power calculation belongs to Lens 9). Order now: T10a Lens 9 -> T9 Lens 2 -> T10b Lens 4 + Lens 5 -> T11 Lens 8 -> T12. Lens 9 writes docs/research/_lens9_power.json ({id, ...}) that Lens 2 reads. Cost if wrong: none.
Task 10a (Lens 9): implementer DONE_WITH_CONCERNS commit 21664be7; 7 computed/34 not computable/103 no artifact; replica MDE 4.24/3.14 pp/yr; xs_momentum 22.66/16.76; target gap 9.57 pp/yr (SPY CAGR 15.05% -> 24.63%); review pending
Ruling R17: Task 10a fix round 1 includes minors 4-6.
Task 10a: review 1 — Spec ❌ Quality NOT approved; 3 Important (month-end Treasury wrongly "not computable" - has Welch t; falsifier-vs-verdict mismatch for "unreachable on five years"; replica-could-detect overstated: book is 85% SPY, sleeve-scale MDE ~45 pp/yr), minors 4-6; commits a825594b..21664be7
Task 10a: review 1 fix round 1/5 (3 Important + 3 minor addressed; commits 21664be7..7fe1ceb4); Lens 9 now 8 computed / 33 not computable / 103 no artifact; month-end Treasury computed (DERIVED)
Task 10a: minor (deferred): _lens9_power.json regime.summary.negative_months still "21 of 64" / n_months 64 (report says 21 of 63); report line ~609 "no p-value key" misses cross-asset sleeve post_hoc.{pinned,long}.p_book_sharpe_gt_spy and diff_ci90 (still not computable per spec: Sharpe bootstrap)
Task 10a: complete (commit 21664be7..7fe1ceb4, review clean) -> lens9 report + _lens9_power.json
Task 9 (Lens 2): implementer DONE_WITH_CONCERNS commit 1b6c60d6; classes: 0 reliable NO / 2 unreliable NO / 103 inconclusive / 0 reliable GO / 9 unreliable GO; 72 tests in 11 families; review pending
Ruling R18: Task 9 fix round 1 includes minors 4-8 (incl. committing scripts/lens2_build.py for reproducibility).
Task 9: review 1 — Spec ❌ (column 14 wrong text for ~47 rows) Quality NOT approved; 3 Important (col 14; French-screens power under-explained; GO-rule asymmetry + A23 caveat), minors 4-8; commits 7fe1ceb4..1b6c60d6
Task 9: fix round 1/5 (3 Important + 5 minor addressed; commits 1b6c60d6..3dfacd83; classes unchanged: 0 reliable NO / 2 unreliable NO / 103 inconclusive / 0 reliable GO / 9 unreliable GO)
Task 9: complete (commits 7fe1ceb4..3dfacd83, review clean) -> lens2 report + _lens2_sensitivity.json + scripts/lens2_build.py (reproduces classifications exactly)
Task 9: minor (deferred): lens2_build.py docstring/Verification says "reads only committed files" but crypto-wfo-textbook-reset row reads gitignored results/research/research_20260511_193048/wfo_stats_snapshot.json; "anchor deployed in 17/17" is an inference from fold-attribution artifact; column 14 keyword match can pick an own quote mentioning sizing in passing
Plan: Task 10b = Lens 5 (Haiku coverage pass first, then Opus) then Lens 4 (Opus). Both depend on Lens 2.
2026-11-04 ~21:5x PST: Opus dispatch for Lens 5 failed with 5-hour usage limit (429), resets 22:20 PST; Haiku Lens 5 coverage extraction (lens5-out-0..3.json) completed before. Re-dispatching Lens 5 after reset. Owner has Max; note usage reporting per phase.
Task 10b-i (Lens 5): implementer DONE_WITH_CONCERNS commit e14494e2; 26 areas: 6 never tested / 3 tested badly / 16 inconclusive / 1 evidence against; 72 tests: 31 internal/19 LLM-web/14 practitioner/8 published; 62 of 82 WEB_RESEARCH candidates never run; review pending (Sonnet to conserve Opus quota)
Ruling R20: Task 10b-i (Lens 5) fix round 1 includes minors 4-10.
Task 10b-i: review 1 (Sonnet) — Spec ✅ Quality NOT approved; 3 Important (FOMC "evidence against" row lacks DERIVED/units caveats and over-scoped; Mid caps class inconsistent with own rule; futures overclaim vs deleted crypto carry/Kraken Futures), minors 4-10; commits 3dfacd83..e14494e2
Task 10b-i: fix round 1/5 (3 Important + 7 minor addressed; commits e14494e2..9769a79d); gap table now 28 areas: 6 never tested / 4 tested badly / 17 inconclusive / 1 evidence against (FOMC corrected-harness only, caveated)
Task 10b-i (Lens 5): complete (commits 3dfacd83..9769a79d, review clean) -> lens5 report + _lens5_coverage_matrix.json
Task 10b-i: minor (deferred): ML/LLM area label broader than evidence (ML entry filter + LLM pilot only); short-interest rule_note tension with report text; midcap400 "tested badly" rests partly on inventory-only-verified row; commit hashes 82931a40/adbae1cb not cited in crypto-perps row
Ruling R21: Lens 4 writes an extra output docs/research/_lens4_ceiling.json.
Task 10b-ii (Lens 4): implementer DONE_WITH_CONCERNS commit bad6d855; target gap by window 9.57 (2021-26) / 3.54 (2006-26) / 2.34 (1994-2026) / 0.77 (1927-2026); highest sound: 2.0x SSO +4.44 pp/yr pre-tax (-84.8% DD) = 1.25x its gap; momentum S&P500 +4.37 CI -7..+16; after-tax: 74 short-term trades keep 76%/63%; review pending (Opus)
Ruling R22: Task 10b-ii fix round 1 includes minors 4-8.
Task 10b-ii: review 1 (Opus) — Spec ✅ Quality NOT approved; 3 Important (funding-rate condition missing: 1.3x +1.61 -> +0.81/+0.80 at +2.5pt funding; headline mixes bases: 2.0x SSO ratio 1.17 not 1.25; caveats carried by codes not plain words, post-hoc best-of-five arm), minors 4-8; commits 9769a79d..bad6d855
Task 10b-ii: fix round 1/5 (3 Important + 5 minor addressed; commits bad6d855..18cac0da)
Task 10b-ii (Lens 4): complete (commits 9769a79d..18cac0da, review clean) -> lens4 report + _lens4_ceiling.json. Headline: needed gap 9.57 pp/yr (2021-26) .. 0.77 (1927-2026); no sound stock-picking result after 1994 reaches its window's gap; highest "sound" figure is post-hoc 2.0x SSO +4.15 (consistent basis) = 1.17x gap, -84.8% DD, only at 1.64% funding; registered 1.3x +1.61 -> +0.80 at +2.5pt funding; after-tax: 74 short-term round trips need +2.42 (24/15) / +5.05 (37/20) pp/yr pre-tax to tie SPY after tax.
Task 10b-ii: minor (deferred): report line ~452 "1.3x +1.61 pre-tax" missing "artifact label" qualifier; "five arms" not listed in report; sweep "realizes gains" (could be losses)
Task 10 (10a Lens 9, 10b-i Lens 5, 10b-ii Lens 4): complete.
Task 11 (Lens 8): implementer DONE_WITH_CONCERNS commit 6f5b0725; no crypto strategy beat BTC buy-and-hold over full 3-yr window; 48.46% headline in-sample, t=1.51, BTC beta 0.26; tape: no dups/shift but 43.7% Binance.US 4h bars copies of Kraken, delisted coins deleted, universe not PIT; 252-day lab convention affects no published number; live trades 0 rows; review pending (Sonnet)
Ruling R23: Task 11 fix round 1 includes minors 4-9.
Task 11: review 1 (Sonnet) — Spec ❌ (upstream implications missing) Quality NOT approved; 3 Important ("all in-sample" overstated for edge-run 2/3 YTD wins which overlap the OOS holdout; upstream implications of recovered 05-01 artifacts + mis-pointed inventory raw_artifact for textbook-reset (frictionless vs realistic-fee run); pre-05-05 tape contamination not sized), minors 4-9; commits 18cac0da..6f5b0725
Task 11: fix round 1/5 (3 Important + 6 minor addressed; commits 6f5b0725..392158e5)
Task 11 (Lens 8): complete (commits 18cac0da..392158e5, review clean) -> lens8 report + _lens8_tape_checks.json
Task 11: minor (deferred): betas 0.104/0.195 and t 0.43/-0.44 for runs 2/3 are full-12-month regression values but read as holdout-only (holdout-only: beta ~0.06/0.16, t -0.32/-0.53) - label "full-window"; changelog suggests pre-2026-03-08 live-written bars may also be time-shifted but undetectable by off-grid test; 04-02 YTD window is 26/365 not 26/366
Next: Task 12 (Phase 1 checkpoint).
Task 12: implementer DONE_WITH_CONCERNS commit 47fabb39: 29 proposed Phase 2 items (10 under 1 hour), coverage 144/144, Phase 2 not skipped. INCIDENT: implementer ran a `--help` loop that started scripts/pead_correlation_check.py (no argparse) for ~2 min (23:47-23:49:45 PST); its price loader may have refreshed cached ohlcv bars from ~2026-08-11. Lead spot-check (read-only, 23:5x PST): HAS 2026-11-04 partial bar (vol 978,471) intact; VZ 07-17/07-20 seam intact; SPY 10-28..11-03 closes intact; guard snapshot diff clean; git status clean. Review pending (Sonnet).
RULE (new, after incident): no agent may execute any script it has not read first; never run a script to see its --help (pead_correlation_check.py has no argparse and executes main()). Include in all further dispatches.
Ruling R24: Task 12 fix round 1 includes minors M1-M5 and dated errata notes at top of lens1/lens2 reports (and lens4/5 if they quote PARA prices).
Task 12: review 1 (Sonnet) — Spec ❌ Quality NOT approved; 1 Critical (incident understated: symbol_inception rows rewritten 2026-11-04 23:48:18 PST for 19 symbols [APC, ARES, CAM, CSRA, CZR, EMC, GDDY, LITE, NE, PARA, PCL, POM, QRVO, SBNY, SEDG, SPLS, STI, VEEV, XYZ] with full-range refetch (~31,967 rows) on today's adjustment basis; PARA prices now ~28.9x smaller than Lens 1 recorded; lead's earlier 'bars unchanged' check covered SPY/HAS/VZ which were not among the 19), 3 Important (Decision 0 tagging, scope text, PARA caveat), minors; commits 392158e5..47fabb39. Owner informed in chat.
Task 12: fix round 1/5 (C1 + 3 Important + minors addressed; commits 47fabb39..b195ccc0)
Task 12: complete (commits 392158e5..b195ccc0, review clean) -> docs/research/2026-11-04-phase1-checkpoint.md (29 Phase 2 items P2-01..P2-29, Decision 0 = cache-writing items default not run; guard snapshot unchanged; incident documented)
Task 12: minor (deferred): "up to 31,967" rows - only 27,579 on/after recorded first_date for CSRA/PARA/QRVO/SBNY; PARA median volume 9 on pinned window unchanged (8 on membership window); _lens1_tape_state.json lacks PARA erratum; "29 scripts" count depends on fetch_stock_ohlcv users (26 by literal grep); summary table owner-flag column omits DB-cache for P2-26/27/28
GATE (R6/spec section 6): STOP here for owner decision on Phase 2 re-run list + Decision 0 + restore question. Task 13 not started. Task 14 (synthesis) can issue with 'unknown, re-run pending' entries.
OWNER DECISIONS (2026-11-05 ~00:3x PST): Phase 2 = NONE (all 29 items P2-01..P2-29 not run; marked "unknown, re-run not run" in synthesis); Decision 0 = NO (no cache writes); the 19 rewritten symbols = LEAVE AS IS AND DOCUMENT (no restore attempt).
Ruling R25: changelog.md entry and research-snapshot regeneration are deferred until the owner accepts the synthesis (spec section 7: "once the synthesis is accepted"); Task 13 skipped by owner decision.
Task 13: skipped (owner approved none).
Task 14 started: step 1 Opus synthesis -> step 2 fresh-Opus challenger (opposite case, evidence + claims only) -> resolve -> sonnet task review -> final guard diff -> final whole-branch review.
Task 14 step 1: synthesis committed e4362512 (35-row trust table: 0 green / 20 yellow / 5 red / 10 grey)
Task 14 step 2: independent challenge committed 965d78c0 (docs/research/2026-11-04-root-cause-challenge.md): 40 claims: 20 confirmed / 16 caveat / 3 wrong / 1 not checked; objections 0 Critical / 7 Important / 12 Minor; verdict: headline numbers hold; framing too lenient on 2x question; paper shortfall -3.76 mislabelled ("broker's own" EOD = -4.57); "103 inconclusive" includes 42 status labels; replica had power to detect 2x effect at book level; leverage mixed bases; incident detail (CSRA/PARA/QRVO/SBNY keep 4,388 older rows with price-level breaks)
Task 14 step 3: resolving objections (resume synthesis author; sections "Challenge and resolutions").
Task 14 step 3 done: challenge resolved in 50983bbc (19 objections: 18 fixed / 1 partial disagreement (obj 6) / 0 unresolved; trust table 36 lines 0 green/20 yellow/6 red/10 grey).
Ruling R26: Task 14 fix round 2 includes minors 4-10.
Task 14: task review (Opus) — Spec ❌ Quality NOT approved; 3 Important (obj-6 disagreement rests on misreading: cash0 arm = zero interest, keeps 5% reserve, so challenger was right: replica -0.64 is mostly cash-reserve drag; table does not cover all 144 rows (map needed); summary lacks crypto and tax sentences), minors 4-10; commits e4362512..50983bbc. Numbers/incident facts confirmed by SELECT.
Task 14: fix round 2 (3 Important + minors addressed; commits 50983bbc..4cef0d7c); re-review (Sonnet) all addressed; objection totals 18 fixed / 1 no change / 0 disagreed / 0 unresolved (objection 6 disagreement withdrawn: cash0 arm = zero interest, keeps 5% reserve)
Task 14: minor (deferred): Q1 "most of the -0.64 (about 0.9 pp/yr)" reads as > all of -0.64 (offset +0.26 from picks); objection 6 text says B1 cites 3bp/10bp/cash0 arms but B1 cites only same-day arm; per-class counts of not-shown rows only in map file; summary still has t-values/CI/PARA/37%/20% labels undefined; line 19 groups July summary with event studies
Task 14: complete (commits b195ccc0..4cef0d7c, review clean) -> docs/research/2026-11-04-root-cause-investigation.md (+ _synthesis_inventory_map.json, 2026-11-04-root-cause-challenge.md)
Next: final whole-branch review (MERGE_BASE=main 1e419c27 .. HEAD).
Final review (Opus): Merge readiness 'ready after listed fixes' (5 must-fix + deferred triage); fix wave e7e9418f (single fixer, 8 items incl. R27 quick fixes); scoped re-review pending
Merged branch into main (fast-forward) at e7e9418f; branch deleted; not pushed. Guard baseline line 'main:' now legitimately differs.
Ruling R19 (logged at the end): Lens 5 writes an extra output docs/research/_lens5_coverage_matrix.json beyond the plan's file list.
Ruling R27 (logged at the end): the final whole-branch fix wave also included three quick consistency fixes (Lens 9 JSON negative-month count 21 of 63; Lens 8 betas relabelled "full 12-month regression"; one summary sentence on the 2026-11-04 accidental database refresh).
Final review (Opus): merge readiness "ready after listed fixes"; fix wave e7e9418f (single fixer) re-reviewed clean; branch merged into main by fast-forward at e7e9418f on 2026-11-05 and deleted locally.
