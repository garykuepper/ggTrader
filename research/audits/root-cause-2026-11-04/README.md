# Root-cause investigation, 2026-11-04: start here

This folder holds the investigation into why the project's stock and crypto research has not beaten SPY (the S&P 500 index fund used as the benchmark) by much. It is diagnosis only: nine checks ("lenses"; together they are Phase 1, the main part of the investigation) on the existing record, one synthesis, an independent challenge and two direction reviews (one written by Claude Opus, one by Google's Gemini 4 Argon). This page is a route into the evidence, not another summary.

## Read in this order

1. [Synthesis](findings/2026-11-04-root-cause-investigation.md): the committed conclusions, with every objection from the challenge resolved in its section 9.
2. [Consolidated direction review](reviews/2026-11-05-direction-reviews-consolidated.md): the reading copy that compares the two direction reviews, lists the owner's open decisions and offers at most three next-step options.
3. [Independent challenge](findings/2026-11-04-root-cause-challenge.md): a second reader's attack on the synthesis, checked against the raw files.

## What was and was not done

- **Phase 2 re-runs were not performed.** Phase 2 is the optional re-run list at the end of the first investigation phase. Its 29 items (P2-01 to P2-29 in the [Phase 1 checkpoint](findings/2026-11-04-phase1-checkpoint.md)) were never run, by the owner's decision of 2026-11-05. Every question they would settle is "unknown, re-run not run", not "passed".
- **The Argon review is unverified as written.** It is output from a different model family (Gemini 4 Argon). Of its 105 path citations, 43 are broken or unsupported, and several specifics are invented. Read it only together with the [verification](reviews/2026-11-05-argon-review-verification.md), which lists what survives. The consolidated review uses only the verified parts.
- **Phase 1 evidence on 19 price series is not exactly reproducible.** An accidental database refresh rewrote price history for 19 symbols on 2026-11-04. The [Phase 1 checkpoint](findings/2026-11-04-phase1-checkpoint.md) records the incident, and Lens 1 and Lens 2 carry an erratum.

## Files

NO-GO means a test failed its pre-set standard; the tape is the project's stored price history.

### Summary

| File | What it is |
|---|---|
| [2026-11-04-root-cause-investigation.md](findings/2026-11-04-root-cause-investigation.md) | The synthesis of the nine lenses: findings, limits (section 7), owner decisions recorded and open (section 8), and challenge resolutions (section 9). |
| [2026-11-05-direction-reviews-consolidated.md](reviews/2026-11-05-direction-reviews-consolidated.md) | Editorial comparison of the Opus and Argon direction reviews; no new computation. |
| [2026-11-04-phase1-checkpoint.md](findings/2026-11-04-phase1-checkpoint.md) | Where Phase 1 stopped, the database incident, and the proposed 29-item Phase 2 re-run list that was not run. |
| [2026-11-04-phase0-trust-audit.md](findings/2026-11-04-phase0-trust-audit.md) | Early single-reviewer trust audit of the stock results (unreviewed subagent output; crypto not covered). |

### Lenses

| File | What it is |
|---|---|
| [2026-11-04-lens1-backtest-validity.md](lenses/2026-11-04-lens1-backtest-validity.md) | Lens 1: what contamination remains in the stock price data and backtests after the five known fixes. |
| [2026-11-04-lens2-sensitivity-spec.md](lenses/2026-11-04-lens2-sensitivity-spec.md) | Lens 2 method, fixed in advance: the rules for classifying each result. |
| [2026-11-04-lens2-verdict-audit.md](lenses/2026-11-04-lens2-verdict-audit.md) | Lens 2: which NO-GO and GO verdicts can be trusted, and for what size of effect. |
| [2026-11-04-lens3-live-parity.md](lenses/2026-11-04-lens3-live-parity.md) | Lens 3: how the live paper trader compares with its backtest. |
| [2026-11-04-lens4-feasibility-and-ceiling.md](lenses/2026-11-04-lens4-feasibility-and-ceiling.md) | Lens 4: whether the target is feasible and the measured ceiling on what could be reached. |
| [2026-11-04-lens5-coverage-gaps.md](lenses/2026-11-04-lens5-coverage-gaps.md) | Lens 5: what the project's record contains and lacks. |
| [2026-11-04-lens6-implementation-correctness.md](lenses/2026-11-04-lens6-implementation-correctness.md) | Lens 6: whether the code does what the spec says and whether live runs what was tested. |
| [2026-11-04-lens7-research-accuracy.md](lenses/2026-11-04-lens7-research-accuracy.md) | Lens 7: whether the quoted research numbers recompute from the raw files. |
| [2026-11-04-lens8-crypto.md](lenses/2026-11-04-lens8-crypto.md) | Lens 8: crypto, what was tried, what the evidence shows and what is unknown. |
| [2026-11-04-lens9-power-and-regime.md](lenses/2026-11-04-lens9-power-and-regime.md) | Lens 9: statistical power (what size of effect each test could detect) and regime. |

### Reviews

| File | What it is |
|---|---|
| [2026-11-04-root-cause-challenge.md](findings/2026-11-04-root-cause-challenge.md) | Independent adversarial review of the synthesis, checked against the raw artifacts. |
| [2026-11-05-senior-quant-direction-review.md](reviews/2026-11-05-senior-quant-direction-review.md) | The Opus direction review: a recommended direction plus the owner's decision list (reliable, single reviewer). |
| [2026-11-05-argon-synthesis-review.md](reviews/2026-11-05-argon-synthesis-review.md) | The Argon direction review, kept unmodified and **unverified as written** (see the warning above). |
| [2026-11-05-argon-review-verification.md](reviews/2026-11-05-argon-review-verification.md) | Claude's line-by-line check of the Argon review: which citations and claims hold and which do not. |

### Evidence

| File | What it is |
|---|---|
| [2026-11-04-investigation-inventory.md](records/2026-11-04-investigation-inventory.md) | Inventory of the 144 material results (every result that could change a verdict) that every lens covers, rendered from the inventory JSON. |
| [2026-11-04-investigation-ledger.md](records/2026-11-04-investigation-ledger.md) | Copy of the working ledger: task completions, review rounds, owner decisions and rulings. A working log, not a deliverable. |
| [_investigation_inventory.json](records/_investigation_inventory.json) | Source data for the inventory page (144 rows). |
| [_synthesis_inventory_map.json](records/_synthesis_inventory_map.json) | Map from the synthesis tables to inventory rows. |
| [_investigation_guard_baseline_2026-11-04.txt](records/_investigation_guard_baseline_2026-11-04.txt) | Snapshot of the intraday-observer test copy, its timers, the live container and the main checkout, taken before the work started to prove the live system was not touched. |
| [_lens1_tape_state.json](lenses/_lens1_tape_state.json) | Lens 1 price-tape state for each inventory row. |
| [_lens2_sensitivity.json](lenses/_lens2_sensitivity.json) | Lens 2 classification rules and the classification of every result. |
| [_lens3_snapshot_gap.json](lenses/_lens3_snapshot_gap.json) | Lens 3 live-account snapshot and return comparison by era. |
| [_lens4_ceiling.json](lenses/_lens4_ceiling.json) | Lens 4 target, tiers and measured ceiling. |
| [_lens5_coverage_matrix.json](lenses/_lens5_coverage_matrix.json) | Lens 5 coverage classification of all 144 rows. |
| [_lens6_replays.json](lenses/_lens6_replays.json) | Lens 6 hand replays: four strategies, one variant and one synthetic check of backtest fill semantics. |
| [_lens6_verdicts.json](lenses/_lens6_verdicts.json) | Lens 6 code-correctness verdicts for 90 inventory rows; 33 are marked unverifiable. |
| [_lens7_recompute.json](lenses/_lens7_recompute.json) | Lens 7 recomputed figures and the academic claims it checked. |
| [_lens8_tape_checks.json](lenses/_lens8_tape_checks.json) | Lens 8 crypto tape checks, benchmark comparison and annualization. |
| [_lens9_power.json](lenses/_lens9_power.json) | Lens 9 power result for each of the 144 rows; where a detectable effect could not be computed the row says so. |

Raw JSON and text files here are historical evidence: treat them as read-only. To reproduce Lens 2's classifications, run `scripts/lens2_build.py`; the other lenses' scripts were not kept in the repository (the synthesis, section 7, says what a fresh reader can and cannot reproduce).
