# Complete the ggTrader Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native execution or superpowers:subagent-driven-development if explicitly selected. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish and validate the project-wide graph without losing the existing graph or misrepresenting missing sources and historical evidence.

**Architecture:** Reuse the AST extraction and verified semantic cache from the prior run. Complete the missing semantic sources in bounded pieces, audit graph integrity, then build a separate candidate and publish only after its coverage and health are independently checked.

**Tech Stack:** Installed graphifyy interpreter recorded in `graphify-out/.graphify_python`, Python, graphify semantic cache, JSON, HTML.

**Spec:** The owner's `/graphify` request and the graphify skill at `/home/flynn/.agents/skills/graphify/SKILL.md`; the current evidence is `graphify-refresh/GRAPH_REPORT.md` and `graphify-refresh/missing.json`.

## Global Constraints

- Preserve `graphify-out/graph.json`, its report, the current `graphify-refresh/` candidate, all source data, and the semantic cache until a replacement has passed verification. Never use `--force` to bypass the shrink guard.
- The project is research-first. Preserve NO-GO findings, dates, paper-trading freezes, and the owner's standing instructions in `AGENTS.md`; historical recommendations are not current decisions.
- Run no more than one semantic delegate at a time: ten concurrent delegates caused an out-of-memory kill and heavy swap use. Check `free -h` before each phase; stop launching work if memory pressure returns.
- The owner reported ample quota on 2026-11-13; remaining quota and delegate token usage are not visible. Report actual available usage after each phase, or say it is unavailable. Recheck headroom if the owner changes that preference.
- When delegating, follow the project's builder/reviewer model separation and exclude private account records from delegate prompts. The lead verifies coverage, extraction, and figures; do not delegate unless authorized by the chosen execution method.
- Do not modify source documents, database, cron, production containers, or `data/`. Do not commit generated graph artifacts or the plan unless the owner separately asks.
- Treat reported token counts of zero as unknown when the provider did not return usage. Preserve source provenance and distinguish EXTRACTED, INFERRED, and AMBIGUOUS relationships.

## Review Focus

- **Dense or unusually long source lines:** prove the register and intraday text coverage reaches the final byte, not just the claimed line count (Task 1).
- **Empty source:** record the zero-byte issuer page as successfully inspected without inventing a semantic node (Task 1).
- **External imports:** distinguish valid references to standard-library or third-party modules from missing internal targets; do not delete real edges to make a diagnostic look clean (Task 2).
- **Old plans and verdicts:** sampled historical claims retain their document date and are not presented as current code or recommendations (Task 2).
- **Reruns and publication:** a failed, smaller, or incomplete candidate cannot overwrite the prior graph; verify artifact consistency and manifest behavior before publication (Task 3).

## File Map

- Read: `graphify-refresh/extraction.json`, `graphify-refresh/detection.json`, `graphify-refresh/missing.json`, `graphify-refresh/GRAPH_REPORT.md`, and `graphify-out/` semantic cache.
- Read: `research/VERDICT_REGISTER.md` and the three intraday paths named in `graphify-refresh/missing.json`.
- Create during execution: a temporary, isolated candidate directory outside tracked source, with extraction fragments, explicit coverage records, graph, report, HTML, manifest, and diagnostics. Keep interim fragments until the candidate is verified.
- Publish after verification: `graphify-refresh/` as the explicitly named project-wide graph; preserve the pre-existing `graphify-out/` graph and its update metadata.
- Modify: no application code, research evidence, or dependency files.

### Task 1: Complete and audit semantic coverage

**Files:** Read the four paths in `graphify-refresh/missing.json`; write temporary fragments and a coverage ledger in the candidate directory.

**Interfaces:** Consumes existing semantic cache and extraction spec at `/home/flynn/.agents/skills/graphify/references/extraction-spec.md`. Produces `semantic_complete.json` (`nodes`, `edges`, `hyperedges`) and `coverage.json` mapping each source to its SHA-256, byte size, disposition, and contiguous byte ranges reviewed.

- [ ] **Step 1:** Re-read `graphify-refresh/missing.json`; record `sha256sum`, byte size, and line count for each file. Verify the issuer page is zero bytes. Inspect `market_data_docs_canonical.txt` and `market_data_faq.txt` in byte-bounded pages; check whether the prior line-count discrepancies are missing-newline counting errors or unread content.
- [ ] **Step 2:** Split the 496,615-byte, 2,079-line verdict register into non-overlapping, line-aligned temporary sections small enough for one context. Record original start/end byte offsets and line numbers for every section. Ensure the sections concatenate byte-for-byte to the original (`cmp`), and never modify the source.
- [ ] **Step 3:** Extract each section sequentially using the graphify schema. Preserve the original absolute source path and original line numbers on every node and edge; use stable IDs based on the original source path and entity, not the temporary section. Merge section fragments without silently dropping duplicate IDs with conflicting content. Read the two nonempty intraday sources completely and extract them the same way.
- [ ] **Step 4:** Validate JSON, allowed `file_type` values, confidence and confidence scores, absolute originating paths, and complete non-overlapping coverage against the current source hashes. Confirm the empty file is recorded as `empty` and produces no invented content. Compare a sample of at least five register entries, including a NO-GO and a historical/superseded item, against their original lines. If a section cannot be completed or validated, mark it partial and do not stamp the source complete.
- [ ] **Step 5:** Save verified results to the semantic cache with `save_semantic_cache` using the same extraction-spec path as the previous run. Record unknown provider usage as unknown, not zero. Confirm the three nonempty files are cache hits and the empty file is separately recorded as inspected; otherwise retain their missing status.

### Task 2: Audit graph integrity and historical claims

**Files:** Read `graphify-refresh/extraction.json`, `graphify-refresh/graph.json`, and Task 1 outputs; write `integrity.json` and a short human-readable integrity note in the candidate directory.

**Interfaces:** Consumes Task 1's `semantic_complete.json` and coverage ledger; produces `integrity.json` with counts, categories, and representative source-backed examples for dangling endpoints, self-loops, and collapsed endpoint pairs.

- [ ] **Step 1:** Reconstruct the combined extraction from the existing AST and semantic cache plus Task 1. Compare node IDs before deduplication: retain a unique ID only when its source and meaning match; resolve conflicts by tracing the actual source, never by selecting the first record silently.
- [ ] **Step 2:** Run `graphify.diagnostics.diagnose_extraction` on the raw extraction. Compare against the current baseline of 4,939 dangling-endpoint edges, 33 self-loops, and 1,519 collapsed undirected endpoint pairs. Classify dangling targets into external modules, references to omitted sources, and unresolved internal IDs; report counts and sample each class. Verify standard-library imports such as `argparse` are not automatically called broken internal links.
- [ ] **Step 3:** Sample at least ten cross-file semantic relationships and five historical plan or verdict nodes against their cited source. Correct unsupported confidence labels, false source attribution, or stale-status claims in the candidate; if a claim cannot be verified, mark it AMBIGUOUS or omit that claim, recording the reason. Do not alter the underlying evidence.
- [ ] **Step 4:** Re-run the diagnostic after corrections. The candidate passes this task only if every remaining internal dangling edge is accounted for with source and reason, self-loops and collapsed pairs are explained or corrected, and before/after counts are reported. External references may remain and must be disclosed. A clean count alone is not a reason to drop edges.

### Task 3: Build, verify, and publish the candidate

**Files:** Read Task 1 and 2 outputs; create candidate `graph.json`, `GRAPH_REPORT.md`, `graph.html`, `manifest.json`, and diagnostics; publish to `graphify-refresh/` only after all checks pass.

**Interfaces:** Consumes the verified combined extraction, `coverage.json`, and `integrity.json`. Produces a graph whose source-root metadata is `/home/flynn/ggTrader`, plus an auditable report and a manifest that stamps only successfully inspected files.

- [ ] **Step 1:** Recheck `free -h`; use `extract(..., parallel=False)` if AST regeneration is necessary. Build the graph in a new temporary candidate directory, with the repository root as the graph source root. Reject an empty graph and investigate any shrink versus `graphify-out/graph.json`; never force an overwrite. Record the installed graphify version and exact commands.
- [ ] **Step 2:** Generate clusters, meaningful community labels where supported by sampled members, the report, graph JSON, and HTML. Warn that the graph exceeds 5,000 nodes and verify HTML uses the aggregated community view. Report semantic coverage, graph-health counts, unknown token usage, and the skipped Office conversion and PDF limitations if still applicable.
- [ ] **Step 3:** Check JSON parses; every graph node's community label matches the report; HTML and report exist and refer to the same candidate; every nonempty detected source is covered or explicitly listed as incomplete; the zero-byte source is recorded as inspected. Verify a known `Bars` node and a sampled verdict-register node resolve to their original source paths. Test incremental detection against the candidate manifest: unchanged sources stay unchanged, while a temporary copy with one modified source is detected as changed. Never edit real research files for this test.
- [ ] **Step 4:** Check that a deliberately incomplete or smaller candidate cannot overwrite the old graph, using a temporary copy and the standard graphify shrink guard. If any required check fails, leave both existing graphs untouched and report the failure and retained candidate path.
- [ ] **Step 5:** Once every check passes, back up the current `graphify-refresh/` directory and atomically replace it with the candidate; do not replace `graphify-out/`. Reopen the published JSON and report to verify them, and report output paths, coverage exceptions, health counts, token-usage availability, and whether the output is ignored or untracked in Git.
