# Academic Survey Report Rebuild and Skill Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the stock-picking literature survey (version 4, 66,200 words) as a paper that reads as an independent academic literature survey in strict APA 7 and plain language, and update the `literature-survey-report` skill so the next survey is built to the same standard.

**Architecture:** A new build folder holds writer part files that use citation placeholders; an updated assembly script converts them to APA, builds the reference list, table of contents and anchors; three check scripts (preservation, voice, APA) plus the existing quotation checker gate every build; review and fix passes follow; the report is installed only after all gates and the final review pass. Process history moves to a separate log file.

**Tech Stack:** Python 3 scripts (standard library only), markdown, the existing pipeline in `/home/flynn/.claude/skills/literature-survey-report/scripts/` and `/media/thesix/ggTrader-data/library-reading/rewrite_v2/`, agents (Claude Sonnet and Opus, Codex Sol and Astra).

**Spec:** `/home/flynn/ggTrader/docs/superpowers/specs/2026-11-22-academic-survey-report-redesign-design.md` (including its two amendments). Executors read the spec first.

## Global Constraints

- Build folder: `/media/thesix/ggTrader-data/library-reading/academic_v5/`. The installed `CONSENSUS_REPORT.md` is touched only in Task 11, after a backup.
- Quotations stay character-for-character; each keeps its APA citation and hidden `<!--id-->` tag on one line; `python3 -I /media/thesix/ggTrader-data/library-reading/inventory/verify_report.py <file>` must end "0 problems".
- Grades, numbers, sample periods, draft or journal labels, and caveats are unchanged from version 4. Nothing is added that version 4 or its reader notes do not contain.
- Strength rule: STRONG = two or more independent published sources agree on measured data; MODERATE = one published source with consistent support; WEAK = single source, preprint, simulation or opinion; "No evidence" is a separate label.
- Visible text uses none of: owner, library (as the user's own), readers, our, we, I, round, revision, gap round, V4, outside run, Analyst judgment, recommended path, paper trade, Step N labels. Absence statements read "no surveyed passage" and stay scoped.
- Headings in APA title case and unnumbered; tables numbered with a bold number line, italic title and a note line; numbers per APA (words for 0 to 9, numerals from 10); "&" in parentheses, "and" in running text; "et al." from three authors.
- Technical term glossed in the same sentence at first use in each major section; "4 basis points, or 0.04%" style for units.
- Delegation: theme writers Claude Sonnet; framing sections and skill rewrite Claude Opus; cold reviews Codex Sol (cross-family) and Claude Opus (labelled less independent); final review `gpt-6.2-astra`, once, findings only. No agent edits the installed report. Orchestrator re-runs every check itself and verifies each reviewer finding against source text. Report token usage after each phase.
- Tests for the new scripts are small and memory-light and live under the skill's `scripts/tests/`; run them with `python3 -I -m pytest <that folder> -q` (they do not touch the ggTrader repo or its test tiers).

## Review Focus

- A paragraph whose caveat was left behind when its claim moved to a new theme: the check must show each moved claim next to its caveat.
- A number that appears in version 4 and nowhere in the new report or the log: `check_preservation.py` must list it.
- A gloss that is missing at the first use of a term in a section that is entered from the table of contents.
- A citation with "and" inside parentheses, "et al." with two authors, or an unnumbered or unmatched table, which breaks APA.
- Project vocabulary surviving in an appendix or a table note.
- A draft figure presented as a journal figure after the merge of "round" paragraphs into themes.

---

### Task 1: Check scripts

**Files:**
- Create: `/home/flynn/.claude/skills/literature-survey-report/scripts/check_preservation.py`
- Create: `/home/flynn/.claude/skills/literature-survey-report/scripts/check_academic_voice.py`
- Create: `/home/flynn/.claude/skills/literature-survey-report/scripts/check_apa.py`
- Create: `/home/flynn/.claude/skills/literature-survey-report/scripts/tests/test_checks.py` and small fixture files beside it.

**Interfaces:**
- Produces: `check_preservation.py OLD NEW [--log LOG] [--allow-drop JSON]`: prints counts and the items of four kinds (numbers, quotations, citation ids from `<!--id-->` tags, grade labels) present in OLD but in neither NEW nor LOG nor the allowed-drop list; exit 1 when any remain.
- Produces: `check_academic_voice.py REPORT [--terms FILE]`: exit 1 on a forbidden word (list in the script), a missing required heading (Abstract, Summary of Findings, Table of Contents, Introduction, Background, Survey Method, Literature Review, Discussion, Directions for Future Research, Conclusion, References), a numbered heading, a paragraph over 120 words outside tables and appendix quotations, or a glossary term at its first use in a section without a gloss (a comma, "meaning", "that is" or parenthetical within the same sentence).
- Produces: `check_apa.py REPORT`: exit 1 on "and" inside a parenthetical citation or "&" in a narrative citation, "et al." after fewer than three authors, contractions, "etc.", a numeral under 10 outside units or percentages, a sentence starting with a numeral, a table without number line, title and Note line, a citation without a reference entry or the reverse, and a reference entry without a year.

- [ ] **Step 1: Write failing tests** in `test_checks.py`: one fixture per rule, each with one violation and one clean twin (for example `"(Smith and Lee, 2020)"` fails, `"(Smith & Lee, 2020)"` passes; "We found" fails the voice check; a table missing its Note line fails APA; an old number absent from new fails preservation but passes when listed in `--allow-drop`).
- [ ] **Step 2: Run `python3 -I -m pytest /home/flynn/.claude/skills/literature-survey-report/scripts/tests -q`.** Expected: all fail (scripts missing).
- [ ] **Step 3: Implement the three scripts** with the interfaces above, regex-based, standard library only, each with `main()`, `argparse` and `--help`.
- [ ] **Step 4: Rerun the tests.** Expected: pass.
- [ ] **Step 5: Run each script on version 4** (`CONSENSUS_REPORT.v4_before_layout_fix.md`) to confirm they flag it (expect many voice and APA findings; preservation of V4 against itself reports none). Save the outputs to `academic_v5/baseline_v4_checks.txt`.

### Task 2: Workspace, inventory and theme map

**Files:**
- Create: `academic_v5/` with `parts/`, `log/`, `inventory/`
- Create: `academic_v5/inventory/v4_inventory.json`, `academic_v5/inventory/theme_map.md`
- Create: `academic_v5/WRITER_BRIEF.md`

**Interfaces:**
- Produces: `v4_inventory.json`: for every V4 section and Appendix B block, its id, heading, word count, the citation ids it cites and the numbers it states.
- Produces: `theme_map.md`: a table assigning each V4 section, "round" paragraph and Appendix B block to exactly one of the new themes (Backtest Validity and Data Bias; Trading Costs and Execution; Taxes; Short-Horizon Patterns and Daily Trading; Slower Patterns and Published Signals; Machine Learning; Evidence on Individual Investors and Funds), to a framing section, to an appendix or to the log.
- Produces: `WRITER_BRIEF.md`: the outline, voice rules, APA rules, glossing rule, theme template, placeholder syntax (`{id|loc}`, `{n:id|loc}`), forbidden words, the no-new-claims rule and the check commands, copied from the spec.

- [ ] **Step 1: Copy** version 4 (`CONSENSUS_REPORT.md`), `references.json` and the scripts into `academic_v5/`; record file hashes in `academic_v5/HASHES.txt`.
- [ ] **Step 2: Build `v4_inventory.json`** with a short script (`build_inventory.py`) and print the section word counts.
- [ ] **Step 3: Write `theme_map.md`**; check with a script that every V4 section and block is assigned exactly once.
- [ ] **Step 4: Write `WRITER_BRIEF.md`**; confirm it names every forbidden word and APA rule in the spec.

### Task 3: Theme writers (seven parallel agents)

**Files:**
- Create: `academic_v5/parts/theme_1.md` to `theme_7.md`, `academic_v5/parts/theme_summary_table.md`

**Interfaces:**
- Consumes: `theme_map.md` rows for the theme, the V4 text and Appendix B blocks assigned to it, reader notes (`gap-research/R9_returned_reports/read/M*_notes.json`, earlier `L*_notes.json`), `WRITER_BRIEF.md`.
- Produces: one part file per theme in the fixed template (heading; "In plain words" line; finding and grade; evidence table with caption and note; short paragraphs; points of disagreement; limits; "What the literature establishes" line), placeholders for every citation.

- [ ] **Step 1: Dispatch seven Sonnet writers** in parallel, each with the brief, its theme's source text and the template.
- [ ] **Step 2: Run on each part** `verify_report.py` after placeholder conversion (Task 5) is available; until then run `check_academic_voice.py` and a placeholder-id check on the part. Expected: 0 forbidden words, every id known.
- [ ] **Step 3: Orchestrator spot check** of at least five numbers and glosses per part against source text; reject and resend a part that fails.

### Task 4: Framing sections and appendices (Claude Opus)

**Files:**
- Create: `academic_v5/parts/front.md` (title block, abstract with keywords, summary of findings), `introduction.md`, `background.md` (with glossary), `method.md`, `discussion.md`, `future_research.md`, `conclusion.md`, `appendix_a.md` to `appendix_g.md`, `log/REVIEW_AND_VERSION_LOG.md`

**Interfaces:**
- Consumes: the seven theme parts and the V4 text; the design spec.
- Produces: framing parts that cite themes by heading name with links; `method.md` states corpus composition (four source categories), reading procedure with the language-model disclosure, verification, the grading rule and corpus limits; `future_research.md` states each step impersonally with success and failure criteria and labels the survey's own steps "Interpretation"; appendices keep quotations and tables with APA captions; the log receives every dropped process item with a reason.

- [ ] **Step 1: Dispatch one Opus writer for front matter, introduction, discussion, future research, conclusion;** one for background and method; one for appendices A to G and the log.
- [ ] **Step 2: Check `method.md`** names "No evidence" and the four categories and contains the disclosure sentence agreed with the owner.
- [ ] **Step 3: Check the grade table** in the summary of findings equals V4's grades (script comparing label counts per theme).

### Task 5: Assembly

**Files:**
- Modify (copy into `academic_v5/` first): `assemble.py`, `convert_cites.py`
- Create: `academic_v5/CANDIDATE.md`

**Interfaces:**
- Consumes: part files in outline order and `references.json`.
- Produces: `CANDIDATE.md` with unnumbered title-case headings and stable anchors (`<a id='...'>` derived from the heading), a table of contents with one short phrase per top-level entry, APA in-text citations plus hidden tags, the sorted reference list, appendix lettering and table numbering; the script reports unresolved links and placeholders and exits 1 on any.

- [ ] **Step 1: Adapt `assemble.py`** to the new outline, unnumbered headings and appendix lettering (Appendix A to G only).
- [ ] **Step 2: Run it.** Expected: 0 unresolved links, 0 leftover placeholders.
- [ ] **Step 3: Run `verify_report.py CANDIDATE.md`.** Expected: "0 problems".

### Task 6: Gates

**Files:**
- Create: `academic_v5/GATES.txt` (the output of the run)

- [ ] **Step 1: Run the five gates** on `CANDIDATE.md`: `verify_report.py`; `check_preservation.py CONSENSUS_REPORT.md CANDIDATE.md --log log/REVIEW_AND_VERSION_LOG.md --allow-drop inventory/allowed_drops.json`; `check_academic_voice.py`; `check_apa.py`; anchor and link check.
- [ ] **Step 2: Fix** each finding in the owning part file and rerun until every gate is clean or the finding is a recorded deliberate drop.
- [ ] **Step 3: Orchestrator reads** the abstract, summary of findings and two themes end to end and compares them with V4 for changed meaning.

### Task 7: Cold reviews and fix pass

**Files:**
- Create: `academic_v5/REVIEW_BRIEF.md`, `REVIEW_SOL.md`, `REVIEW_OPUS.md`, `FIX_LIST.md`

- [ ] **Step 1: Write `REVIEW_BRIEF.md`** (fact fidelity of at least 50 claims, label justification against the rule, scope of absence statements, voice and APA conformance, glosses, whether it reads as a paper from title to appendices; draft versus journal labels).
- [ ] **Step 2: Dispatch Codex Sol (high effort) and a fresh Opus reviewer** with only the brief, the candidate and the sources.
- [ ] **Step 3: Orchestrator verifies each finding** against the source text and writes `FIX_LIST.md` (accept or decline with evidence).
- [ ] **Step 4: Opus applies the fix list** to the part files; reassemble; rerun all gates until clean.

### Task 8: Style pass and Astra final review

- [ ] **Step 1: Run the APA and voice scripts again;** an Opus pass fixes remaining findings in the part files; reassemble; gates clean.
- [ ] **Step 2: Dispatch `gpt-6.2-astra`** (high effort) once with the design, the candidate and the style rules to review the whole report; it writes `academic_v5/REVIEW_ASTRA.md` and edits nothing.
- [ ] **Step 3: Orchestrator verifies each Astra finding,** applies the real ones in the part files, reassembles, reruns all gates, and records Astra's usage.

### Task 9: Install

- [ ] **Step 1: Confirm the gates** are clean on the final `CANDIDATE.md`.
- [ ] **Step 2: Back up** `CONSENSUS_REPORT.md` to `CONSENSUS_REPORT.v4_before_academic_v5.md`, copy the candidate over it, and copy the log to `/media/thesix/ggTrader-data/library-reading/REVIEW_AND_VERSION_LOG.md`.
- [ ] **Step 3: Run `verify_report.py`** on the installed file. Expected: "0 problems".
- [ ] **Step 4: Update the project memory note** with the new layout, backups and the log location.

### Task 10: Skill update and test

**Files:**
- Modify: `/home/flynn/.claude/skills/literature-survey-report/SKILL.md`, `references/style_and_citation_brief.md`, `scripts/assemble.py`, `scripts/convert_cites.py` (the versions proven in Task 5)
- Check and adjust if needed: `/home/flynn/.claude/skills/objective-report/SKILL.md`

- [ ] **Step 1: Rewrite `SKILL.md`** with the new outline, paper voice, APA 7 rules, plain-language rule, theme template, process-log rule, the three check scripts in the pipeline, and a lessons section from this build.
- [ ] **Step 2: Rewrite `style_and_citation_brief.md`** from `WRITER_BRIEF.md`; move project specifics into a "project notes" block that governs scope wording only.
- [ ] **Step 3: Read `objective-report/SKILL.md`** for a reader's guide, project vocabulary or "Analyst judgment"; change the shared habits and list the changes.
- [ ] **Step 4: Test the skill:** assemble one theme and the front matter from the updated brief into a scratch folder and run the three check scripts and `verify_report.py`. Expected: all clean.
- [ ] **Step 5: Report** what changed, check results and token usage by phase.

## Self-Review Notes

- Spec coverage: front matter, body themes, directions for future research, appendices A to G and log (Tasks 3 to 5); voice, APA, glossing (Tasks 1 and 6); grades and non-corpus material (Task 4); disclosure (Task 4 step 2); reviews and Astra (Tasks 7 and 8); skill update (Task 10); install and backups (Task 9).
- Proportion: this plan names interfaces and gates and leaves writing to the agents.
