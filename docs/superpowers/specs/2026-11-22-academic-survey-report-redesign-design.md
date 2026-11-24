# Academic literature survey: redesign of the stock-picking report and of the report skill (design for owner approval)

Date: 2026-11-22. Status: draft, nothing built. The report (`/media/thesix/ggTrader-data/library-reading/CONSENSUS_REPORT.md`, version 4, installed 2026-11-22) and the skill (`~/.claude/skills/literature-survey-report/`) are unchanged by this document.

## Intent and why

The owner keeps the survey open as a reference and reads one section at a time while working. They want the whole document, from the abstract to the appendices, to read as a published academic literature survey in an objective voice. The earlier clean version (31,500 words, before the gap round) had that shape. Four rounds of additions turned it into a project report: 66,200 words, "gap round" 36 times, "V4 round" 19 times and "fourth round" 13 times inside the findings, the owner and the local library named throughout, a reader's guide, a version-history appendix and 184 bold lead-ins. The report skill also still describes that project-report shape and must produce the paper style from now on.

## Goals

1. Every part of the report reads as a paper by independent authors surveying a literature: no project voice, no mention of an owner, a local library, readers, rounds, reviews or revision history in the body or appendices.
2. Each section stands alone for a reference reader: one fixed layout for every theme, terms defined on first use in each section, stable numbered headings and anchors, one table of contents as the only navigation aid.
3. Nothing true is lost: every number, quotation, citation and strength label in version 4 appears in the new text or in the separate log file, or is deliberately dropped and listed.
4. Everything stays honest: the corpus is described as what it is (a curated set of 148 sources plus checked leads), absence statements stay scoped, draft versus journal labels stay, the strength rule is applied literally.
5. The skill is updated so the next report follows the same rules with scripts that check them.

## Non-goals

- No new findings, sources or searches in this pass.
- No change to the evidence, strength labels or conclusions beyond what the wording rules force (for example "the readers read" becomes "the surveyed passages").
- No advice about trading or accounts; the research setting (individual US investor, taxable account, free data, NYSE and Nasdaq stocks) is stated as the scope, not as a personal constraint.
- Not building the all-stocks test bench; its design stays unapproved.

## Anti-goals (failure even if the checks pass)

- A report that still sounds like a project (owner, library, round, "Analyst judgment", "recommended path", paper trading) anywhere, including the appendices.
- Process history left in the body, or removed in a way that hides a limitation (the verification and the corpus limits must stay in section 3 and section 5.3).
- A caveat lost while moving a claim; a draft number presented as a journal number.
- Sections that need other sections to be understood (undefined terms, "as above").

## Constraints

- Quotations stay character-for-character; every quotation keeps its citation and hidden id tag, so `verify_report.py` still reports 0 problems.
- Strength rule (section 3.4): STRONG = two or more independent published sources agree on measured data; MODERATE = one published source with consistent support; WEAK = a single source, preprint, simulation or opinion; NONE FOUND is a separate label.
- Evidence sources stay in the library; the log and id lookup live beside the report in `/media/thesix/ggTrader-data/library-reading/`.
- Existing backups stay; the new report is built in a separate folder and installed only after checks and review.

## Approaches considered

| | What | Trade-off |
|---|---|---|
| A. Edit version 4 in place | Strip process words, merge round paragraphs into themes, shorten | Least work, but the structure under it stays uneven |
| **B. Rebuild each theme to a fixed template (picked)** | Rewrite section 4 theme by theme to one layout, rewrite the framing sections in paper voice, move process history to a log, trim the appendices | More work; consistent and easy to look things up in; body about 20,000 to 25,000 words |
| C. Two documents (short survey plus long evidence book) | Survey and evidence tables in separate files | Clean to read, but a reference reader must jump between files |

The owner picked B in the conversation and added that the entire document must read as a legitimate academic paper, and that the skill must be updated to follow the same style.

## The design

### Front matter

- **Title:** "Can an individual investor beat a passive S&P 500 fund after costs and taxes by selecting stocks? A literature survey". Author line and date line kept neutral (date and version only).
- **Abstract** (about 200 words): question, corpus and method, main finding, what remains unknown. No mention of rounds.
- **Summary of findings** (about one page; replaces "Executive summary"): the five research questions answered, each in one to three sentences with its strength grade and a link to its section. The limits that matter most follow in a short list. No "recommended path".
- **Table of contents** with one short phrase per top-level section; the only navigation aid. No reader's guide and no "how to read" section.

### Body

1. **Introduction:** 1.1 problem; 1.2 why it is hard; 1.3 research questions; 1.4 scope and success criteria; 1.5 organization (one paragraph, the usual "the remainder of this survey is organized as follows").
2. **Background:** 2.1 how stock selection is measured; 2.2 the index-fund yardstick; 2.3 costs and taxes in brief; 2.4 how tests on past data mislead; 2.5 changes in market and tax rules over time (dated); 2.6 glossary.
3. **Survey method:** 3.1 corpus and its composition (counts by type); 3.2 reading procedure and read status; 3.3 verification (a short paragraph: every quotation machine-checked against the source text, independent review of the text, no history); 3.4 evidence grading (the strength rule, stated once); 3.5 coverage and limits of the corpus.
4. **Literature review by theme** (method warnings first), one subsection per theme, same layout each time:
   - heading in APA title case that names the topic
   - "In plain words:" one line
   - **Finding and grade** in one sentence, with the sources counted
   - **Evidence** as a captioned table (source, version, sample and period, market and stock group, result, before or after costs, before or after tax) followed by short paragraphs, at most two claims each
   - **Points of disagreement** next to the evidence
   - **Limits** of the evidence in this theme
   - "What the literature establishes:" one line
   Themes: 4.1 Backtest validity and data bias (survivorship, look-ahead, delisting returns, multiple testing); 4.2 Trading costs and execution (spreads, order costs, short-sale costs, rule changes); 4.3 Taxes; 4.4 Short-horizon patterns and daily trading; 4.5 Slower patterns (momentum, 52-week high, earnings drift, published signals after publication); 4.6 Machine learning; 4.7 Evidence on individual investors and funds; 4.8 Summary table (theme, finding, grade, sources). Version 4's 4.2 splits into costs and taxes because it is the densest section.
5. **Discussion:** how the themes fit (the survey's own connection, labeled as interpretation), main disagreements in brief with pointers, gaps in the literature, limits of the survey.
6. **Directions for future research** (replaces "Recommendations"): ordered research steps stated impersonally (a complete dated universe including dead stocks; a test that prices costs and taxes and compares with the index over the same dates; a placebo check; a slow ranking signal on larger stocks; a daily large-stock design; an earnings-event signal; machine learning only after simple signals; prospective out-of-sample evaluation), each with what would count as support and as failure. Items that are the survey's own are labeled "Interpretation", with the evidence they rest on.
7. **Conclusion:** the five research questions answered in order with grades, then a short closing paragraph.

### References and appendices

- **References** in APA 7, unchanged in content.
- **Appendix A** source-by-theme table; **B** evidence tables and long quotations (kept, because the quotations are what a reference reader checks); **C** source index (reference, type, read status, themes; no library ids shown); **D** sources excluded or set aside and why; **E** claims to verify first; **F** literature not obtained or still to find; **G** dated timeline of tax and market rules.
- Removed from the report and moved to `REVIEW_AND_VERSION_LOG.md` beside it: corrections after reviews, details of the gap round and fourth round, the outside research runs and their checks, version history, readability edits, reviewer findings. The visible method text says only what a paper would say about verification.

### Voice rules (stated once, enforced by a script)

- Objective, attributed, past tense for studies, present for what the literature establishes; no "I" or "we"; no "the owner", "the library", "the readers", "our", "round", "revision", "gap round", "V4", "outside run", "Analyst judgment", "recommended path", "paper trade", "Step 1" project labels.
- Replace "the library holds" with "the surveyed corpus contains"; "no passage the readers read" with "no surveyed passage".
- Define each term on first use in each top-level section; link to the glossary.
- Paragraphs of at most two claims; bold only for the grade and the "Finding" label; bullets for lists, tables for numbers; each table has a numbered caption and a note on version and units.
- Quote sparingly in the body; long quotations in Appendix B.
- Draft versus journal version stated at first use of every draft figure; absence statements scoped.

### Mapping from version 4

Each version 4 section is assigned to a new section; process paragraphs go to the log; "round" additions are merged into the theme that holds their subject; the evidence in version 4's Appendix B blocks is reorganized to follow the new theme numbers. A script lists every version 4 number, quotation, citation id and label and reports where it landed (new text, log, or dropped on purpose with a reason).

### Checks and review

1. `verify_report.py`: 0 problems. 2. New preservation check (numbers, citation ids, quotations, labels) against version 4: nothing unaccounted. 3. New voice check: a forbidden-term and structure scan (listed above), run on the whole file including appendices. 4. Anchor and link check. 5. Cold review by a different model family (Codex Sol) on the new text against the sources and against the style rules, plus a same-family review labeled as less independent. 6. Findings are verified against the sources before fixing.

### The skill update

Changes to `~/.claude/skills/literature-survey-report/`:
- `SKILL.md`: new outline (abstract, summary of findings, contents; sections 1 to 7 as above; appendices as reference only); no reader's guide; "Directions for future research"; a paper voice that forbids project vocabulary; theme template; process history goes to a log file, never into the report; honest corpus description; findings by theme not by question; lessons from this pass.
- `references/style_and_citation_brief.md`: rewritten to the new outline, the voice rules and the APA 7 rules below, with the theme template and a short list of forbidden words; the ggTrader specifics (taxable account, no paid data) move to a "project notes" block that governs scope wording only.
- New scripts: `check_academic_voice.py` (forbidden terms, missing sections, bold and bullet counts, paragraph length) and `check_apa.py` (citation form, number style, headings, tables, reference entries, contractions) and `check_preservation.py` (old report versus new); `assemble.py` updated for the new outline and anchors.
- Update `objective-report` only if it shares the reader's-guide wording (to be checked).

### Size and effort

A project. Estimated work: references data reuse, about 8 theme and framing writers (Claude Opus), assembly and checks, two cold reviews (Codex Sol and a labeled same-family review), one fix pass, skill update. Comparable to the fourth round; the owner will be asked for current Claude quota headroom before the plan is written, and usage is reported after each phase.

## Calls made by the author (the owner can change any)

- "Executive summary" becomes "Summary of findings".
- "Analyst judgment" becomes "Interpretation".
- "Recommendations" becomes "Directions for future research", with the steps written impersonally.
- Appendix B keeps the long quotations; Appendix C drops library ids from view (the ids stay in hidden tags and in a lookup file beside the report).
- The costs-and-taxes theme is split into two themes.

## Left for the builder

Exact table layouts per theme, the caption numbering scheme, how to word the corpus description in 3.1 (it must stay truthful), how to treat the current section 5.2 conflicts in the new layout, and the script details.

## Open questions for the owner

1. Is current Claude quota enough for a project the size of the fourth round (Codex quota is plentiful)?
2. Keep the five research questions as they are (daily picking; holding period; costs and taxes; machine learning; backtest validity)?

## Calls made after a builder read of this draft (the owner can change any)

- **Grades stay as in version 4.** No re-grading. Section 3.4 is rewritten to state the rule exactly as it was applied (two or more independent published sources agree on measured data for STRONG; one published source with consistent support for MODERATE; single source, preprint, simulation or opinion for WEAK; a separate label for "no evidence", the name version 4 uses). A label that cannot be justified by that wording after the rewrite is listed for the owner, not silently changed. The preservation check treats every label as must-survive.
- **Non-corpus material.** The 58 official rule documents become a fourth source category in 3.1 ("regulatory and official texts, cited for the dated rule timeline and rule descriptions, not for performance evidence"). The free-data feasibility test (37 live tests) is presented as "a data-availability check carried out for this survey", labeled as the survey's own check and not literature, with its table in an appendix. The outside research runs, their checks and the single outside-run citation go to the log; the one citation is re-examined and, if kept, labeled as a lead.
- **Verification is disclosed truthfully, in paper voice.** Section 3.3 states that sources were summarized by language-model readers working from each source's text, that every quotation was verified by script against the source text, that the text was reviewed by independent model reviewers and that edits made after the last review were checked by script only. This is longer than "independent review" but it is the accurate statement and it belongs in a method section.
- **Quotations in the body.** Every number the body states is traceable to an exact quotation, either in the evidence-table note or in the linked Appendix B block; statements without numbers may be paraphrased with a citation. `verify_report.py` covers all quotations; the cold review checks the paraphrases.
- **Evidence-table cells** are filled only from the existing report text and the reader notes; an unknown cell reads "not stated". A writer may open the source passage to resolve an ambiguity, but adds no finding.
- **Size.** Body 20,000 to 25,000 words; appendices A to G about 20,000 words in total (Appendix B trimmed of duplicated blocks to about 12,000 words, Appendix C about 5,000); references unchanged. Anything cut from Appendix B is listed in the log with the reason.
- **Writers and reviewers (owner's delegation rule).** Theme writers: Claude Sonnet. Framing and synthesis sections (abstract, summary of findings, discussion, directions for future research, conclusion) and the skill rewrite: Claude Opus. Cold reviews: Codex Sol (cross-family, high effort) and a Claude Opus review labeled less independent. Codex Terra is available for mechanical checks. Orchestrator: this session, which re-runs every check itself.
- **Five research questions stay as they are.** Default plan size depends on Claude quota (open question 1); the phased plan is: (1) outline, references, id mapping and tooling; (2) themes; (3) framing sections and assembly; (4) checks and reviews; (5) fix pass and install; (6) skill update. Usage is reported after each phase.

## Amendment (2026-11-22, owner request): strict APA 7 and plain-language explanations

The owner asked for strict adherence to APA style and objective academic language, with plain-language explanations wherever technical jargon appears. This amendment is part of the design and supersedes any earlier line that conflicts with it.

### APA 7 rules the report follows

- **Headings:** APA levels, in title case, unnumbered (APA 7 does not number headings). The report is cross-referenced by heading name with a working link ("see Trading Costs and Execution") and through the table of contents. The numbered scheme of version 4 (4.3, 4.4) is dropped from the visible text; anchors keep stable identifiers. *Author's call, reversible:* if the owner prefers numbered sections for quick reference, numbers can be restored at the cost of departing from APA.
- **Title block:** title and date; there is no author line because the survey has no named author (stated as a call).
- **Abstract:** one paragraph of 150 to 250 words, then a line "Keywords:" in italics with five to seven terms.
- **In-text citations:** author-date throughout; "&" inside parentheses and "and" in running text; three or more authors cited as "First et al." from the first citation; several citations in one parenthesis alphabetical and separated by semicolons; a direct quotation always carries a locator (p., pp., section, Table, chapter, lecture); "as cited in" for relayed sources; a, b suffixes for same author and year; the survey's own hidden id tag follows each quotation's citation, invisible to the reader.
- **Quotations:** fewer than 40 words in quotation marks in the text; 40 words or more as an indented block quotation without quotation marks (the body rarely needs these; long quotations live in the appendix).
- **Tense and voice:** past tense or present perfect for what studies found ("Chen and Velikov (2023) reported"), present tense for what the literature establishes and for the survey's own statements; no first person; no contractions; no "etc."; bias-free language; serial comma.
- **Numbers:** words for zero through nine, numerals for 10 and above; numerals with units, percentages ("%" with a numeral), statistics, dates and sample sizes; no number at the start of a sentence; consistent decimals; "percentage points" for differences between percentages; statistical symbols in italics where used (for example *t*).
- **Abbreviations:** spelled out at first use with the abbreviation in parentheses, then used consistently; few abbreviations, none that the reader must look up (S&P 500 is spelled out at first use; "bps" replaced by "basis points").
- **Tables:** each table has a bold number line ("Table 3"), an italic title in title case on the next line, the body, and a note beginning "*Note.*" below with source, version and units; tables are cited in the text by number; no vertical rules.
- **Reference list:** APA 7 entries, hanging format as far as markdown allows, alphabetical, sentence-case article titles, italic journal title and volume, DOI or URL as an https link, working papers and preprints marked as such with the series and number, one entry per cited source, every in-text citation matched to an entry and the reverse. Details never invented: an unknown element is omitted, as before.
- **Appendices:** lettered ("Appendix A"), each with a title, in order of first mention in the text; tables inside an appendix are numbered "Table A1", "Table A2".
- **Spelling and style details:** US spelling; "while" for time and "whereas" for contrast; "that" for restrictive clauses; consistent hyphenation (for example "short-term" before a noun only).

### Plain-language explanations for technical terms

- Every technical term is defined in the sentence where it first appears in each major section, in an appositive or a short parenthetical clause in ordinary words: "the bid-ask spread, the gap between the highest price a buyer offers and the lowest price a seller accepts". The glossary repeats the definition and points to the section where the term matters.
- A term used only once may be replaced by the plain phrase. A term used repeatedly is defined once per section and used in its technical form afterward.
- Explanations stay neutral and short (about 25 words or fewer), carry no opinion, and never replace the term's correct name. A number with a unit is paired with its everyday meaning the first time it appears in a section ("4 basis points, or 0.04%, per month").
- The check script flags the first use of each glossary term in each section that lacks a gloss within the same sentence.

### Added checks

`check_apa.py` scans the whole file (appendices included) for: "and" inside parenthetical citations and "&" in narrative citations; "et al." with fewer than three authors; contractions and "etc."; numerals for numbers under 10 outside units; numbers starting a sentence; numbered headings; sentence-case headings; tables without a number line, title or note; references without a year or DOI or URL where the source has one; citations without a reference entry and the reverse. `check_academic_voice.py` flags glossary terms with no gloss at their first use per section. Both are run on every build and added to the skill.

### Effect on size and plan

The APA conversion adds a phase to the plan (a style pass after assembly, run by script first and by Claude Opus on the findings), and removes numbered sections from the visible text. No change to the evidence, grades or the quotation checks. The skill update additionally records the APA 7 rules and the plain-language rule as part of the template.

## Amendment (2026-11-22, owner request): skill update is part of this design, and Astra performs the final review

- **Skill update, in scope and required for sign-off.** The report skill (`~/.claude/skills/literature-survey-report/`) is rewritten to the outline, voice rules, APA 7 rules, plain-language rule, theme template and process-log rule in this document, together with the two new check scripts and an updated `assemble.py`, so the next survey is built to this standard from the start. The `objective-report` skill is read for the same project-report habits (reader's guide, project vocabulary, "Analyst judgment") and brought into line if it shares them. The skill's "lessons" section records what went wrong in this report (process language creeping into findings, round-by-round additions, front-matter bloat) and how the checks prevent it.
- **Final review by Astra.** After the build, the Sol cold review, the Opus review and the fix pass, `gpt-6.2-astra` (Codex, high effort) performs the final review of the finished report: it reads the whole report against this design and the style rules and writes a findings file only; it does not edit the report. The lead verifies each finding against the source text, applies the real ones, reruns every check and installs only after that. Astra's quota is small by the project's own rules, so it is used once, for this final review, and its usage is reported.
- **Order of the phases:** outline, references and tooling; theme writers; framing writers and assembly; scripted checks (quotations, preservation, voice, APA, anchors); Sol review and Opus review; fix pass; style pass; Astra final review; fixes; install with backups; skill update and its own test (rebuild one section from the updated skill brief to confirm the scripts pass).
