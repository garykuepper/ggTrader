# Research

Everything ggTrader has learned from research lives here: one folder per study, with its report, its registered plan (the "brief"), and its raw result files side by side. The code and the live paper trader are elsewhere; this folder is the evidence.

To find a study by question, date, status or nickname (A23, PEAD and so on), start with [RESEARCH_INDEX.md](RESEARCH_INDEX.md). For the roster of everything tried and its verdict, see [RESEARCH_SNAPSHOT.md](RESEARCH_SNAPSHOT.md). The backlog of untried ideas is [WEB_RESEARCH_CANDIDATES.md](WEB_RESEARCH_CANDIDATES.md), and the report format is [TEMPLATE-research-report.md](TEMPLATE-research-report.md).

## Current review

The newest look at the whole record is the 2026-11-04 root-cause investigation. Start with its [reading route](audits/root-cause-2026-11-04/README.md), or go straight to the [consolidated direction review](audits/root-cause-2026-11-04/2026-11-05-direction-reviews-consolidated.md). Phase 2 re-runs from that investigation were not performed.

## The folders

Each folder holds one subfolder per study, with dated file names.

| Folder | Question it answers |
|---|---|
| [allocation/](allocation/) | Does changing exposure, leverage or a portfolio mix help? |
| [audits/](audits/) | Can existing evidence, shared machinery or recorded conclusions be trusted? |
| [stock-signals/](stock-signals/) | Can a stock-selection mechanism work, and is the data available to test it? |
| [paper-system/](paper-system/) | How does the paper system measure, size, account for and operate its book? |
| [programme/](programme/) | Which shared standards and methods govern how research is done? |
| [intraday/](intraday/) | Is an intraday strategy feasible? |
| [crypto/](crypto/) | Is there a crypto-specific mechanism or venue result? |
| [prompts/](prompts/) | Which reusable prompts commission new research? |

## Where a new document goes

Take the first line that matches. A document that touches two categories gets links and index tags, never a second copy.

1. Reusable navigation, roster, idea backlog or report template? The `research/` root. Reusable commissioning text goes to `prompts/`.
2. Companion (brief, amendment, result or review) to a study or investigation that already has a folder? That study's folder, whatever its file type.
3. Mainly checking the reliability of existing evidence or shared machinery, rebuilding baselines, or auditing a deployed construction? `audits/`.
4. Setting a shared experimental standard, or studying research-policy behavior? `programme/`.
5. An intraday-specific experiment or feasibility study? `intraday/`.
6. A crypto-specific mechanism or venue study? `crypto/`.
7. Changing exposure, leverage, index or fund allocation, or a blend of sleeves? `allocation/`.
8. Operating behavior or accounting of the paper system, with no main return hypothesis? `paper-system/`.
9. Selecting individual stocks, screening stock factors, or assessing data for candidates? `stock-signals/`.

The full rule is in section 5 of the [design spec](../docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md).

## What is not here

The `docs/` folder keeps architecture ([architecture.md](../docs/architecture.md)), the CLI reference ([cli_reference.md](../docs/cli_reference.md)), the changelog ([changelog.md](../docs/changelog.md)), the roadmap ([roadmap.md](../docs/roadmap.md)), current decisions ([next_steps.md](../docs/next_steps.md)), the working task list ([working_task_list.md](../docs/working_task_list.md)) and `superpowers/` (specs and plans). Research reports used to live in a `research` subfolder of `docs/`; they no longer do.

## Finding an old path

Reports written before 2026-11-05 cite paths under the old `docs` research folder. [MOVED.csv](MOVED.csv) lists the old path and the new path of every moved file (174 rows), and marks the three briefs that gained a `-brief` ending. It is a lookup table only: it does not make a missing path exist.
