# Research

Everything ggTrader has learned from research lives here: one folder per study, with its report, its registered plan (the "brief"), and its raw result files side by side. The engine and the live paper trader are elsewhere (in `src/`); this folder is the evidence, and each study's own code sits next to it.

To find a study by question, date, status or nickname (A23 = the 70/30 SPY/SSO allocation; PEAD = post-earnings-announcement drift), start with [RESEARCH_INDEX.md](RESEARCH_INDEX.md). For the roster of everything tried and its verdict, see [RESEARCH_SNAPSHOT.md](RESEARCH_SNAPSHOT.md). The backlog of untried ideas is [WEB_RESEARCH_CANDIDATES.md](WEB_RESEARCH_CANDIDATES.md), and the report format is [TEMPLATE-research-report.md](TEMPLATE-research-report.md).

## Current review

The newest look at the whole record is the 2026-11-04 root-cause investigation. Start with its [reading route](audits/root-cause-2026-11-04/README.md), or read the one-page [executive summary](audits/root-cause-2026-11-04/2026-11-05-executive-summary.md) first, then the [findings in plain English](audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md) and the [consolidated direction review](audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md), with the [reference sheet](audits/root-cause-2026-11-04/reference-sheet.md) open for the terms. The 29 optional re-checks (called Phase 2) proposed at the end of the first phase were deliberately not run.

## The folders

Each folder holds one subfolder per study, except `prompts/`, which holds the two prompt files directly. File names are dated. A study folder with more than about 30 files may group them into sub-folders named for what they are; only [audits/root-cause-2026-11-04/](audits/root-cause-2026-11-04/) does.

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

`intraday/` holds four folders: [spy-momentum-feasibility/](intraday/spy-momentum-feasibility/) (the feasibility study, the r8 protocol and its data check), [spy-momentum-prospective-observer/](intraday/spy-momentum-prospective-observer/) (the forward practice journal), and two supporting packages, [artifacts-2026-10-25-intraday-gap/](intraday/artifacts-2026-10-25-intraday-gap/) and [artifacts-2026-10-26-intraday-readiness/](intraday/artifacts-2026-10-26-intraday-readiness/). Their questions and status are in [RESEARCH_INDEX.md](RESEARCH_INDEX.md).

A study's code lives in its own folder, not in `src/`: `code/` holds the driver (`run.py`) and the study's modules, `code/tests/` holds its tests, and a category's `_lib/` folder (for example `intraday/_lib/`) holds helpers that several studies of that category share. Create a study with `scripts/study.py new` and check it with `scripts/study.py check`. [CODEMOVED.csv](CODEMOVED.csv) sits next to [MOVED.csv](MOVED.csv) and records every file that moved out of `src/`, `scripts/` or `tests/` into a study, with its hash before and after. Reports written earlier cite the old code paths; look them up there.

## Where a new document goes

Take the first line that matches. A document that touches two categories gets links and index tags, never a second copy.

| Order | Test | Home |
|---|---|---|
| 1 | Reusable navigation, roster, idea backlog or report template? | `research/` root; reusable commissioning text goes to `prompts/` |
| 2 | Companion (brief, amendment, result, review) to an already identified study or investigation? | That study's folder, whatever its kind, asset or file type |
| 3 | Mainly checking the reliability of existing evidence or shared machinery across studies, reconstructing baselines, or auditing a deployed construction? | `audits/` |
| 4 | Establishing a shared experimental standard, or studying research-policy behavior? | `programme/` |
| 5 | An intraday-specific experiment or feasibility study? | `intraday/` |
| 6 | A crypto-specific mechanism or venue study? | `crypto/` |
| 7 | Changing exposure, leverage, index/ETF allocation or a blend of sleeves? | `allocation/` |
| 8 | Operating behavior or accounting of the paper system, without a primary return hypothesis? | `paper-system/` |
| 9 | Selecting individual stocks, screening stock factors, or assessing candidate data availability? | `stock-signals/` |

This list is copied from section 5 of the design spec (`docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md`), which governs if they ever differ.

## What is not here

The `docs/` folder keeps architecture ([architecture.md](../docs/architecture.md)), the CLI reference ([cli_reference.md](../docs/cli_reference.md)), the changelog ([changelog.md](../docs/changelog.md)), the roadmap ([roadmap.md](../docs/roadmap.md)), current decisions ([next_steps.md](../docs/next_steps.md)), the working task list ([working_task_list.md](../docs/working_task_list.md)) and `superpowers/` (specs and plans). Research reports used to live in a `research` subfolder of `docs/`; they no longer do. Superseded point-in-time documents go to `docs/archive/`.

## Finding an old path

Reports written before 2026-11-05 cite paths under the old `docs` research folder. [MOVED.csv](MOVED.csv) lists the old path and the new path of every moved file (248 rows: the 174 moved by the first reorganization plus the 74 intraday files moved when the intraday branch was brought onto this layout), and marks the three briefs that gained a `-brief` ending. It is a lookup table only: it does not make a missing path exist.
