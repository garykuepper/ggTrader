# Trading course site: design

Date: 2026-11-15. Status: draft for the owner's approval. Revised the same day after a
builder check (Claude Opus) and an independent review (Claude Fable).

## 1. Intent and the why

The owner has tested many trading ideas in ggTrader. None has held up against SPY, and they
describe their approach as a "shotgun": trying everything and seeing that nothing works. They
want to step back and understand the fundamentals: where an advantage can come from, how
firms beat the market, how to turn an idea into rules, how to test it without fooling
themselves, and how to tell whether they are applying a method correctly.

The owner wants to day trade and use margin, so that is the main path through the course.
Longer-term strategies appear for comparison.

**Success:** after the course, the owner can look at a trading idea, a backtest or a claim
and say whether it could work, and why. The goal is understanding and judgment. Hands-on
exercises inside ggTrader are not the end goal.

## 2. Goals, non-goals, anti-goals

Goals
- 19 lessons plus a final exercise, read in order, in plain language pitched a little
  above a beginner. Every technical term is defined on first use.
- Each lesson covers what the idea is, why it matters, a step-by-step worked example, an
  interactive chart or calculator where it helps, a few practice checks with answers, and
  a ggTrader case study where one honestly fits.
- A checklist for judging a trading idea (lesson 16), used again in the final exercise.
  Its length is whatever the content needs. It is not promised to fit on one page.
- **Facts are stated plainly, with sources (owner decision, 2026-11-15).** When the evidence
  says something unwelcome, for example published studies of individual day traders, the
  lesson states it as a finding, with its source and date. The course never tells the
  owner what to do with their own trading. That decision stays theirs.

Non-goals
- Not a signal service or a trading system. It does not place orders or connect to Alpaca.
- No live market data feeds. Charts use made-up or historical numbers, and each is labeled.
- No changes to ggTrader's code, the paper trader or its research.

Anti-goals (would make it a failure even if it technically works)
- Shading results positive, or leaving the owner thinking a missing trick will make day
  trading pay. ggTrader's NO-GO findings and outside evidence are taught as they are.
- Telling the owner to stop trading, retire the stock sleeve, shut down paper trading or
  the project, or open an IRA or other retirement account (owner instructions in
  ggTrader's `CLAUDE.md`). Stating evidence is allowed. Making those decisions is not.
- Presenting ggTrader's daily-bar results as evidence about day trading. They are not.
- Jargon without definitions, or numbers with no stated source.
- Made-up numbers presented as real results.

## 3. Constraints

- **Built on ggPackMind's study site** (`/home/flynn/ggPackMind/study/`): `topics.toml`,
  one Markdown file per topic in `content/`, directives (`<!--calc-->`, `<!--drills-->`,
  `<!--chart:NAME-->`, `<!--diagram:NAME-->`), one JS module per topic, shared runtime
  (`calc.js`, `chart.js`, `drill.js`, `diagram.js`), a formula module mirrored by a Python
  reference and checked by tests, a build that refuses to write on errors, phone width
  and offline use, and progress kept in the browser. Read `ggPackMind/AGENTS.md` (Study
  section) for the full rules. Section 6 lists which of its checks carry over.
- **Its own site.** A new folder and git repo, `/home/flynn/ggTraderLearn`. Copy the
  engine. Do not import from ggPackMind or edit it.
- **Private, home network only.** Served like ggPackMind: a user systemd service running
  a static server bound to 127.0.0.1, published with Tailscale Serve on a new tailnet-only
  port. Never public.
- **Theme:** the owner's house theme (`gg-theme` skill). Charts follow the `dataviz` skill.
- **Sources:** free sources only (the owner refuses paid data). Prefer the local library
  (`gg-library`, subjects `Trading` and `Mathematics-Statistics`) with source IDs and
  locators. When the library lacks a key paper or a current rule, cite a free web source
  (paper page, FINRA, SEC, IRS, broker page) with its URL and an "as of" date. The
  library's archived regulatory copies are marked "currency unverified", so they cannot
  back a current rule on their own. Do not quote long passages from copyrighted books.
- **Rules that change** (margin, the day-trading account minimum, short-sale rules, tax
  rules, regulatory fees) are verified when their lesson is written, each with an "as of"
  date and a source. The day-trading minimum may already have changed in 2026: lesson 6
  states the rule in force on the writing date, any replacement, and Alpaca's
  implementation.
- **Taxes:** in a taxable account only. No retirement-account comparisons.
- **Numbers:** every number is one of (a) from a cited source, (b) from a ggTrader research
  report or its results JSON, quoted with its evaluation window, or (c) made up for
  teaching and labeled "illustrative".
- **Links:** no links to ggTrader (the repo or its reports). External links (Wikipedia,
  sources, rule pages) are allowed, open in a new tab, and simply do not work offline.

## 4. Approaches considered

1. **New site based on ggPackMind's study site (chosen).** It keeps the format the owner
   already knows and keeps the subjects separate. It costs more than option 2 because the
   engine is copied and adapted.
2. **Add trading topics to ggPackMind.** Cheapest, but it mixes two unrelated subjects in a
   site tied to the thermal-engineering course.
3. **Private claude.ai pages, one per lesson.** Fastest to start, but it loses the course
   structure, progress tracking and offline use, and 20 pages are hard to keep consistent.

The owner picked option 1.

## 5. What the owner decided

- Main path: day trading and margin. Long-term strategies for comparison.
- End goal: understanding and judgment.
- Approach: option 1.
- State facts plainly, with sources, never as instructions.
- Lesson plan accepted 2026-11-15 and revised after review (below). The testing-trap
  lesson (13) is the centerpiece.

### Lesson plan

Part 1: Where returns come from
1. What "beating SPY" really means: market return compared with skill, the risk taken,
   and the result after costs and taxes. States up front that ggTrader's evidence covers
   holding periods of a day or longer, and that ggTrader has no day-trading evidence (its
   intraday work stopped at a feasibility review with no data pulled).
2. Where an advantage can come from (taking risk others avoid, others' behavior, providing
   liquidity, information or speed), and who is on the other side of your trade.
3. How firms make money (market makers, high-speed traders, quant funds, long-term stock
   pickers), what they have that an individual does not, and which of those advantages an
   individual can use. Includes the published evidence on individual day traders' results,
   stated plainly with sources.

Part 2: How day trading works in practice
4. Orders, the order book, spreads and fills.
5. Costs: spread, slippage, market impact and fees, and what each trade must earn just to
   break even.
6. Margin: the initial requirement (Reg T), the maintenance requirement (FINRA), broker
   house rules, intraday compared with overnight buying power, the day-trading account
   minimum (verified), interest, why borrowed money makes losses deeper, and the decay of
   leveraged ETFs held over time (relevant because the owner holds SSO).
7. Short selling: locating shares, borrow cost, and losses with no upper limit.
8. Taxes in a taxable account: short-term rates, wash sales, the $3,000 yearly limit on
   deducting net losses, the 3.8% net investment income tax, the limit on deducting margin
   interest, trader tax status and the mark-to-market election, and state tax. All
   verified with an "as of" date.

Part 3: Turning an idea into rules
9. An indicator is not an advantage: signal, strategy and edge.
10. Expected profit per trade, win rate compared with the size of wins, and position sizing
    (fixed fraction, the Kelly rule, and why traders use a fraction of it).

Part 4: Testing without fooling yourself
11. Using information you could not have had at the time, delisted stocks, and bad data.
    Case study: ggTrader's one-day date shift in the price data.
12. Overfitting inside one idea: tuning settings until the past looks good.
13. **Centerpiece. Two traps, both of them yours.**
    - *False winners:* testing many ideas guarantees some that look good by luck. Case
      study: the ensemble's 1.12 Sharpe, which fell to 0.59 after a leak and a stock-list
      error were fixed.
    - *Too little power:* about five years of daily data can only detect a large edge, so
      "no evidence of an edge" is not "no edge". Case study: the root-cause audit's 72
      measured tests, none reliable, none showing an edge, none ruling out 1 to 4
      percentage points a year.
    - Base rates (few ideas are true beforehand) and the arithmetic of active management
      after costs.
    - Why a count of failed ideas is not a count of independent tests (the 38 registered
      strategy names include voters, variants and shared machinery). How to estimate the
      effective number of independent tests, and corrections (Holm, deflated Sharpe) using
      the factor screen's real thresholds.
    - Every count is pinned to a named report and its ggTrader commit.
14. Walk-forward testing and holdouts: what they do, ggTrader's finding that its
    walk-forward results could mainly measure the fallback, and how a holdout gets used up.
15. What a fair test of an intraday idea requires (quote-level data, modeling fills and
    queue position), and what free data cannot support.
16. How to read a backtest report: the checklist.

Part 5: From a test to real money
17. What paper trading proves and what it does not.
18. Losing streaks, drawdowns and the risk of ruin, for any strategy. Illustrative inputs
    only.
19. Long-term strategies for comparison: momentum, factor investing, buy-and-hold SPY, and
    how the hurdle changes with holding period.

Final exercise: judge three trading ideas with the checklist. One of them may be the
owner's own day-trading idea if they bring one. Otherwise all three are made up.

Every case study carries a label with its holding period (intraday, daily, monthly) and
its evaluation window.

## 6. Calls made after review (Claude's calls, owner to confirm)

- **Citations:** the site keeps its own `sources.toml`, a snapshot of the sources it cites
  (ID, title, author, year, locator or URL, "as of" date for rules). The build refuses an
  unknown ID. A citation opens a small source card with at most a short quote and no
  files, because the books are copyrighted.
- **ggTrader case studies:** no build-time reads from the ggTrader repo. Each case study
  quotes the report's name, date and evaluation window. Its numbers go into a data file
  with the ggTrader commit they came from, and a flag for numbers produced by the legacy
  engine (before 2026-11-06).
- **Broker and tax:** general federal rules, with Alpaca's published margin terms as the
  worked example. All capital, expense and tax-bracket figures are illustrative and
  labeled. The owner's real account, bracket or spending never appear.
- **Voice:** plain words with terms defined on first use (the owner's rule for owner-facing
  writing), not ggPackMind's first-person interview voice.
- **ggPackMind checks, kept / replaced / dropped:**
  - Kept: the build refuses to write on unbalanced math, a missing module, an unknown
    source ID or a broken internal link; the `**How to read it.**` block after every chart;
    chart palette and direct-label rules; Wikipedia links on first use; the formula module
    mirrored by a Python reference, with tests comparing every drill answer, calculator
    output and chart point; the SVG diagram lint; the "What this is and why you care"
    opening; the worked-example structure (adapted from physics givens to trade inputs).
  - Replaced: `papers.toml` and `[P2]`-style IDs become `sources.toml`; `physics.js` and
    `ref_physics.py` become a trading-math module and its Python reference (break-even per
    trade, margin and interest, expectancy, sizing, compounding of losses, leveraged-ETF
    decay, multiple-testing odds, test power, seeded random numbers that match in JS and
    Python); `units.js` unit codes become simple formats (dollars, percent, count, days);
    the no-em-dash voice lint becomes ggTrader's `scripts/check_reading_edition.py` rules,
    copied in.
  - Dropped: the 737 thread and `data737.js`, the uncited-737 lint, the SI-first and
    kelvin lints, and the Boeing provenance rule.
- **First slice:** the engine plus lessons 5 and 13, then a hard stop for the owner's review
  before the rest is written. The delegation rules in ggTrader's `CLAUDE.md` apply, and
  quota headroom is asked for when the plan is written.

## 7. Left to the builder

- The port, service name and folder layout inside `/home/flynn/ggTraderLearn`, following
  ggPackMind's pattern.
- Which lessons get a calculator, a live chart or a diagram. The default is one interactive
  piece per lesson where it teaches something reading cannot.
- Which ggTrader report backs each case study, starting from
  `research/RESEARCH_SNAPSHOT.md`, the root-cause audit and the research-process audit.
- Which sources back each lesson, found with `gg-library` searches first, then free web
  sources.
- Delivery order after the first slice.
