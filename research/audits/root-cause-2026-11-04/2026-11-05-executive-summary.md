# Executive summary: why the research has not doubled SPY

**Date:** 2026-11-05. **One-page summary of the [plain-English findings](findings/2026-11-05-findings-plain-english.md) and the [direction review](reviews/2026-11-05-direction-reviews-consolidated.md). The originals stay as the record.** Terms: [reference sheet](reference-sheet.md).

SPY is the fund that tracks the S&P 500. 2x means twice SPY's total return (price change plus reinvested dividends). pp/yr means percentage points a year.

> **Bottom line**
>
> - The question: why has the research not come close to 2x?
> - The answer: the built design shows no 2x-sized effect, and its test could have seen one (the PARA price fault, a stock stored with another security's prices, is unmeasured). Stock picking itself is undecided.
> - What you decide: whether to accept the synthesis. All 13 decisions have defaults, and under them no re-run runs.

## What we found

- **Built design.** The replica (live rules re-run on history) returned 14.19% a year against SPY's 14.83% over 2021-2026, or -0.64 pp/yr (t -0.69, which cannot be told from zero).
- **Power.** The replica could detect about 4.2 pp/yr at whole-account level, if the PARA price fault proves immaterial.
- **Stock picking.** Of 72 measured tests, 0 are reliable, 11 unreliable, 58 inconclusive (could not decide) and 3 stopped on their own terms. None shows an edge, and none rules out 1 to 4 pp/yr.
- **Ceiling.** Doubling needs 9.57 pp/yr over 2021-2026, 3.54 over 2006-2026 and 0.77 over 1927-2026. Only 2x leverage (+4.15 over 2006-2026, worst fall -84.8%) and momentum before 1994 reached their window's gap. Leverage is exposure, not skill.
- **Momentum, 2021-2026.** +4.37 pp/yr (tested 2022-2026), 95% interval -7.07 to +16.05, so it cannot be told from zero or the full target.
- **Paper.** Since 2026-06-24 the paper account trailed SPY in total (not per year) by 3.76 pp at run-time broker value and 4.57 on end-of-day equity.
- **Tax.** A book that sells within a year needs about +2.42 pp/yr before tax at 24%/15% (short/long-term) brackets to tie SPY after tax over 2021-2026 (stylised model).

![Gap that doubling SPY needs in four windows, beside the best result](figures/gap-vs-measured.png)

*Source: the Ceiling check. Points a year beyond SPY or the French market index, so benchmarks differ by row. Leverage is exposure, not skill, and holds only at the window's funding rate.*

## Why the target was missed

Ranked by judgement, no share measured:

1. The target is large next to every measured stock-picking effect.
2. The book is about 85% SPY, so it cannot stray far. Arithmetic takes no feedback.
3. Early "beat SPY" results came from leaks (information a test should not have had). They explain false optimism, not the shortfall.
4. The tests could not tell a small stock-picking edge from zero.
5. The paper shortfall is mostly idle cash and broker quirks (estimated split).
6. The search stayed narrow: large-cap, long-only, mostly one signal family.

## What the two models recommend

**Opus** (one reviewer):

- The built design cannot reach 2x, and no affordable free-data test can decide a 1 to 4 pp/yr edge.
- Order: paper-trader trust first, then cheap re-runs that write nothing to the price table, then a small pre-registered (pass bar written down first) long-history screen budget. Best case about +1 pp/yr (unverified judgement).
- Leverage is the owner's risk decision, dollar drawdown tolerance first.

**Argon** (Gemini model, unverified as written):

- 43 of 105 citations are broken or unsupported. Numbers tracing to a check report hold. Dates, item numbers, hashes and payoff odds do not.
- Survived: leverage as the owner's risk decision, the long-hold tax direction, and parking crypto, small caps, long-short and options (as judgement).
- Corrected: re-run item numbers, trader-patch payoff and dates. Dropped: the concentrated sleeve's payoff and odds.

**Agree:** the built design cannot reach 2x, and no-database re-runs come first. **Differ:** stock-picking prospects, and patching versus measuring the trader. The checked evidence sits closer to Opus.

**Three options**, in Opus's order. You decide which, if any, to run:

- Option 1, paper-trader measurement and parity. Stop gate: "parity measured and archived".
- Option 2, funding-honest leverage figures for the 2027-09-30 review of A23 (70% SPY / 30% SSO, a fund targeting twice SPY's daily move).
- Option 3, one pre-registered long-history stock screen. Stop gate: a long-history-test fail closes it.

![Three option columns: question, cheapest step, stop gate](figures/options-at-a-glance.png)

*Source: the consolidated review.*

## What you decide

Five of the 13. Each default holds until you answer.

| # | Decision | Default |
|---|---|---|
| 1 | Accept the synthesis | Open, nothing changes |
| 3 | Run any Optional Re-run List item | None run |
| 4 | A separate research copy of the price table | The no-writes rule stays |
| 9 | Funding-honest leverage figures before 2027-09-30 | Review on its date with existing figures |
| 10 | One benchmark everywhere | Not adopted |

## What this cannot tell you

- **Unmeasured and unrun.** No cause's share of the shortfall was measured. None of the 29 Optional Re-run List items was run, so their questions stay "unknown, re-run not run".
- **Price history.** On 2026-11-04 an agent ran a script only to read its help text, and it rewrote prices for 19 symbols.
- **Who checked.** Argon is unverified. Both models read one synthesis, so their agreement is not independent evidence.

## Read next

| Document | What it is | When to read it |
|---|---|---|
| [Findings in plain English](findings/2026-11-05-findings-plain-english.md) | The synthesis, terms defined, caveats kept | For the evidence behind any bullet here |
| [Consolidated direction review](reviews/2026-11-05-direction-reviews-consolidated.md) | What Opus and Argon recommend, with the 13 decisions | Before you answer a decision |
| [Reference sheet](reference-sheet.md) | Every term and check name, with its file | When a word stops you |
