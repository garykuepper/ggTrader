# SPY last-half-hour rule, 2016 to 2026: historical screen result

**Date:** 2026-11-15
**Study:** research/intraday/spy-momentum-feasibility, protocol r9
([approved protocol](2026-11-15-intraday-feasibility-r9.md),
[label amendment](2026-11-15-intraday-r9-amendment-labels.md))
**Status:** Single scheduled score, done. This report gives evidence, whether the preset
thresholds were met, and the uncertainty. It gives no GO/NO-GO label (the owner retired new
ones) and makes no trading recommendation.

## 1. Summary

- **What was tested.** One fixed day-trading rule on SPY, from the published intraday-momentum
  idea (Gao, Han, Li and Zhou, 2018):
  - If SPY is up from the prior close at 10:00 ET, buy at 15:30 and sell at the close.
  - Otherwise do nothing that day.
  - The rule has no settings, so nothing was tuned.
- **Data.** Alpaca's free minute bars (SIP feed, raw prices) for every trading day from
  2016-01-05 to 2026-09-30: 2,677 scored days. Two days with missing bars were left out under a
  list fixed before any price was seen.
- **Result.** Neither preset threshold was met. The average day lost money after costs, and the
  loss is statistically clear (t = −3.48).
- **Why it lost.**
  - About three quarters of the loss is trading costs.
  - The rest is that SPY's last half hour fell slightly on average in this decade.
  - Being up at 10:00 told nothing about the last half hour: the measured link is about zero
    (t = −0.31).
- **Scope.** This is ggTrader's first test of a true day trade on market data. It says nothing
  about 1993–2013 (the original paper's sample) or about other intraday ideas.

## 2. Results

### Preset thresholds

| Threshold | Registered bar | Measured | Met? |
|---|---|---|---|
| A1: one-sided Newey-West t (5 lags) on the average daily result after base costs | 3.0 or more | −3.48 | No |
| A2: yearly growth after r8's illustrative 24% tax ledger | +3.0% a year or more | −3.68% a year | No |

Newey-West is a significance test that allows for patterns repeating over time. Base cost is 1
basis point (0.01%) per side. The tax rate is an example from the protocol, not the owner's
bracket.

### Where the result came from

| Measure | Value |
|---|---|
| Scored days | 2,677 |
| Days traded (SPY up at 10:00) | 1,479 (55%) |
| Days out (down: 1,193; unchanged: 5) | 1,198 |
| Average result per day, after costs (days out count as zero) | −1.46 bp |
| Of which trading costs (2 bp a round trip × 55% of days) | −1.10 bp |
| Of which the last half hour on traded days, before costs | −0.35 bp |
| Average result per traded day, after costs | −2.64 bp |
| Cost per side at which the rule breaks even | −0.32 bp (it loses even with free trading) |
| Buying the last half hour every day, before costs | about −0.47 bp a day |
| Link between the 10:00 sign and the last half hour (slope, t) | −0.38 bp, t = −0.31 |

### Over time

| Period | Days | Traded | Average, bp a day | t | Traded days won |
|---|---|---|---|---|---|
| 2016–2019 | 994 | 535 | −1.65 | −3.57 | 44.7% |
| 2020–2023 | 1,000 | 552 | −1.52 | −1.61 | 41.3% |
| 2024–2026-09 | 683 | 392 | −1.09 | −1.90 | 42.9% |

Yearly averages were negative in every year except 2020 (+1.11 bp a day).

### Risk and the SPY comparison

| Measure | Value |
|---|---|
| Deepest drop from a peak (before tax) | −33.0% |
| Longest losing streak | 19 traded days |
| Worst month | October 2020, −2.6% |
| Rule, yearly growth after the tax example | −3.7% |
| Rule, with doubled costs (2 bp a side) | −6.2% a year, t = −6.07 |
| SPY held with dividends reinvested, before tax, same window | +15.1% a year |
| $10,000 over 10.7 years (illustrative): rule after tax / SPY before tax | about $6,690 / $45,140 |

The SPY comparison is a required side check, not a threshold. SPY is untaxed in it, so it
flatters SPY slightly. That cannot close a gap of 18.7 percentage points a year.

## 3. How much to trust it

- **The negative average holds across every variation checked.**
  - Newey-West t with 0, 10 and 20 lags: −3.79, −3.16 and −2.81.
  - Leaving dividend ex-dates out: t = −3.40.
  - Doubled costs: t = −6.07.
- **The test favored the rule.**
  - Trades fill at exact minute-bar prices with 1 bp a side.
  - The 15:59 bar close used as the exit is typically within 0.8 bp of the official close
    (95% of days within 3.7 bp; the widest gap was 280 bp).
  - Real orders would cost at least this much.
- **Data quality.**
  - The download passed every check before any price was used.
  - 2,701 calendar sessions, all 132 data chunks complete, no defects beyond the two
    registered days.
  - Alpaca's 41 dividends match State Street's at 6 decimals. The two Alpaca lacks
    (2016-03-18 and 2018-06-15) came from the issuer file.
- **The two excluded days** (2019-08-12 and 2019-08-13) are 0.07% of the sample and cannot
  account for the result. Their effect cannot be measured without prices, so they remain a
  named limitation.
- **What this does not show.**
  - Whether the effect existed in the original sample.
  - Anything about other intraday rules, other symbols, or futures.
  - That no such effect exists anywhere: one fixed rule failing is not that proof.

## 4. Earlier history

- **r8 (2026-10-25)** approved this same rule and its thresholds. It stopped as UNRESOLVED
  (data) for two reasons: two missing bars on 2019-08-12, and two dividends missing from
  Alpaca. The October gap investigation advised waiting for Alpaca to explain the gaps.
- **r9 (2026-11-15).** The owner judged that Alpaca would not explain them. A fresh count
  found the gaps tiny (104 of 1,062,090 minute bars), so the owner approved r9. r9 adds a
  closed two-day exclusion and the issuer dividend file, and keeps everything else from r8.
- **The October data pull** was made by an agent without approval and was not reused. The
  r9 data were downloaded fresh.

## 5. What follows from the protocol

- **No re-score, no re-tuning.** The protocol allowed one score. Any variant of this rule
  (other times, filters, symbols or costs) would be a new registered question, with this
  result counted as one test already spent.
- **The forward observer is separate.** It has recorded this rule on live data since
  2026-10-27 and is unaffected by this report. Its first checkpoint, on 2026-11-30, reviews how
  reliably it runs.
- **Action decisions** belong to the owner and are outside this report.

## 6. Open questions and parked ideas

- **Did the effect fade after publication?** Answering would need minute data before 2016.
  Alpaca's free archive starts in 2016, and the owner does not buy data, so this is parked.
- **Is the last half hour a cost problem or a signal problem?** Here it is both: the rule
  lost before costs, and costs were three quarters of the loss. Any future intraday idea on
  SPY has to clear about 2 bp a round trip on every trade before it can show anything.
- **Evidence files**
  - Results: `_intraday_momentum_r9_results.json` and `_intraday_momentum_r9_results.md`.
    These write-once files print "NO-GO"; read that as "neither threshold met".
  - Validation: `_intraday_momentum_r9_validation.json`.
  - Vendor documentation review: `_intraday_r9_vendor_adjustment_review.json`.
  - Gap count: `../artifacts-2026-11-15-spy-minute-gap-count/`.
  - Prices: `data/external/intraday_momentum_r9/`, not tracked by git.
- **Fingerprints (SHA-256)**
  - Protocol `222521ce…`
  - Code `ebef81e1…`
  - Data manifest `fec4cce7…`
  - Validation `22b696ad…`
