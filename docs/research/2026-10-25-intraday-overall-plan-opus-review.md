# Independent review of the intraday research plan (head `8fbea0a`)

I made no edits, network calls or credential reads. I inspected the probe artifact programmatically and looked only at its metadata fields: dates, timestamps, venue codes and grids.

## Verdict

**Proceed, but not in the proposed order, and not with another probe next.** The evidence justifies one cheap, frozen, single-shot test of the long-only Gao variant. It does not yet justify more staged data work, for two reasons:

- **The downstream decision chain doesn't fit together.** A pass here would lead into a forward standard that this rule is very unlikely to meet as written.
- **The protocol is still unfrozen**, and the proposed order looks at prices (0B) before freezing it.

Stage 0A did its job. The "unresolved" start input is not a real blocker, and the auction discrepancy may not matter. Freeze first, then pull and validate the whole window in one pass, then score once.

The prior is weak, and the plan should be sized to match it. The original effect was measured in-sample on 1993–2013 data with long and short trades. The plan acknowledges published post-publication decay (Rosa), and a near-identical 2014–2026 replication reported net losses. The long-only edge has to clear 2 bp per round trip.

## Findings, most important first

**F1 — Critical: a pass on A2 leads into a forward stage that is very unlikely to pass.**
- **What A2 measures.** A2 is a standalone hurdle: +3% CAGR after the 24% tax haircut, with idle cash earning zero (`briefs/2026-10-24-intraday-feasibility.md:92`).
- **Where a pass leads.** The brief sends a pass to a shadow "under the approved forward-paper standard" (`:117`).
- **What that standard requires.** At least +1.0 pt/yr after tax *over SPY*, a stop if cumulative relative wealth falls below −15% at any quarter-end, and T-bill cash (`briefs/2026-10-23-forward-paper-standard.md:139-152`, `:101`).
- **Why that's a mismatch.** A sleeve holding SPY 30 minutes on about half of sessions gives up the equity premium SPY earns the rest of the time. A sleeve that just clears A2 would trail SPY by roughly the gap between the two CAGRs. In most rising markets that likely hits the −15% stop in under two years.
- **The standard also conflicts with the brief mechanically.** It requires decisions committed before the session opens, fills at the next session's close, and no broker orders (`:20-21`, `:99-103`). The brief instead plans "intraday orders in the Alpaca paper account."
- **The standard's own status is inconsistent.** Line 3 says "approved"; line 40 says "draft".

This needs an owner decision before anything else (see D4). It is not an argument against testing the idea.

**F2 — High: the plan has no power or minimum-detectable-effect check.**
- **Illustrative arithmetic, not a measured result.** I assumed a daily standard deviation of about 25 bp for the whole-calendar strategy series (flat days count as zero) and about 2,670 eligible sessions.
  - A1's t ≥ 3 then needs a mean of about 1.5 bp/day net.
  - A2's 3% after a 24% tax needs about 3.9%/yr before tax, also about 1.5 bp/day.
  - Trading about half the days, that is roughly 3 bp net per trade, or about 5 bp gross per trade at base cost.
- **Comparison.** The paper's in-sample long/short mean is 6.67%/yr, about 2.6 bp/day across all days (`2026-10-24-intraday-trading-feasibility.md:24`).
- **What to do.** Before approving A1/A2, compute the minimum detectable effect from published or pre-2016 inputs only (for example the paper's last-half-hour volatility). The owner then approves the thresholds knowing the likely pass rate, and nobody is tempted to adjust them after seeing data. I'm not proposing new thresholds.

**F3 — High: the order is wrong; freeze before any price is seen.**
- In the proposed sequence, 0B (prices on 20 dates) comes before step (4), the freeze.
- 0B prices would show the morning signal and the afternoon return on 20 sessions. That is small leakage, but avoidable.
- Every remaining choice can be made blind:
  - window start
  - auction vs 15:59 close, which the brief already says is chosen "by availability only" (`brief:58,62`)
  - A1 meaning
  - A2
  - data-quality scope
- Freeze them, hash the scoring specification, then touch prices.

**F4 — Medium-high: the missing 2015-12-31 prior close does not block anything.**
- The brief already pre-registered a fallback: "If it is unavailable, the first test session is the next day" (`brief:69`).
- The first selected sample date is **2016-01-04**, and its SIP session has a complete 390-minute grid (artifact `manifest.selected_dates[0]`; coverage report `:56-58`). So the input needed for a 2016-01-05 start is already evidenced by timestamps.
- The 2014 and 2015 sentinels were empty on both feeds, matching the documented "since 2016". That makes the "earlier 2014 option" (`brief:139`) effectively moot.
- **No supplemental probe is needed**, only an owner decision. The 0A result stays "unresolved" as recorded (`contract:28`). The decision applies the fallback; it does not relabel 0A as a pass.

**F5 — Medium: the auction schema error is probably our own parser; the real auction problem is a different one.**
- **Likely cause (unverified).** `src/ggTrader/lab/intraday_probe.py:152-154` raises a schema error unless *both* the `o` (opening) and `c` (closing) lists are present. A 15:59–16:01 request window probably returns no opening list. The response body was discarded, so this can't be confirmed, but it is not a reason to run another request.
- **The real problem.** In the artifact, the 2016–2021-01 samples have one closing auction record per day from venue `P`. From **2021-08-02 onward, every sample has two `P` closing records**. Stage 0A deliberately discarded condition codes, so nothing in hand can pick the official primary-listing close.
- **Consequence.** If the auction branch is kept, the freeze needs a documented rule for choosing by condition code, chosen without looking at prices. The bulk pull must then keep condition codes.
- **Encouraging note.** Full-session auction requests worked on all 20 samples.

**F6 — Medium: decide the executable exit before deciding the data work.**
- Market-on-close orders can't be fractional (`brief:67`).
- The 15:59-bar close is not an executable price either: it is the last trade before 16:00, which no order can target.
- The exit choice decides whether auction data matter at all:
  - **Whole-share market-on-close exit:** auction close is primary, and F5's condition-code rule is required.
  - **Otherwise:** the auction branch becomes a diagnostic, and the 0A discrepancy is irrelevant.
- **Simplification found.** The live rule needs no real-time data. The 10:00 signal is needed only by 15:30, and the free plan allows SIP data at least 15 minutes old (`brief:31`). So the free delayed feed is enough, and IEX has no decision value.

**F7 — Medium: two data-quality rules are disproportionate and contradict each other.**
- **Missing minutes.** `brief:123` stops the whole study for any unexplained missing minute anywhere. The same paragraph also defines eligibility per session based on whether the endpoint prices exist. One minute missing at 11:17 on a single day would void a test that never uses that minute.
- **Close reconciliation.** `brief:125` requires explaining every close difference across about 2,700 sessions. Bar-level data can't do that, so the study could stall without ever reaching a verdict.
- **What to decide before the pull:**
  - whether only the endpoint minutes (prior close, 09:59, 15:30, exit) are binding
  - how an ineligible session is handled (for example as a no-trade day, which matches what a live trader would do)
  - whether there is a cap on ineligible sessions, and its value (an owner choice)
  - that reconciliation applies only to the chosen convention's series, with everything else diagnostic

  The brief currently forbids relaxing these after data are seen. That is correct, so the change has to happen now.

**F8 — Medium: the paper shadow needs its own account.**
- The production paper trader runs at **15:45 ET** (`scripts/paper_trade.sh:5`), inside the 15:30–16:00 holding window.
- Its cash-sweep symbol defaults to **SPY** (`src/ggTrader/paper/cash_sweep.py:38`).
- Intraday SPY orders in the same paper account would collide with production reconciliation.
- **What's needed.** A separate paper account or a ledger-only shadow. I have not verified whether these credentials can get an additional paper account.

**F9 — Low-medium: work that can be cut.**
- **IEX diagnostics.** IEX had 9 of 20 samples empty and 3,607 missing minutes. Drop the IEX rerun (`brief:106`) and the IEX parts of 0B.
- **The separate 20-date 0B stage.** It becomes redundant once the whole window gets mechanical checks: positive and finite prices, consistent bar fields, duplicates, calendar, endpoint presence.
- **The proposed supplemental 0A manifest** (`next_steps.md:11-15`). Not needed.
- **Size of the pull.** For a single symbol it is roughly 100 pages per feed at 10,000 bars per page. Sampling first saves almost nothing.

**F10 — Low: multiple testing across the intraday track.**
- The free archive gives one shared 2016–2026 sample. Every later SPY intraday candidate reuses it, and `brief:73` already admits no correction applies.
- **What to decide:**
  - keep a running count of intraday hypotheses tested on this window, reported in each verdict
  - forbid this study's diagnostics (subperiods, attribution, convention table) from setting the next candidate's parameters
- "OOS" in step 6 should read "post-original-sample, single-shot". A related in-window negative result is already public (`replication-source-audit.md:142`).

**F11 — Low: two items to note, neither blocking.**
- **A1's meaning is still open** (`brief:91,142`). My recommendation: keep it as a profitability gate, which is what matters to the owner. Also pre-register how the always-long last-half-hour comparison is read: if that rule earns a similar return, a pass is labelled "afternoon drift; signal unsupported". Don't add a second gate.
- **Wash sales.** Interaction with the owner's real SPY holdings (the 70/30 SPY/SSO mix) is a pre-live accounting item, not a research blocker. The policy tax haircut is adequate as a research hurdle.

## What binds now vs what is still proposed

**Binding now:**
- owner instructions (`AGENTS.md:5-15`)
- the production freeze and no changes under `src/ggTrader/paper/` (`brief:133`)
- no paid data
- the executed 0A contract and its recorded "unresolved" result
- no authorization yet for 0B, bulk pulls, returns or orders (`brief:150-152`)
- the forward-paper standard, approved per line 3

**Proposed and still changeable before any data:** window, price conventions, A1/A2, costs, §9 data-quality stops, 0B/0C staging, and §8's route to a shadow.

## Owner decisions needed (all blind to data)

- **D1.** Apply the pre-registered fallback: first test session 2016-01-05, end 2026-09-30, and close the 2014 option.
- **D2.** The executable exit (whole-share market-on-close or something else), and with it whether auction close is primary. If it is, decide what "covers the window" means: exactly one condition-code-selected `P` closing record per eligible session, otherwise the 15:59 close for the whole window.
- **D3.** A1 as a profitability gate plus the drift-interpretation rule, or a formal predictive gate instead.
- **D4.** Make A2 consistent with what comes next:
  - **Option 1:** keep A2 standalone and register an intraday-specific forward standard (cash yield, intraday timestamping, separate account, its own pass/stop rules).
  - **Option 2:** accept the existing SPY-relative standard, knowing the likely outcome after the F2 check.

  Also decide zero vs T-bill cash.
- **D5.** Data-quality scope from F7, including how ineligible sessions are handled and any cap.
- **D6.** Replace the staged 0B/0C with one authorization: freeze, then pull the whole window, then run mechanical validation that ends in pass or UNRESOLVED (data). Optionally pause for an owner "go" before scoring.
- **D7.** Shadow isolation: a separate account or a ledger-only shadow.

## Revised sequence

1. **Freeze the protocol (r8, documents only).** Settle D1–D7. Include the minimum-detectable-effect note from out-of-window or published inputs and the exact scoring specification. Owner approves, and the r8 file is hashed.
2. **Build the pull, validator and scorer** with mocked tests only. Commit the scorer, and record its hash, before any real data exists.
3. **One authorized whole-window SIP pull** of bars, plus auction metadata with condition codes if D2 needs them, recording where each response came from. The mechanical validation covers what 0B was meant to; no returns are computed. The outcome is pass or UNRESOLVED (data).
4. **Score once.** Run A1/A2 and the registered diagnostics, write raw JSON, and record GO or NO-GO for this candidate. A NO-GO closes this candidate only; the next intraday candidate gets its own brief and is added to the track's hypothesis count.
5. **Only on a pass:** write the forward-shadow brief under whatever D4/D7 settled, approved separately.

## Single immediate next task

Draft **brief r8, the protocol freeze**: a documents-only decision memo that sets out D1–D7 with the F2 power note and the scoring specification, for owner approval. No requests, no prices, no code.
