# Project Roadmap

Where ggTrader is going, in plain words. Updated 2026-11-20. The long June-to-November version of this
document (with every research arc, the old "in-flight tasks" and the full timeline) is kept as history in
[docs/archive/roadmap-2026-06-to-2026-11.md](archive/roadmap-2026-06-to-2026-11.md); read its warning banner
before citing any number from it.

- The current worklist, one or two steps deep: [Next Steps](next_steps.md). **Start there.**
- Every strategy tried, with its verdict: [Research Snapshot](../research/RESEARCH_SNAPSHOT.md).
- How the code is built: [Architecture Guide](architecture.md). What changed and when: [Changelog](changelog.md).

---

## 1. North Star

Build an honest, reproducible research lab, and put real money behind a trading rule only if it survives
strict out-of-sample tests and a long clean paper-trading record. Nothing has cleared that bar yet.

- **Benchmark:** SPY buy-and-hold, in a taxable account. A rule has to beat it after costs and tax.
- **Evidence standard:** the owner-approved [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
  Part A needs a long free-history test with a Holm-corrected t of at least 3 and at least +1.0 percentage
  point a year net since 1994. B0 is a kill-only replay. B1 is a forward paper-only copy (a "shadow") kept
  for at least 18 months. Real money needs six months of clean paper trading first.
- **Labels:** GO/NO-GO labels are retired for new work. A new result states the evidence, whether each preset
  threshold was met, and the uncertainty. Older verdicts stay as history.
- **Limits:** no paid data; the account is taxable only; the owner decides every deploy and restart.

---

## 2. Where things stand (2026-11-20)

Basis points (bps) measure return differences: 100 bps is one percentage point a year.

| Component | Status | Notes |
|---|---|---|
| Research | Paused since 2026-11-12 | No validated stock picker. The [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md) shows each category's gap to its benchmark and approximate bands; none is proven alpha. |
| Paper trader | Stopped since 2026-11-06 | Cron lines are commented out. The code on `main` is not deployed: the image is rebuilt on every push, but the container is not pulled or restarted. |
| Engine | Rebuilt | vectorbt 1.2.0 with the Rust simulator; the only code that imports vectorbt is `src/ggTrader/engine/`. |
| Paper-trader code | Refactored | The run is split into step objects behind typed ledger and broker interfaces; a live run refuses a non-durable ledger. Behavior was proven unchanged by recorded transcripts, oracle runs and reviews. Merged and pushed 2026-11-20. |
| Membership | Running | The S&P 500 membership list lives in the database as versioned vintages; the live gate fails closed. The weekly check runs Saturdays at 06:00 Pacific. |
| Intraday observer | Recording | Four timers record a hypothetical SPY journal and place no orders. Supervision is paused. Checkpoint 2026-11-30. |
| Study code | Isolated | Each study keeps its code, tests and reports under `research/`; the test tiers (fast core, full, studies) are described in [AGENTS.md](../AGENTS.md). |
| CI | Green | The full suite runs on every push to `main` (it cannot run the real-tape oracles, so those run locally before a merge). |

---

## 3. What the evidence says

- **The fixed-rule stock model** (the deployed ensemble plus a SPY sweep) earned 14.19% a year against SPY's
  14.83%, excess −64 bps, Newey-West t −0.69, on the pinned primary window. The audit rated its own test too
  weak to see a modest edge, so the result is inconclusive against its +1.0 point bar, not a proof of no skill.
  ([audit](../research/audits/production-replica/2026-10-19-production-replica-audit-nogo.md))
- **The standalone SP500 core** (ensemble WFO on the Rust engine, OOS 2022-01-31 to 2026-04-29): Sharpe 0.3905,
  CAGR 3.50%, against SPY 0.7792 and 13.02%. The corrected membership list moves it only to 0.3958 and 3.56%.
  ([snapshot, section 1](../research/RESEARCH_SNAPSHOT.md))
- **Stock screens** of 2026-11-10 and 2026-11-11 (eight published-portfolio screens), plus the earlier momentum,
  factor and net-issuance screens: none cleared both hurdles. Expected investment growth came closest on the
  recent-return threshold but not on significance.
- **Allocation:** the cross-asset 80/20 sleeve is a shadow only (thin edge, mostly gold). The owner adopted a
  70/30 SPY/SSO allocation (A23) knowing its after-tax edge is +0.80 point a year at today's funding costs,
  which is below the bar. That is extra market exposure, not selection skill.
- **Intraday:** the SPY last-half-hour rule, scored once on free Alpaca minute bars (2016 to 2026), lost money
  before and after costs: t −3.48, −3.7% a year after the illustrative tax ledger.
  ([result](../research/intraday/spy-momentum-feasibility/2026-11-15-intraday-half-hour-rule-result.md))
- **Process findings that outlive any verdict:** point-in-time membership, one-day tape shifts, duplicate rows and
  unpinned windows each overturned earlier headline numbers; a closed test is a verdict on the test, not proof of
  no edge ([root-cause summary](../research/audits/root-cause-2026-11-04/2026-11-05-executive-summary.md)).

---

## 4. Open tracks

Each track names what must happen before it moves. None starts without the owner's word.

| Track | Gate | Next step |
|---|---|---|
| **Paper trader restart and parity** | The untradable AVB holding must be resolved; the rule for what starts or resets a clean window must be set; the owner says so. | The two unsent drafts (AVB broker inquiry, data-permission request) wait on the owner. Then plan the deploy: rebuild and pull the image, restart in dry run first. |
| **Forward evidence** | A clean observation window. Owner choices of 2026-11-11: the original 80/20 shadow for future parity, a three-month mechanics review and a 12-month performance review. | Define the governing reconciliation rule, daily evidence and how decision inputs are preserved. |
| **Intraday observer** | Operational checkpoint 2026-11-30; final slot 2027-11-03. No performance result is implied. | Before the checkpoint, run the manual health checks (supervision is paused) and look for Saturday's weekly heartbeat. |
| **New research** | A registered question, a brief and the owner's approval. Expected-growth inputs still lack verified identity, timing and accounting history. | Pick from the ranked ideas in [snapshot section 6](../research/RESEARCH_SNAPSHOT.md); candidates from outside are queued in [WEB_RESEARCH_CANDIDATES](../research/WEB_RESEARCH_CANDIDATES.md). |
| **Allocation (A23)** | Annual review on 2027-09-30. | Decide 30% versus 15% SSO at the review; month-end advisory checks once funded. |
| **Two-symbol paper sweep** | After the freeze and readiness gates. | Design exists ([note](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md)); building it needs its own decision. |
| **Engineering follow-ups** | None blocking. | A real-database regression test for the persist writers on a disposable cluster (the merged fixture is flagged not safe for a superuser URL); review the unwired whole-job test broker before wiring any of it in; three small code observations: `lab/a23_rebalance.py` validates a lot's date before normalizing it, `data/membership_weekly.py` has a silent `except Exception` around the prior-attestation lookup, and `tests/paper/equivalence/run_with.py` still patches a method that no longer exists (unreachable). |
| **Parked** | Owner decision of 2026-10-23: no paid data. | PEAD and insider-trade studies; crypto. |

---

## 4a. Infrastructure

| Feature | Status | What it is |
|---|---|---|
| Price data | Done | A frozen Parquet tape for research and a database table for the live path. The live fetch chain reads the database cache, then yfinance, then Tiingo for gaps. |
| Membership database | Done | Versioned vintages (the original file import, the upstream download to 2026-08-18, a curated gap-fill to 2026-11-04) read through one reader; a weekly job cross-checks it against the SPY fund holdings. |
| Walk-forward engine | Done | `ggt lab --wfo` runs rolling folds from one grouped simulation, with gates, an anchor and a circuit breaker, and writes a JSON record with provenance. |
| Blend runs | Done | `ggt lab --blend` mixes `strategy@universe` sleeves and stores a `blend:` run in the `lab_*` tables. |
| Universes | Done | S&P 500, Nasdaq 100, Russell 2000, MidCap 400 and the allocation universes (`--universe`). |
| Test tiers and CI | Done | Fast core, full and study tiers through `scripts/run_tests.sh` (one heavy job at a time, memory-capped); CI runs the fast core on pull requests and the full suite on every push to `main`. |
| Evidence tooling | Done | Pinned hashes (`verify_pins.py`), study scaffolding and repro (`scripts/study.py`), the verdict register and the research-snapshot skill. |
| Not built | Idea only | A side-by-side run comparator (`ggt compare`) and storing walk-forward fold detail in the database. |

---

## 5. Timeline (2026)

- **June 10 to 16:** honest monthly walk-forward and the new lab core; legacy crypto execution removed on the 16th.
- **June 20:** ensemble paper trading started on Alpaca.
- **June 24 to 28:** 3% position sizing, five-voter default, then the ML entry gate, exit sweeps and
  IC-weighted voting were tested and rejected.
- **July:** leveraged-ETF rotation and trend, pairs, MAX-effect and other candidates tested and rejected.
- **August 19 to 22:** the baseline Sharpe of 1.12 did not reproduce; a one-day shift in every stored equity bar
  and duplicate SPY rows were found and fixed. Earlier figures were re-measured.
- **September 22 to 30:** standalone core and the core-plus-SPY blend re-evaluated with point-in-time
  membership; live exit parity fixed; cross-asset shadow started; 70/30 SPY/SSO adopted by the owner.
- **October 19 to 23:** production-replica audit (inconclusive, 64 bps behind SPY); momentum, factor and
  net-issuance screens closed; no paid data; PEAD and insider studies parked.
- **November 5 to 8:** research reorganized under `research/`; root-cause investigation accepted; engine rebuilt on
  vectorbt 1.2.0 with Rust and the trader stopped (November 6); study code isolated from `src/`.
- **November 8 to 12:** membership database and weekly job (enabled November 9); eight stock screens, the verdict
  register and the category report; development paused.
- **November 15:** the intraday half-hour rule scored once: thresholds not met.
- **November 18 to 20:** test-run safety guards; refactor phases 0 to 5, lint and docstring rules; CI repaired;
  project directory cleaned.

---

## Terms

- **WFO (walk-forward optimization):** tune on a past block, score on the next unseen block, roll forward.
- **OOS / IS:** out-of-sample (unseen) and in-sample (tuning) data.
- **Sharpe ratio:** return per unit of risk; higher is better.
- **Drawdown:** the biggest peak-to-trough fall in account value.
- **Shadow:** a paper-only copy of a book that places no orders.
- **Parity:** the paper trader doing what the backtest says it would.
- **A23:** the owner's 70% SPY / 30% SSO allocation. **AVB:** the paper account's frozen holding, which the broker
  marks untradable. **PEAD:** drift in a stock's price after earnings news.
