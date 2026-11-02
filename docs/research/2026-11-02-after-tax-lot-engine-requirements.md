# After-Tax Multi-Stock Lot Engine: requirements and gap analysis

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** Requirements specification and gap table only. **No engine code was written** and no tax validity is asserted. A build needs its own plan and owner approval.

Terms: **B0** = kill-only replay on the clean 2021-01-31 → 2026-04-30 tape; **B1** = forward shadow book (≥ 18 months). **ST/LT** = short-/long-term capital-gain treatment. **HIFO** = highest-basis-first lot sales. **Wash sale** = a loss on a sale disallowed when substantially identical shares are bought within 30 days before or after, with the loss added to the replacement lot's basis and holding period. **PIT** = point-in-time. Source of requirements: [forward-paper standard §§3, 4, 6](briefs/2026-10-23-forward-paper-standard.md).

## 1. Why this is a prerequisite

The standard states that B0 and B1 "need a tested multi-stock lot engine before a verdict is possible" and that data or accounting failures mean **no verdict**, not a pass (§3, §6). Until such an engine exists and is tested, no future stock candidate can be run through B0/B1 under the approved standard.

## 2. Numbered requirements (extracted from the standard)

Each is written to be testable.

| ID | Requirement | Source |
|---|---|---|
| R1 | Multi-stock lots: a dynamic symbol set with per-lot quantity, basis and acquisition date | §6, B0/B1 universe |
| R2 | Primary tax model: $100,000 standalone notional; 24% ST/ordinary; 15% LT | §6 |
| R3 | HIFO lot sales with a fixed deterministic tie-break | §6 |
| R4 | Annual ST/LT netting with carryforwards that **preserve character**; no $3,000 ordinary-income credit; unused terminal carryforward valued at zero | §6 |
| R5 | Wash sales tracked **both 30 days before and 30 days after** a loss sale, with partial quantities, reinvestment and overlapping lots | §6 |
| R6 | Disallowed loss carried into the replacement lot's **basis and holding period**; loss provisional until the forward window closes; decision-time records preserved when adjustments are appended | §6 |
| R7 | Holding periods preserved through partial sales, washes and corporate actions (splits, mergers, ticker changes) | §6 |
| R8 | Dividends: qualified vs unqualified separated; T-bill interest taxed as ordinary income accrued with information available then; a registered free distribution source and a policy for facts not yet known | §6 |
| R9 | Year-end tax paid from book cash; any sale to fund it follows the same advance-decision/next-session-fill rule (no same-day retroactive tax trade); no external top-ups or tax reserve | §6 |
| R10 | Identical assumptions applied to SPY buy-and-hold: reinvestment, costs, tax-payment source and timing | §6, §3 |
| R11 | Quarterly comparable wealth on a hypothetical-liquidation basis (accrued tax plus unrealized-lot tax), not an extra trade; provisional while wash/distribution facts are pending; retain original and adjusted values | §6 |
| R12 | Endpoint: terminal liquidation with tax paid for both books | §6 |
| R13 | B1 execution convention: next-session-close fills, T-bill cash, fixed costs frozen in advance | §3, §4 |
| R14 | Append-only ledger: late data corrections appended with discovery time; decisions never overwritten | §4 |
| R15 | Standalone limit: no assumption that losses offset outside income or gains | §6 |
| R16 | Report ST, LT, dividend and cash tax drag explicitly | §6 |

## 3. Existing code and gap table

The only tax-lot code found is `Book` in `src/ggTrader/lab/a23_rebalance.py` (used by `src/ggTrader/lab/century_leverage.py:176`), built for the three-symbol A23 study (SPY, SSO, BIL). `src/ggTrader/lab/replica.py` carries no tax code (grep found none). Line numbers verified 2026-11-02.

| ID | Status | Evidence in `a23_rebalance.py` |
|---|---|---|
| R1 | **Missing** | Symbols hard-coded: `lots` and `recent_buys` default to `{"SPY","SSO","BIL"}` (lines 29, 38); `pay_tax`/`liquidate` iterate that tuple (lines 133, 145) |
| R2 | **Partial** | `st_rate`/`lt_rate` are constructor arguments (line 26); the $100,000 notional is set by the caller, not enforced |
| R3 | **Partial** | `sell` sorts long-term lots first and then by highest basis (line 66). The standard says HIFO; the extra long-term-first rule must be reconciled in the candidate brief. Ties follow list order, deterministic but not an explicit lot-id rule |
| R4 | **Partial** | `year_end` nets `year_st + year_lt - loss_cf` (lines 116–128) and taxes the ST bucket first. The carryforward is a **single scalar** (`loss_cf`), so character (ST vs LT) is **not preserved**. No $3,000 credit appears, as required; terminal carryforward is not valued separately |
| R5 | **Partial** | Wash handling is **backward-only**: it checks buys in the 30 days before the sale (`recent_buys`, lines 78–79) and ignores same-date lots; there is **no forward 30-day window** and no partial-quantity matching. The standard warns the existing approximation is insufficient |
| R6 | **Missing** | A disallowed loss is added to the *newest* lot's basis (lines 83–85) rather than the actual replacement lot, and the **holding period is not adjusted**. Loss is final at the sale date, not provisional |
| R7 | **Missing** | No split, merger or ticker-change handling; `Lot` holds only date, quantity, basis (line 18) |
| R8 | **Partial** | `distributions` supports only a `lt`/`st` label and reinvests (lines 100–114); no qualified/unqualified test, no T-bill interest as ordinary income, no distribution-source registration |
| R9 | **Missing** | `pay_tax` sells pro-rata with a 1.002 gross-up on the same date (line 139): a same-day retroactive tax trade, which the standard forbids; there is no next-session decision record |
| R10 | **Partial** | The same `Book` can run an SPY-only mix (as `century_leverage.simulate` does for mixes), but no function enforces identical assumptions across two books |
| R11 | **Missing** | No hypothetical-liquidation valuation (accrued plus unrealized-lot tax) at quarterly checks; `value` is mark-to-market only (line 43) |
| R12 | **Present** | `liquidate` sells everything, runs `year_end`, pays tax (lines 144–149) |
| R13 | **Missing** | Fill convention is a constant 1 bp cost (`COST_BPS`, line 10) at the supplied price; no next-session-close logic or frozen-cost registry |
| R14 | **Missing** | In-memory only; no append-only ledger, hashing or correction records |
| R15 | **Present by construction** | The book is standalone; no outside income is modelled |
| R16 | **Partial** | `realized_st_total`, `realized_lt_total`, `taxes_paid`, `disallowed` are tracked; dividend and cash tax drag are not separated |

Tests: `tests/lab/test_a23_rebalance.py` and `tests/lab/test_century_leverage.py` exist; the only wash-related test found (`test_a23_rebalance.py:281`) checks that a flat-price sale triggers no wash-sale warning. No test covers forward-window washes, partial lots, character-preserving carryforwards or multi-stock books (observation from a grep for "wash").

## 4. Proposed test list for a future build

Each test uses a hand-computed expected value.

1. `test_wash_sale_forward_window_disallows_loss`: sell at a loss, repurchase 20 days later → loss disallowed and added to the replacement lot's basis.
2. `test_wash_sale_backward_window_partial_quantity`: buy 100, sell 60 at a loss within 30 days of a prior 40-share buy → only the matched 40 shares disallowed.
3. `test_wash_sale_adjusts_holding_period`: replacement lot inherits the original acquisition date (tacked holding period).
4. `test_wash_sale_loss_provisional_until_window_closes`: a later replacement flips an already-recorded loss to disallowed with the original record preserved.
5. `test_hifo_tie_break_is_deterministic`: equal-basis lots sold in a fixed lot-id order.
6. `test_carryforward_preserves_character`: a ST loss carried forward offsets next year's ST gain first.
7. `test_no_3000_ordinary_income_credit`: net annual loss stays a carryforward and is worth zero at the terminal date.
8. `test_split_preserves_basis_and_holding_period`, `test_merger_ticker_change_keeps_lot_dates`.
9. `test_qualified_vs_unqualified_dividend_rates` and `test_tbill_interest_taxed_as_ordinary`.
10. `test_year_end_tax_funded_by_next_session_sale_not_same_day`.
11. `test_spy_benchmark_uses_identical_assumptions`: same engine, same costs, same tax payment source.
12. `test_quarterly_liquidation_valuation_includes_unrealized_lot_tax` and `test_terminal_liquidation_after_tax_cagr_matches_hand_calculation`.
13. `test_ledger_append_only_correction_preserves_original_decision`.
14. A property test: total tax paid plus terminal liquidation proceeds is invariant to lot ordering when all gains are one character.

## 5. Estimate and boundaries

Planning estimate (unmeasured): three to five focused engineering days for the engine, tests and a reconciliation note, because the standard's wash-sale, character and corporate-action rules are each stateful. A build is not scheduled by this document. It should be its own plan, with the existing `Book` kept unchanged and A23 results reproducible, per the owner's reproducibility rules.

Unknowns: how Alpaca-reported corporate actions will be sourced for historical lots on a free tier; whether the standard's "free distribution source" exists for every held symbol.
