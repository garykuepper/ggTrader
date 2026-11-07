# Review: "ggTrader Research Prompt Kit" (received 2026-11-05)

**Status:** review only. No code, config or prompt file was changed, and nothing here
reopens a closed study or touches the production freeze.
**Source (verbatim copy):** [_research-prompt-kit-source.md](_research-prompt-kit-source.md).
It is a six-turn prompt sequence (shared context, literature, spec, vectorbt code,
overfitting audit, adversarial review) for an outside chat model. Its author is not
recorded in the file.

**Terms.** *DSR* (deflated Sharpe ratio) is a Sharpe ratio discounted for how many
parameter combinations were tried. *WFO* is walk-forward optimization. *Holm* is the
correction that controls false positives across a family of tests. *Part A / B0 / B1*
are the stages of the [forward-paper standard](../evidence-standard/2026-10-23-forward-paper-standard.md).
*Look-ahead* means a decision used information that did not exist yet.

## 1. Verdict

**Mine it for checks; do not adopt it as written.** Its testing discipline is better than
what the local implementation template asks for. Its setup assumes a different project:
a crypto-first system with its own pipeline, a 12-month holdout and no tax. Run as
written, it would produce results that cannot be compared with our baselines and that
cannot meet our evidence standard.

## 2. What it gets right and we do not already have

I searched `local-implementation-prompt-TEMPLATE.md` and the `research-report-review`
skill for each item. None of these five is in either.

1. **Truncation test for look-ahead.** Recompute signals on data cut at 20 random dates
   and assert they match the full-history signals up to each date. This is the cheapest
   direct check for look-ahead, and it catches full-sample percentiles and z-scores.
2. **Independent profit-and-loss recomputation.** Rebuild five trades in plain pandas,
   outside vectorbt, and match the portfolio's trade records. We have had accounting
   bugs that green tests missed (see the day-shift and duplicate-row findings).
3. **Running trial count N** at the top of every response, including reruns and abandoned
   variants. The repo's DSR gate uses N, but nothing makes a research session track it.
4. **Fragility tests:** drop the five best trades and the best month, delay every fill one
   extra bar, move each parameter one grid step. The template's neighbor and delay checks
   are absent.
5. **Review in a fresh conversation with a different model.** This agrees with the
   delegation rule that a same-context challenger is not independent.

Also sound and consistent with us: citations only from retrieved sources, UNVERIFIED
labels, kill criteria written before results, honest REJECTED as a success, and a
verdict scale that never says "production-ready" from a backtest.

## 3. What does not fit ggTrader

| # | Kit says | Repo fact | Consequence |
|---|---|---|---|
| 1 | Plug into `wfo_pipeline` "with DSR gating, WFE tracking, regime state machine" | No `wfo_pipeline` exists (searched all `.py` and `.md`). The real pieces are `lab/wfo.py`, `lab/gates.py` and the `Strategy` protocol (`select`, `to_targets`). | Turn 3 would produce a module for an interface we do not have. Replace with the `Strategy` protocol and `STRATEGY_REGISTRY`. |
| 2 | Crypto and stocks, Binance.US, fees 0% maker / 0.02% taker | Crypto is parked by design. The 0.02% figure is in the venue report's low-volume table, but the lab's crypto default is 0.40% fee plus 0.15% slippage (`lab/data.py:42-43`) and the same report says to use at least 0.4%. | The kit's base case understates crypto cost by about 20x versus our own default. Drop crypto, or use the lab default. I did not verify the kit's claim that taker fees killed earlier higher-frequency candidates. |
| 3 | Costs and returns pre-tax | The account is taxable only. A strategy must beat SPY after tax (owner instruction; forward-paper standard). | No after-tax view anywhere in the kit. Add one, using the existing tax-lot library. |
| 4 | Benchmarks: buy-and-hold, equal-weight basket, SPY | Our rules add: the *static* version of the same instruments for any timing rule, SPY de-risked to the same equity share with the rest in T-bills, and Sharpe on excess-of-T-bill returns. | Without these, "lower drawdown" cannot be told from "held less stock", and "diversification" from "timing skill". |
| 5 | Locked holdout of the final 12 months; DSR as the gate; top verdict "SHADOW CANDIDATE" | The standard: Part A needs a long free-history test with Holm t at least 3 and at least +1.0 pt/yr net since 1994; then B0 kill-only replay and B1 forward shadow of at least 18 months. | One 12-month holdout cannot meet Part A. The kit's SHADOW CANDIDATE must be read as "worth registering for Part A", never as meeting the standard. |
| 6 | Signal on close of t, fill at open of t+1, `price=open` | `lab/simulate.py` passes `close` to `from_orders` and `from_signals` with no `price=open` (lines 61 and 277). The live trader runs 12:45 PT, about 15 minutes before the US close, so live fills sit near the day's close. | Adopting the kit's convention makes new results incomparable with every earlier lab number and unlike live. I did not trace whether weights are lagged upstream of `simulate.py`; confirm before deciding. Best handling: pick one convention per study, register it, and report the other as a sensitivity. |
| 7 | Example hypothesis: ATR-percentile regime filter on a slow EMA trend | Trend and volatility timing is already closed: `leveraged_trend_*` NO-GO, `commodity_trend` NO-GO, and the cross-asset trend arm added nothing over static holding. | The example invites a repeat. Any timing hypothesis must name the static benchmark it has to beat. |
| 8 | Starts from one hypothesis; Turn 1 asks a chat model to find papers | Discovery already runs through `research -d deep -k academic`, which writes a claim ledger of verified, contradicted and unverified rows, then the backlog dedupe in `research-prompts` Mode B. | Turn 1 duplicates a stronger local step. Use the ledger, and keep the kit's "spot-check two IDs" habit on top. |
| 9 | Data "[SOURCE]" left open | No paid data, now or later (owner, 2026-10-23). Point-in-time universe membership is required for index stocks. | Name free sources and the point-in-time rule in Turn 0. The kit mentions survivorship only for single-stock lists. |
| 10 | Stack: vbt indicator `.run()` plus a standalone script | The lab simulates all strategies in one `from_orders` / `from_signals` call and the harness runs the folds. | A standalone script duplicates logic and drifts from the lab. Keep only the checks (truncation, plain-pandas P&L) as test code against the real strategy class. |

## 4. How it connects to the goal

The goal as recorded: an active stock-trading path in the owner's taxable account,
judged against SPY after tax, on free data, with the paper trader frozen and effort
going to research evidence and paper-trading reliability.

- **Helps most:** stage 3 of the idea pipeline (implementation and test of a chosen
  candidate). The five checks in section 2 would harden every future candidate and
  directly address the failures that cost us months: look-ahead and bad accounting.
- **Does not help:** discovery (we have a better loop) or the live path (it contains
  nothing about paper-trading reliability, the Alpaca order path or the freeze).
- **Risk if misused:** a clean-looking REJECTED / SHADOW CANDIDATE label from an outside
  model on a 12-month holdout could be mistaken for the registered standard.

## 5. Suggested adoption (not done; awaiting owner go-ahead)

1. Add the five checks to `local-implementation-prompt-TEMPLATE.md` under "What you must
   produce", adapted to the `Strategy` protocol. This is a template edit; the
   `research-snapshot` skill owns its context sections, so only the production-checks
   section would change.
2. Add truncation and plain-pandas P&L checks to the `research-report-review` skill's
   checklist.
3. Settle the fill-convention question (item 6) in a short note before any new
   candidate is registered. This is the one issue that changes comparability.
4. Do not store the kit under `research/prompts/`; it would sit beside the two live
   prompts and look endorsed. It stays here as a reviewed source.

## 6. Limits of this review

Read in one pass against the repo on 2026-11-05. Not checked: the kit's author or
provenance; whether Binance.US fees are currently 0.02%; whether the kit's DSR and
CSCV (combinatorially symmetric cross-validation) instructions match `gates.py`'s
corrected DSR; and the upstream lag of weights in the lab harness.
