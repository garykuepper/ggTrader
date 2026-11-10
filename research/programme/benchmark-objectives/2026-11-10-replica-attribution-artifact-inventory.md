# Replica attribution: artifact availability

**2026-11-10 — feasibility inventory; no attribution or return test.** This is a bounded inspection of tracked artifacts and code, not a search of private accounts or a claim that no other artifacts exist. The historical replica NO-GO remains unchanged; live parity remains unverified.

The [benchmark synthesis](2026-11-10-objectives-benchmarks-consolidated.md) asks whether the retained model permits a self-financing SPY replacement for the active sleeve. The available aggregates support some descriptive comparisons but do not establish that the complete transaction-level counterfactual can be reconstructed. In particular, per-order cash flows and holdings by symbol were not found in the retained outputs inspected here. Re-running a model is not authorized by this inventory and could use changed inputs.

## Available artifacts

- [Replica results JSON](../../audits/production-replica/_production_replica_results.json): SHA-256 `95c00309be1b03d62443028f3f2c64fc738752997b7972d2420a75316b0f8d0b`. It records configuration, aggregate statistics, counts by symbol and side, and identified artifact-adjacent trades. Those counts are not a chronological fill ledger. The stopped holdout is not clean evidence.
- [Curves CSV](../../audits/production-replica/_production_replica_curves.csv): SHA-256 `5934e81c0e078282397abb47972e924f32d2aa53c4c19aa366a6a7dde8609e42`. Its header contains date, window, arm, total equity, SPY, an exposure-matched curve, and aggregate stock, SPY and cash weights. It does not name individual positions, fills, dividends or fees. Daily aggregate weights could support a *defined aggregate proxy*, subject to timing, cash and cost assumptions; they do not themselves establish an executable sleeve ledger.
- [Replica model](../../audits/production-replica/code/replica.py): SHA-256 `4ce0281a6f1301333baeb83bd1b5474ecedd56bd4a15e83d3295524988b22847`; [driver](../../audits/production-replica/code/run.py): `f11610135888489059b0078ccad7998dc1bdfeeb0618cceebcda6f1f6715a6f5`. These hashes identify the current files, not proof they are the exact historical executable. The driver saves aggregate results and curves; the model constructs trade and position state in memory. Historical code and input binding would need separate verification before any reconstruction.
- The [original design](../../audits/production-replica/2026-10-19-production-replica-audit.md) describes a prior-bar signal and next-session close fill proxy, adjusted-close data, costs and cash-rate assumptions. It describes the model's rules, not a recorded live execution ledger. FRED TB3MS is loaded via a cache; the inspected outputs do not establish the exact historical cash-rate input hash.
- The [engine port check](../vectorbt-engine/p5c-evidence/replica_port_check.json) describes a smaller frozen-tape panel for a later port comparison. It does not establish that the original full replica input tape, per-trade ledger or live-account history was retained.

## Input-by-input assessment

| Required input | Assessment from inspected records |
| --- | --- |
| Daily active capital and idle cash | Aggregate stock, SPY and cash weights and equity are present; the valuation timing and capital allocation conventions need verification before using them in a counterfactual. |
| Per-symbol positions and decision times | Full position snapshots and a chronological decision ledger were not found in these outputs. The design describes the decision rule and timing, and the summary has counts; neither establishes each historical position. |
| Fills, transaction costs and external cash flows | The design specifies model costs and starting cash. The output schema does not supply all dated fills, fees or external cash-flow records needed to reconcile a transaction-level replay. |
| Distributions and corporate actions | The design uses adjusted closes, not a recorded distribution ledger. A tax-lot or explicit dividend counterfactual needs additional data and accounting assumptions. |
| Membership and price inputs | The driver calls historical membership and the DB-first price loader. An exact original data-vintage and complete input-hash binding was not established by this inspection. Do not assume today's tape reproduces the original input. |
| Tax basis and actual displaced allocation | Not established by these model outputs. Any comparison requires an explicitly specified hypothetical starting state or separately authorized, verified records. |

The [root-cause audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens1-backtest-validity.md) documents a PARA security-identity problem and says its effect on the replica was not measured. This is an unresolved validity limitation, not a quantified attribution. The [replica report](../../audits/production-replica/2026-10-19-production-replica-audit-nogo.md) separately says historical live replication is unverified.

## Next decision

If a future study uses the saved daily aggregate weights, first write a dated proposal stating exactly what the aggregate approximation estimates, its timing and costs, and its inability to recover symbol-level selection or actual fills. Treat the previously inspected period as descriptive. A complete transaction-level analysis needs a provenance-bound ledger and input tape, or a separately proposed prospective collection method. Neither option is approved by this note; no original verdict or production setting changes.
