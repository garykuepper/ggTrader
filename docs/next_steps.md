# Next Steps

This file is the **only-look-1-2-steps-ahead** worklist — literally the next
thing(s) to do, nothing further out. It exists so work can be handed to a
cheaper model without them needing to re-derive context from `roadmap.md`'s
full history. When a step here is done, delete it and add the next one (if
any) — don't let this file accumulate a backlog.

**Step 3: Re-measure the SP500-core 1.12 baseline under `eligible_at()` PIT
universe construction** (same mechanism `blend.py`'s sleeves already use),
same `--eval-start 2021-01-31 --eval-end 2026-04-30` window, so it's directly
comparable to the July-12 leverage-realistic blend result (Sharpe 1.14, MaxDD
−5.39%, `roadmap.md` §3). The current 1.12 baseline used
`equity_universe_between()`, a static union of every symbol that was ever a
member across the full window — not true per-fold membership — so the
1.14-vs-1.12 comparison isn't single-methodology yet. This is the one open
item blocking a deploy call on the 3-sleeve blend.
