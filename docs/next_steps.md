# Next Steps

This file is the **only-look-1-2-steps-ahead** worklist — literally the next
thing(s) to do, nothing further out. It exists so work can be handed to a
cheaper model without them needing to re-derive context from `roadmap.md`'s
full history. When a step here is done, delete it and add the next one (if
any) — don't let this file accumulate a backlog.

For "why" context behind these two steps, see `roadmap.md`'s **Future
Research** row and the July-7 entries under **§3 Future Research
Directions**. Don't re-read the whole roadmap to execute — everything needed
is below.

---

## Step 1: Fix `blend.py` so it can blend weight-based strategies

**Why:** `run_blend()` in `src/ggTrader/lab/blend.py` calls `run_wfo()` (line
~102) without passing `universe_fn`. Weight-based strategies (anything with
`target_kind = "weights"` in `STRATEGY_REGISTRY` — currently `xs_momentum`,
`dual_momentum`, `idio_vol`) require `universe_fn` or `run_wfo` raises
`ValueError` (see `wfo.py`'s `_sweep_fold_dispatch`). Right now `--blend` can
only combine signal-based strategies (e.g. `ensemble@sp500,ensemble@midcap400`
works fine) — it cannot include `idio_vol` in a blend at all.

**The fix:** mirror the exact fix already applied to `cli.py`'s plain `--wfo`
path (see `git log -p --all -- src/ggTrader/lab/cli.py` around commit
`e9f20ba`, message "feat(lab): allow weight strategies through --wfo" — the
change was a one-line addition of a `universe_fn=` kwarg to a `run_wfo(...)`
call). In `run_blend()`, each sleeve already has its own `universe` string in
the `(strategy, universe)` tuple (see the `for strategy, universe in
sleeves:` loop). Pass:

```python
universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe=universe)[0],
```

into that loop's `run_wfo(...)` call. `eligible_at` needs importing from
`ggTrader.lab.data` (check whether `blend.py` already imports it — it may
not; `cli.py` already does this exact import for reference).

**Watch out for:** `universe` is the loop variable — make sure the lambda
captures it correctly per-iteration (Python late-binding closure gotcha: a
lambda defined inside a `for` loop captures the loop variable by reference,
not by value, so if `run_wfo` doesn't call the lambda until after the loop
moves on, every sleeve's lambda would see the *last* `universe` value, not
its own). Test this explicitly: blend a `midcap400` weight sleeve and an
`sp500` weight sleeve together and confirm each one's eligible-symbol set is
actually scoped to its own universe, not both resolving to the same one.

**Verification:**
1. Add a test to `tests/lab/test_blend.py` (check if it exists first) that
   blends a weight-based strategy (e.g. `xs_momentum@sp500,idio_vol@sp500` or
   two synthetic fixtures) and confirms `run_blend()` completes without the
   `ValueError`.
2. `pytest tests/lab/ -q` — full suite must still pass, no regressions to the
   existing signal-only blends (e.g. `ensemble@nasdaq100,ensemble@midcap400`
   must still work identically).
3. `ruff check src/ggTrader/lab/blend.py`.
4. Smoke-test for real: `.venv/bin/python -m ggTrader.lab.cli --strategy
   ensemble --blend "ensemble@sp500,idio_vol@sp500" --eval-start 2021-01-31
   --eval-end 2026-04-30` — must complete without exceptions.

Commit this as its own commit before moving to Step 2.

---

## Step 2: Run the 3-4 sleeve diversification blend study and record it

**Why:** Three sleeves now have documented pairwise correlations against the
deployed SP500 core (0.70 MidCap, 0.447 idio_vol) or against each other
(0.35 MidCap-vs-Nasdaq), but nobody has run them together in one blend. The
MidCap+Nasdaq pair alone already produced the best risk-adjusted result found
to date (Sharpe 1.39, drawdown -4.0%) per `roadmap.md`'s June-27
diversification-measurement note — adding `idio_vol` into that mix is the
open question.

**Prerequisite:** Step 1 must be done and merged first (this needs
`idio_vol` to work inside `--blend`).

**What to run** (after Step 1 lands):

```bash
.venv/bin/python -m ggTrader.lab.cli --strategy ensemble --blend \
  "ensemble@sp500,ensemble@midcap400,ensemble@nasdaq100,idio_vol@sp500" \
  --eval-start 2021-01-31 --eval-end 2026-04-30
```

Also run the 3-sleeve version without `idio_vol` (`ensemble@sp500,
ensemble@midcap400,ensemble@nasdaq100`) as the comparison baseline, since the
existing 1.39-Sharpe MidCap+Nasdaq number was a 2-sleeve result, not this
3-sleeve combination — you need a fair baseline run alongside the 4-sleeve
one, not just a comparison against the old 2-sleeve figure.

**Record the result** in `docs/roadmap.md`'s §3 Future Research Directions,
following the exact format of the existing July-7 entries (verdict word in
the header — Deployed/Rejected/Researching — CAGR/Sharpe/MaxDD numbers for
each blend, and an explicit call on whether adding `idio_vol` improved the
blend's risk-adjusted return over the 3-sleeve baseline or not). Do **not**
draw a "deploy this" conclusion without checking the blend's numbers run
through the same gated WFO pipeline every other verdict in this roadmap uses
(inverse-vol/target-vol blend numbers from `run_blend` are already
gate-honest per-sleeve, but the blend combination itself is not additionally
gated — say so explicitly in the roadmap entry, matching the existing
"idealized... run through the live gates" caveat language used in the
June-27 diversification note).

**Verification:** the CLI command completes without exceptions and prints
inverse-vol/target-vol blend metrics for both the 3-sleeve and 4-sleeve runs;
`docs/roadmap.md` is updated with the honest comparison; commit.

---

## After both steps

Delete this file's content and either leave it as an empty placeholder or
ask what's next — do not add Step 3 speculatively. The next research
direction should be picked based on what Step 2's actual numbers show, not
decided in advance.
