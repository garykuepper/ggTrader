# Study isolation plan (2026-11-07)

Executable plan for moving study-specific code out of `src/ggTrader` into the studies under `research/`, checked adversarially before use.

- `PLAN_V2.md`: the plan after the adversarial corrections (slices S0-S9, lead calls, oracle O).
- `CORRECTIONS.md`: what the independent check found wrong or missing in the first plan.
- `slices.json`: the machine-readable slices and the oracle commands the execution workflow runs.

Status: S1-S4 are committed on branch `refactor/study-isolation`. The moves (S5a-S9) have not run; the owner capped the work to the safe slices. Paths in `slices.json` point at the original scratchpad; resume notes are in the project memory.
