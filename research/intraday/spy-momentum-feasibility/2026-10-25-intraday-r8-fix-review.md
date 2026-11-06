# Scoped re-review of r8 fixes (`fdea52b..456f5c2`)

I read the source, docs and vendor-review JSON parts of the diff and checked the new test names. I ran no tests and had no network access, so I could not check the URLs in the vendor-review record myself.

## Per-finding status

**I2: ADDRESSED.**
- **Name-change exemption:** it applies only when `old_symbol == new_symbol == "SPY"` and no other alias is present (`intraday_momentum.py:756-762`).
- **Undated same-ticker changes:** these are skipped before the date lookup, so an undated one is counted metadata (`:800-802`).
- **Real symbol changes:** an actual change, or any other non-cash action that can't be placed in time, is still `corporate_action_unsupported_undated` and therefore UNRESOLVED (`:810`).
- **Splits:** an in-window split always hits `corporate_action_stop:split` (`:816`).
- **Capital gains:** the long- and short-term legs are summed. At least one leg is required, each leg must be finite and non-negative, and the sum must be positive (`:111-116`, `:735-743`, `:820`).
- **Duplicates and same-date payouts:** a duplicate of the same type is refused (`:829`). A dividend and a capital gain on the same date are added together (`:844-847`).
- **Quarterly sanity check:** it uses cash dividends only (`:837`).
- **Frozen benchmark:** it is preserved. The verdict still depends only on the A1 t-stat and the A2 test (`:978`, `:1035-1038`), and the benchmark stays a non-gating diagnostic.
- **No new gate:** capital gains used to be UNRESOLVED. Now they are valued, so the fix removes stops rather than adding one.
- **No fake proof:** no text claims the action list is complete (`:155-158`, `:776`, `:1187-1191`).

**I3: ADDRESSED.**
- **What the check requires:** `vendor_review_sha256` (`:250-268`) checks the schema, version, SPY, window, protocol hash and the no-market-data flags. It also requires `status == "reviewed"` and an empty `known_unresolved_changes`. Otherwise it refuses with a named reason (`:267`).
- **When it runs:**
  - Collection calls `build_manifest()` first (`data.py:253`), before any directory is created or request sent. `make_session` only builds the session object and makes no call.
  - The review hash is stored in the manifest (`:343`).
  - Validation recomputes the identities (`:542`, `:851-864`).
  - Scoring checks the review (`:1243`) before the claim is taken (`:1253`).
  - Resume refuses if the review changed (`data.py:259/269`).
- **The record itself:** the committed JSON says the review was bounded and is "not proof that no historical vendor change exists". It also says approval 2 must name this exact review and its limits.
- **Nit (deferred):** the binding check uses `!=`, so `0`/`1` would be accepted in place of `false`/`true`. This has no practical effect because the file is committed and bound by hash.

**M1: ADDRESSED.**
- The USD assumption is stated in the coverage limits (`:161-164`) and the assumptions list (`:1183-1184`).
- It is counted (`:833`).
- Any other currency is UNRESOLVED (`:825`), and lowercase codes are rejected as malformed. Nothing is converted.

**M2: ADDRESSED.**
- The runtime versions are stored in the manifest (`:344`) and in the recomputed identities (`:861`).
- A change gives `identity:manifest_runtime_mismatch` at validation, and resume refuses (`data.py:259`).
- Scoring refuses through the identity comparison, before the claim.

**M6: ADDRESSED.**
- A chunk that is not a JSON object becomes `chunk_unreadable:<id>`, with only the id recorded (`:374`).
- A bad `pages` list or a page entry that is not an object becomes `provenance_invalid` (`:508-511`).
- On resume, a non-object chunk gives `unrecorded_chunk_invalid` instead of a crash (`data.py:285`).
- **Deferred:** a `manifest.json` or `progress.json` that isn't an object still raises `AttributeError` (`:356-360`, `data.py:257-266`). It fails closed and falls outside M6's scope.

**I1: still a disclosed limitation, and it should stay that way.**
- No code or doc text certifies that the corporate-action list is complete.
- The only "never split" wording is in the prior review (`2026-10-25-intraday-r8-task-review.md:15`), phrased as "I believe". The approval-2 request must not restate it as a verified fact.
- **Doc inconsistency (minor, parent-owned):** the prior review's I2 fix text (`:22-27`, `:55`) still says name changes "need owner ratification". The `progress.md` ruling supersedes that. The final record should say so, so the approval-2 request doesn't ask the owner to ratify it.

The other minor findings (M3, M4, M5, M7, M8, M9) were left unchanged, and the diff adds no new tolerance.

## New breakage from the fix range

There are no new Critical or Important issues. I checked these points:
- The rename from `dividends` to `distributions` has no stale callers.
- The report renderer doesn't use the renamed keys.
- Stored capital-gains records match their stored form when re-checked.
- The top-level `runtime` key was removed with nothing left reading it.

## Practical preconditions for approval 2 (not defects)

1. **Pin the environment.** Collection, validation and scoring must all run in one fixed `.venv`. Any library upgrade in between, even a patch release, permanently refuses that collection directory. This is by design.
2. **Freeze the review file before collecting.** Any edit to the vendor-review JSON changes its hash, so the same directory can no longer be resumed or scored.
3. **Single-leg capital gains are trusted.** If the vendor leaves out the second leg, the code can't tell. It only affects the non-gating benchmark diagnostic.

## Verdict

**Spec PASS / Quality APPROVED.** Task 1 can close. The limits on corporate-action completeness (I1), the vendor-review scope and the USD assumption must be stated explicitly in the approval-2 request, along with the three preconditions above. Approving this implementation does not approve real collection, and it does not certify that unknown vendor or corporate-action changes are absent.
