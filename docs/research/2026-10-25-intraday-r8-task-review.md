> Historical task review at `fdea52b`. I2/I3 and M1/M2/M6 were subsequently
> resolved in `7b0c088`; the [scoped re-review](2026-10-25-intraday-r8-fix-review.md)
> passes specification and quality with no new Critical/Important findings.
> The earlier request to ratify unnecessary name-change stops is superseded.
> The reviewer's belief that SPY never split is not verified evidence and is not
> used to certify corporate-action completeness.

**Verdicts:** Spec **PASS** · Quality **APPROVED**

No critical findings. I found no missing mandatory code. The three Important items are about what the owner must accept explicitly in the approval‑2 request: under r8, an UNRESOLVED (data) result closes the study for good, so each of these can end the single shot.

## Findings

### Critical
None.

### Important

**I1. The corporate-action feed can close the study for reasons unrelated to prices** (`intraday_momentum.py:726-735`, `:686-709`, `:116-134`).
- **When it does stop:** an empty or truncated dividend history can never PASS. The quarter rule stops it.
- **When it can still pass:** a split missing from the vendor's response can PASS. The signal-blind validator has no way to see it.
- **Why the real risk is small anyway:** the safety comes from SPY's public record (I believe SPY has never split), not from the +31-day `process_date` margin, which has no documented delay bound.
- **Is the artifact honest about this?** Yes. `ANCILLARY_COVERAGE` and `ASSUMPTIONS` say the response is not certified complete and the margin is not a completeness bound.
- **Practical blocker:** Alpaca does not document how far back corporate-action history goes. If it doesn't reach 2016Q1, the study ends UNRESOLVED without looking at a single price.
- **Fix (no code change):** the approval‑2 request should state both points. The owner should accept them, or approve a separate probe that reads metadata only.

**I2. Two stop types go beyond r8 D4.3's list and close the study.**
- **Probe result:** a ticker-preserving name change (`old_symbol=new_symbol=SPY`, `process_date=2015-12-10`, which is outside the window) gives `corporate_action_undated:name_changes:0`. That is UNRESOLVED, closing the study (`:701-709`).
- **Same result for:** `capital_gains_distributions`, and any action type that documents only `process_date`.
- **The in-window value-affecting part is defensible.** That covers spin-offs, stock dividends and mergers inside the window. The benchmark can't be valued honestly for these, which matches the parent's "no silent partial diagnostics" ruling. It is a measurement limit, not a performance gate.
- **The undated and name-change part is not defensible.** These stops close the study but have no effect on valuation.
- **Plausible real case:** the issuer page URL the implementer cited ("state-street-spdr-…") suggests the fund name changed. Whether Alpaca records name changes that keep the same ticker is unknown.
- **Fix:** r8 says any deviation goes back to the owner. List these stop types word for word in the approval‑2 request for ratification. If ratified, no code change.

**I3. The r8 "documented vendor adjustment change" stop can't be triggered by the code.**
- The code says so honestly (`:131-132`, `:1058-1060`).
- Scoring runs automatically once validation passes, so the only place this check can happen is before collection.
- **Fix:** the approval‑2 request should include a manual check of Alpaca's documented data changes for 2016–2026, or record the stop as unenforced.

### Minor
- **M1. Empty currency is treated as USD** (`:718`). This is a necessary contract choice: the rate only feeds the non-gating benchmark, and SPY is a USD fund. But it isn't recorded in `ANCILLARY_COVERAGE` or `ASSUMPTIONS`. Fix: add one limitation line, and optionally a count of dividends with empty currency.
- **M2. Runtime library versions are recorded but not bound.** `run_score` (`:1115-1126`) binds the pandas-market-calendars version through `offline_calendar`, but not statsmodels, numpy or pandas. That is allowed by the spec, which binds only protocol, code and data identities, so it is visible rather than enforced.
- **M3. The protocol hash covers the whole r8 file** (`:55`, `:175-180`).
  - The runtime appendix must live in a separate file, or the pipeline refuses (it fails closed).
  - `docs/next_steps.md` ("Opus is building…") is stale.
  - `docs/architecture.md` and `docs/cli_reference.md` have uncommitted changes in the worktree. These are parent-owned and I did not review them.
- **M4. A crash after the claim forfeits the score permanently** (`:1130`). This is by design. Run the score on the host, outside the 8G test cgroup.
  - My measurement: about 0.33 KB per parsed bar, so about 0.9 GB for 2.6M bars.
  - Indexes push the peak to about 1.5–2 GB, which is not a blocker.
- **M5. The bar-based `unflagged_half_day` check is weak** (`:540-544`). Real half-days have extended-hours prints between 13:00 and 16:00. The offline calendar check (`:387-415`) is the guard that actually works.
- **M6. A chunk whose JSON is not a dict raises `AttributeError`** in `_provenance` (`:431`) instead of reporting a defect. It still can't produce a PASS.
- **M7. Page body hashes are recorded, but the bodies aren't stored.** So the records can't be re-derived from those hashes, and the binding rests on the stored chunk file hashes. Mention this in approval 2.
- **M8. `x-ratelimit-reset` uses the real clock** (`time.time()`) instead of the injected one (`intraday_momentum_data.py:197`).
- **M9. A near-constant series scores a huge t.**
  - A constant series with one 1e‑15 change gives t ≈ 8e14.
  - That is mathematically correct, because the variance is not zero, and real prices can't reach it.
  - Adding a tolerance would be a new threshold, so no fix.

## Your eight concerns
1. **Coverage:** covered under I1. The text is honest. A missing split can PASS; missing dividend history cannot.
2. **USD:** a necessary contract choice with negligible risk; only the disclosure is missing (M1).
3. **Undated/unsupported actions:** in-window value-affecting actions are a measurement limit; undated actions and name changes need owner ratification (I2).
4. **Claim:**
   - A different `--output` is refused (`:1105-1107`, with a test at `test_intraday_momentum.py:666`).
   - A crash leftover is refused.
   - A copy made before scoring can still be scored, as already admitted. Both artifacts record `data_manifest_sha256`, so a second result would be detectable.
5. **Identities and NaN:** a NaN result can't be reported as GO.
   - What's bound: identities cover the stored chunk file hashes; page body hashes sit inside the chunks.
   - How NaN is blocked: `nw_t` returns None for non-finite input, `cagr` returns None, so A2 fails.
   - A diagnostic that raises turns a GO into `UNRESOLVED_DIAGNOSTICS`.
   - `_json_safe` silently nulls a NaN in a diagnostic. I found no reachable case on validated prices.
6. **Offline calendar:** the comparison runs both ways, so an empty or truncated library schedule produces defects. A library exception is a defect. The version and schedule hash are bound at scoring. The test that drops a session from the calendar, bars and daily close together is at `:393`.
7. **Benchmark, tax and NW:**
   - **Benchmark:** starts at the anchor's close bar, ends at 2026‑09‑30 15:59, reinvests at the ex-date close, and uses the same `y` as the sleeve.
   - **Tax:** the debit comes from after-debit equity; the terminal partial year is taxed; losses get no benefit.
   - **Zero variance:** only exact equality counts as zero variance.
8. **Memory:** see M4. Not a blocker.

## Strengths I confirmed by reading the code
- **Endpoints:** every scored session needs all four endpoints (`:507-530`).
  - The anchor and half-day close bars use the last minute before the official close (`:513`).
  - Endpoint minutes are built from New York wall time.
- **No signal in validation:** the validator never computes a return.
- **Previous fixes hold:**
  - the exact-constant check in `nw_t` (`:778`);
  - the β₁ exact-residual guard (`:886`);
  - the claim stored in the collection directory (`:1105`);
  - the offline NYSE cross-check (`:387-415`);
  - the alias and `effective_date` placement with no `process_date` fallback (`:698-711`);
  - stored corporate-action records revalidated (`:692-694`);
  - honest split vs. unsupported reasons (`:712-715`).
- **Scoring math matches r8:** NW-5, HAC, `use_correction=True`; the tax ledger; CAGR over the registered span; the complete diagnostic list; stress can never change the verdict.
- **Collector:** GET only; redirects off; proxies off (`trust_env=False`); credentials only from the environment; no dotenv anywhere in the import chain (grep).
  - Retries are finite and pages are spaced.
  - A repeated page token or the page cap leaves the chunk incomplete, and partial pages are thrown away.
  - Chunk files are write-once, and a resume rehashes them.
- **Isolation:**
  - `intraday_probe.py` is unchanged.
  - `pyproject.toml` adds only statsmodels.
  - The protocol file's SHA256 matches `be395d95…5b85`.

## What I ran, and limits
- I read the brief, the r8 protocol, the report, the progress ledger, the three code files, the reused Stage 0A helpers, and the test fixtures plus the calendar and corporate-action tests. For the docs and dependency changes I read the diff.
- I ran `sha256sum` on the protocol and grepped for dotenv.
- I ran two in-memory Python probes (name change, capital gains, anchor dividend, NW near-constant, equal-ratio constant) and one memory estimate. None of them wrote files.
- I did not run pytest, Ruff or the full suite. No unanswered risk needed a test.
- No network access, credentials, real data or edits.

**Still unknown until approval 2:** how far back corporate-action history goes; how strictly the real response matches the expected schema (any mismatch leaves collection permanently incomplete, which means UNRESOLVED); the daily-bar timestamp convention (only documented in Alpaca's FAQ); and whether Alpaca records SPY name changes.
