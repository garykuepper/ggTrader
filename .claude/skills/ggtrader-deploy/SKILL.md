---
name: ggtrader-deploy
description: Ship a change to ggTrader's live Alpaca paper trader safely — pre-flight tests and a no-orders/no-Telegram check against real data, commit and push, wait for the GitHub image build, pull and restart the ggtrader_live container, verify the new code and .env inside it, and write the next-run checklist. Use whenever a change to src/ggTrader/paper/, anything the trader imports, dependencies, the Dockerfile, docker-compose or the live .env needs to reach the trader — "deploy", "ship it", "push to live", "rebuild the container", "make this live", "roll this out", "roll back", or after finishing a fix to the trader, even if the user never says "deploy".
---

# ggTrader live deploy

The `ggtrader_live` container runs the code **baked into its image**, not the
repo. A change only reaches the trader after push → GitHub Actions image build →
`docker compose pull && up -d`. Skipping a step has shipped stale code to live
twice, and an unpinned dependency once crashed a whole trading day (2026-09-23:
the rebuilt image pulled plotly 7 / vectorbt 1.0 and `import vectorbt` failed).
The steps below exist to catch exactly those failures before 12:45 PT.

Cron runs `scripts/paper_trade.sh` at **12:45 PT, Mon–Fri**, which does
`docker compose run --rm ggtrader_live python -u ggt.py paper --live`. It
re-reads `.env` on every run through the compose `env_file`.

> **Status 2026-11-06: the trader is stopped.** The owner commented out the
> 12:45 and 06:00 cron lines to move the signal path onto vectorbt 1.2.0 + Rust.
> **Cron stays disabled until the owner re-enables it** (spec OD18): no date,
> green check or finished phase re-enables it, and an agent never edits the
> crontab or runs `ggt paper --live`. While it is off, a deploy means "the
> image and tag are ready", and the first re-enabled run is checked with
> `daily-trader-check`. The image runs vectorbt + vectorbt-rust on **Rust only**:
> there is no numba or legacy fallback, so an engine problem means no orders.

## 0. Classify the change — it decides which steps apply

| Change | Image rebuild? | Steps |
|---|---|---|
| Code the trader imports (`src/ggTrader/paper/`, `lab/strategies/`, `engine/`, `data/`, `lab/` helpers), deps (`uv.lock`), Dockerfile | yes | 1–7 |
| `GGTRADER_TAG` or `UNTRADABLE_SYMBOLS` in `.env` only | **no** rebuild; the tag change pulls an existing image | 1 (tests if code reads a new var), 4, 5, 6, 7 |
| `.env` only | **no** — picked up by the next cron run | 1 (tests if code reads a new var), 5b, 6, 7 |
| docs / research / scripts the trader never imports | no | commit + push only; the push still builds an image, which is harmless |

If unsure whether the trader imports something, check from `ggt.py paper`
(`src/ggTrader/cli/`) down, or just treat it as code.

## 1. Pre-flight

1. **Timing.** Don't deploy between ~12:30 and ~13:15 PT on a weekday: a
   restart or half-pulled image mid-run is worse than waiting. Check the clock.
2. **Tests:** run **both** `scripts/run_tests.sh --full -q` and
   `scripts/run_tests.sh --studies -q`. The deploy gate is the FULL run plus the studies run:
   the default fast core leaves out the live golden and the real-tape oracles, and the
   production-replica tests (`research/audits/production-replica/code/tests/`, which exercise
   `paper` internals) now live in the studies tier, not in `tests/`. Never bare pytest: it has an
   8 GB memory cap for a reason, see CLAUDE.md. A `137` exit is a hang, not a big
   suite. Also run `ruff check` and `ruff format --check` on changed files.
3. **Real-data check, no orders, no Telegram.** Green tests can encode the
   same wrong assumption as the code. Exercise the changed path against the
   real tape and the real account, read-only:
   ```bash
   .venv/bin/python .claude/skills/ggtrader-deploy/scripts/verify_live_path.py
   ```
   It runs today's signal generation (engine self-check, sanity gates, the
   v2 ensemble with the live defaults) and the trader's pure helpers against
   real data with a mocked notifier, and places nothing. Extend it inline for
   the specific change (e.g. call the new helper and print its output).
   Since P7 a plain `ggt paper` (no `--live`) is a **read-only dry run**: no
   ledger, snapshot, peak, split, dividend or tape write, no Telegram unless
   `--notify`. Only `ggt paper --dry-run-writes` keeps the old behaviour that
   wrote state and sent messages; never use that flag as a check. Running even
   the read-only dry run on the owner's behalf is the owner's call (spec 8.4).
4. **Predict the next run.** From step 3, write down what the 12:45 run
   should do (e.g. "sells AEE/HAS/HON/MNST/PNR/VZ, no strategy buys"). If a
   change forces trades, the user should see that list and agree before
   deploy — it is their paper account, and real money is the stated goal.

## 2. Commit

- Stage **only the files of this change**. Other sessions and the user leave
  unrelated edits in the tree (e.g. an uncommitted `AGENTS.md`); never sweep
  them in with `git add -A`.
- The git pre-commit hook runs ruff on staged Python.
- End the message with the attribution lines the session's system reminder
  specifies.

## 3. Push and watch the build

```bash
git pull --rebase origin main   # another session may have committed
git push origin main
gh run list --workflow docker-build.yml --limit 1 --json databaseId,headSha,status
gh run watch <databaseId> --exit-status
```
Confirm `headSha` is your commit before watching. The build takes ~2.5 min
and includes an import check of the trader. A failed build means **nothing
changed on the box** — fix and push again.

## 4. Pull and restart

The trader runs an **immutable tag**, never `:latest` (spec 8.3). `docker-compose.yaml`
reads the tag from `GGTRADER_TAG` in the project `.env`; `docker-build.yml` publishes
the short sha, `sha-<full sha>` and `latest`. Record the current tag and image first so
a rollback is one edit:
```bash
grep '^GGTRADER_TAG=' .env        # only the one line; do not print the rest of .env
docker image inspect ghcr.io/garykuepper/ggtrader:<current_tag> --format '{{.Id}} {{.Created}}'
# set GGTRADER_TAG=<new short sha> in .env, then:
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
docker image inspect ghcr.io/garykuepper/ggtrader:<new_tag> --format '{{.Created}}'
```
The new `Created` must be after your commit time. **A deploy is not done while
`GGTRADER_TAG` is unset or `latest`.** Keep the previous tag's image locally until the new
one has run cleanly (`docker image prune` and `compose pull` orphan an untagged old image).

## 5. Verify inside the container

a. **Code:** import the changed module and assert the new symbol exists,
   inside the image — not on the host:
   ```bash
   docker compose run --rm ggtrader_live python -m ggTrader.engine.selfcheck --engine rust
   docker compose run --rm ggtrader_live python -c "
   import ggTrader.paper.trader  # imports the engine: the 09-23 failure mode
   from ggTrader.paper.trader import PaperTrader
   print('ok', hasattr(PaperTrader, '<new_method>'))"
   ```
   The self-check prints the vectorbt and vectorbt-rust versions and
   `"engine": "rust"` with a positive `rust_calls`; anything else (numba, a
   version mismatch) means the trader would place no orders. Do not "fix" that
   by setting `GGT_VBT_ENGINE=numba`: numba is a research escape hatch, 20-60 s
   of JIT per fresh container, never a trader fallback.
b. **Config:** if the change reads an env var, print the value the trader
   will see through the same code path (e.g.
   `from ggTrader.paper.risk import config_from_env; print(config_from_env())`).
   Remember `.env` changes need no rebuild, but they do need this check.

## 6. Paperwork

- `docs/changelog.md` entry: what changed, why, what the next run should do.
- `docs/next_steps.md`: an item listing what to verify at the next 12:45 run.
- If a deploy decision or finding matters beyond today, save a project memory.

## 7. Next-run verification

After 12:45 PT the next trading day, run the `daily-trader-check` skill and
compare the log (`~/logs/paper_trade_YYYYMMDD.log`) against the prediction
from step 1.4. Mismatches are findings, not noise — investigate before the
following run.

## Rollback

Fastest, no rebuild — select the previous immutable tag recorded in step 4:
```bash
# set GGTRADER_TAG=<previous tag> in .env, then
docker compose up -d ggtrader_live
```
If the old image was pruned locally, every build also pushes `<short_sha>` and
`sha-<full sha>` tags, so `docker pull ghcr.io/garykuepper/ggtrader:<short_sha>` of
the last good commit works too. Never re-tag `:latest` as the rollback.
Then fix forward properly: `git revert <sha>`, push, and run this skill again
so the tag in `.env` names a good build again. For an `.env` change, restore the old line — the
next cron run picks it up.

## Manual broker actions

Sometimes the fix is a one-off trade (e.g. liquidating holdings the strategy
can't exit). Get explicit approval for the exact symbols. Place them with the
Alpaca MCP (`close_position` queues for the open if the market is closed),
then record each in the ledger so the DB matches the broker:
`ggTrader.paper.persist.log_trade(run_date, "SELL", symbol, est_amount,
order_id, reason="manual_<why>")`. Note the action in `docs/next_steps.md` and
verify the fills after the open.
