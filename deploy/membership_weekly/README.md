# Weekly S&P 500 membership job: install and run book

**Status: TEMPLATES ONLY. The timer is NOT installed and NOT enabled.** It may be enabled only
after (1) three manual dry runs matched expectations, (2) the `DryRunNotifier` tests pass,
(3) the restore drill passed, and (4) the owner says go. Nothing here has been deployed.

## What the job does (spec Appendix C1)

`ggt db membership weekly` once a week:

1. fetches and archives the raw bytes of the SPDR SPY holdings workbook, the Wikipedia change
   ledger and current list (MediaWiki API, descriptive User-Agent) and the two chinobing CSVs
   (`universe.check_blobs` for SPY, `universe.source_snapshots` for the rest; about 7 requests);
2. matches the latest vintage against SPY on the holdings' own date. A match writes a `matched`
   row in `universe.checks`: this is what keeps the live trader's membership gate open;
3. compares the other sources with our list (disagreements are warnings only);
4. on any refusal builds a **proposed events file**, keeps its exact bytes in the database
   (`universe.source_snapshots`, source `proposal`), also tries to write a copy under
   `data/exports/membership_proposals/` (git-ignored) and tells you what to check.
   **It never loads a vintage.**

Refusal conditions: our list differs from SPY; the SPY file is missing, stale (over 7 days) or
unusable; a ledger row after what is already attested is dated in the future or on its
announcement date; an unpaired removal would leave 480..520 members; a source is missing, has not
changed for 14 days, has an unreadable last-change date or has changed structure.

## Prerequisites

- The schema is current: `ggt db membership apply-ddl` (adds `universe.source_snapshots`; safe to
  repeat; only schema `universe` changes).
- Telegram settings (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) in the compose `.env`.
- The image contains this code: push, let `docker-build.yml` publish, `docker compose pull`
  (see the `ggtrader-deploy` skill). Until then use the host-venv variant below.
- The unit runs `docker compose run --rm`, and the copy of a proposal under `data/` is only a
  convenience (it depends on the container's bind mount and is gone if the write fails). The
  message therefore names `ggt db membership proposal`, which rebuilds the file from the database.

## Dry runs (no Telegram; they DO record a check row)

```bash
docker compose run --rm -T ggtrader_live python -u ggt.py db membership weekly
# or, host virtualenv:
.venv/bin/python ggt.py db membership weekly
```

Exit 0 = clean (or warnings only), 2 = a refusal or a proposal, 1 = the job failed.
The message that would be sent is printed. `--notify` is the only way a message is sent.

## Enable (only after the owner's go)

```bash
cp deploy/membership_weekly/ggtrader-membership-weekly.{service,timer} ~/.config/systemd/user/
sed -i "s|__REPO_ROOT__|$PWD|" ~/.config/systemd/user/ggtrader-membership-weekly.service
systemctl --user daemon-reload
systemctl --user enable --now ggtrader-membership-weekly.timer
systemctl --user list-timers ggtrader-membership-weekly.timer
```

Host-venv alternative: in the copied service replace the `ExecStart` line by
`ExecStart=__REPO_ROOT__/.venv/bin/python -u ggt.py db membership weekly --notify`
(the shell must be able to reach the database and `.env`; the tests never read `.env`).

## Disable

```bash
systemctl --user disable --now ggtrader-membership-weekly.timer
rm ~/.config/systemd/user/ggtrader-membership-weekly.{service,timer}
systemctl --user daemon-reload
```

With the timer off, the live gate refuses after 45 days without a `matched` check
(10 days once the job is the owner's chosen routine): run `ggt db membership check-holdings` by hand.

## Reading the Telegram message

- **One short line "OK. Load N equals the SPY holdings as of DATE ..."**: a heartbeat. Nothing to do.
- **"note"**: the list matches SPY but another source disagrees (warning). Look, no urgency.
- **"ACTION NEEDED"**: a refusal. The message lists each reason, the sources with their
  last-change dates, and, if there are candidate changes, the proposal rows. `CONFIRM` marks a
  row whose date is not agreed by two independent sources.
- **"JOB ERROR"**: the job could not read the database or could not record its result.
- A silent Saturday is also a signal: check `systemctl --user status ggtrader-membership-weekly`.

## When it proposes a change

1. Write the file from the database, using the check id in the message (read-only on the
   database; it refuses to overwrite a file you have already edited):

```bash
ggt db membership proposal --id N --out proposed_events.csv
```

   Columns are the events-file columns plus `date_source` and `needs_confirmation`.
2. For every row, confirm the effective date in the **S&P DJI press release**
   (press.spglobal.com; search the company). Ledger dates may be the announcement date and
   chinobing dates are observation dates (a day before to two days after). Fix `effective_date`.
3. Fill `evidence_url`; save the page and put its sha256 in `evidence_sha256` (put the saved page
   in a folder and pass `--evidence-dir`); fill `announcement_date`; delete wrong rows; keep a
   `rename` row only if the press release says it is the same company.
4. Set `needs_confirmation` to `N` on each confirmed row. The loader refuses a `Y` row and a
   blank `evidence_url`, so an unreviewed proposal cannot be loaded.
5. Dry run, read the per-date adds/removes and counts, then commit:

```bash
ggt db load-sp500 --from-events proposed_events.csv --evidence-dir DIR
ggt db load-sp500 --from-events proposed_events.csv --evidence-dir DIR --commit
```

6. `ggt db membership check-holdings` (or wait for the next Saturday run) must now say `matched`.

## Restore procedure (schema `universe`)

The dump is `pg_dump --schema=universe --format=custom` (`ggt db membership export`, files in
`data/exports/universe/`, git-ignored). To restore into an empty database:

```bash
psql DBNAME -c 'CREATE EXTENSION btree_gist'          # the dump does not carry the extension
pg_restore --no-owner --dbname=DBNAME data/exports/universe/universe_<UTC>.dump
ggt db membership verify --load-id N                  # for every load id
```

Rehearse it without touching the real data: `python scripts/membership_restore_drill.py`
(dumps, restores into a scratch database, compares every table's row count and every load's
hashes, runs `verify` for every load, drops the scratch database).
