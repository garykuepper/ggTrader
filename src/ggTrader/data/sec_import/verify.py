"""Check import records, source files, and stored research data."""

import hashlib
import json
import stat
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypedDict

from psycopg2 import Error
from psycopg2.extensions import connection
from psycopg2.extras import Json

from ggTrader.data.kaggle_import.guards import safe_error
from ggTrader.utils.paths import REPO_ROOT

DETAIL_LIMIT = 50
CRSP_SOURCES = ("crsp-daily", "crsp-daily-pilot")
LEGACY_DATASETS = (*CRSP_SOURCES, "mcgill-fiam", "crsp-derived-bars")
STORED_TABLES = {
    **dict.fromkeys(CRSP_SOURCES, "kaggle_crsp.daily"),
    "mcgill-fiam": "kaggle_chars.mcgill",
    "crsp-derived-bars": "market.daily_bars",
    "sec-cache": "sec.fetched_files",
    "sec-headers": "sec.filing_headers",
    "sec-labels": "sec.death_labels",
}


class Check(TypedDict):
    """Result of one check."""

    ok: bool
    detail: str


class Report(TypedDict):
    """Results of one verification run."""

    ok: bool
    checks: dict[str, Check]


def _rows(conn: connection, query: str, params: tuple = ()) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def _detail(items: Sequence[str], total: int | None = None) -> str:
    count = len(items) if total is None else total
    shown = ", ".join(items[:DETAIL_LIMIT])
    return f"{shown} ({count} total)" if shown else "none"


def _result(detail: str, failures: Sequence[str] = ()) -> Check:
    if failures:
        detail += f" Found {_detail(failures)}."
    return {"ok": not failures, "detail": detail}


def load_row_counts_match_log(conn: connection) -> Check:
    """Reconcile the row counts recorded for finished batches and loads."""
    rows = _rows(
        conn,
        """SELECT l.load_id, l.rows_accepted, COALESCE(sum(c.rows), 0), l.rows_quarantined
           FROM ref.loads l LEFT JOIN ref.load_chunks c USING (load_id)
           WHERE l.status = 'complete'
           GROUP BY l.load_id ORDER BY l.load_id""",
    )
    failures = [
        f"load {load_id} logged {accepted}, batches add up to {total}"
        for load_id, accepted, total, _ in rows
        if total != accepted
    ]
    set_aside = {load_id: count for load_id, _, _, count in rows if count}
    return _result(
        f"Compared recorded batch totals for {len(rows)} completed loads. "
        f"Rows set aside as unreadable: {sum(set_aside.values())} "
        f"({_detail([f'load {k}: {v}' for k, v in set_aside.items()])}). "
        "Actual table rows require a deep run.",
        failures,
    )


def _recorded_files(conn: connection) -> list[tuple]:
    return _rows(
        conn,
        """SELECT f.load_id, f.path, f.sha256, f.bytes
           FROM ref.load_files f LEFT JOIN ref.loads l USING (load_id)
           WHERE (l.load_id IS NULL OR l.status = 'complete')
             AND (l.file_path IS NULL OR NOT starts_with(l.file_path, 'load:'))
             AND (l.dataset IS NULL OR NOT (l.dataset = ANY(%s)))
           ORDER BY f.load_id, f.path""",
        (list(LEGACY_DATASETS),),
    )


def _file_size(path: str | None) -> int:
    if not path:
        raise ValueError("source file path is missing")
    info = (REPO_ROOT / path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path} is not a regular file")
    return info.st_size


def _parent_problem(
    load_id: int, dataset: str | None, source: str, accepted: int | None, loads: dict[int, tuple]
) -> str | None:
    reference = source.removeprefix("load:")
    parent = loads.get(int(reference)) if reference.isascii() and reference.isdecimal() else None
    if not parent or parent[0] == load_id or parent[3] != "complete":
        return f"load {load_id} needs a different, completed parent ({source})"
    parent_id, parent_dataset, _, _, parent_rows, _ = parent
    if dataset == "crsp-derived-bars" and parent_dataset not in CRSP_SOURCES:
        return f"load {load_id} needs a CRSP daily parent, got load {parent_id} ({parent_dataset})"
    if dataset == "crsp-derived-bars" and accepted != parent_rows:
        return f"load {load_id} logged {accepted} rows, parent {parent_id} logged {parent_rows}"
    return None


def load_files_exist_with_recorded_size(conn: connection) -> Check:
    """Check file presence and sizes, or the source load for derived data."""
    loads = {
        row[0]: row
        for row in _rows(
            conn,
            """SELECT l.load_id, l.dataset, l.file_path, l.status, l.rows_accepted,
                      EXISTS (SELECT 1 FROM ref.load_chunks c WHERE c.load_id = l.load_id)
               FROM ref.loads l ORDER BY l.load_id""",
        )
    }
    files = _recorded_files(conn)
    loads_with_files = {load_id for load_id, *_ in files}
    failures = []
    for load_id, path, _, expected in files:
        try:
            size = _file_size(path)
            if size != expected:
                failures.append(f"load {load_id} file {path} is {size} bytes, recorded {expected}")
        except (OSError, ValueError) as exc:
            failures.append(f"load {load_id} file {path} could not be checked ({safe_error(exc)})")

    legacy, derived = [], []
    for load_id, dataset, source, status, accepted, has_batches in loads.values():
        if status != "complete":
            continue
        if source and source.startswith("load:"):
            derived.append(f"load {load_id}")
            problem = _parent_problem(load_id, dataset, source, accepted, loads)
            if problem:
                failures.append(problem)
        elif dataset in LEGACY_DATASETS:
            legacy.append(f"load {load_id}")
            try:
                _file_size(source)
            except (OSError, ValueError) as exc:
                failures.append(f"load {load_id} source file check failed ({safe_error(exc)})")
        elif (
            dataset
            and dataset.startswith("sec-")
            and (accepted or has_batches)
            and load_id not in loads_with_files
        ):
            failures.append(f"load {load_id} has no source file records")

    return _result(
        f"Checked presence and size for {len(files)} recorded files. "
        f"Checked parent links for completed derived loads ({_detail(derived)}). "
        f"Checked file presence for older loads with no recorded size or fingerprint "
        f"per file ({_detail(legacy)}). File contents require a deep run.",
        failures,
    )


def no_incomplete_loads(conn: connection) -> Check:
    """Report unfinished loads and distinguish them from closed unsuccessful loads."""
    rows = _rows(
        conn,
        """SELECT load_id, status FROM ref.loads
           WHERE status IN ('running', 'failed', 'withdrawn') ORDER BY load_id""",
    )
    running = [f"load {load_id} still running" for load_id, status in rows if status == "running"]
    ended = [f"load {load_id} ({status})" for load_id, status in rows if status != "running"]
    return _result(f"Looked for unfinished loads. Failed or withdrawn ({_detail(ended)}).", running)


def every_complete_load_has_one_coverage_row(conn: connection) -> Check:
    """Require one record describing what each completed load covers."""
    rows = _rows(
        conn,
        """SELECT l.load_id, count(c.load_id), l.dataset
           FROM ref.loads l LEFT JOIN ref.load_coverage c USING (load_id)
           WHERE l.status = 'complete'
           GROUP BY l.load_id, l.dataset ORDER BY l.load_id""",
    )
    # The older market-data loaders do not write a label themselves (the database allows that);
    # for them one label is expected after the schema upgrade, but a missing one is not an error.
    failures = [
        f"load {load_id} has {count} coverage records"
        for load_id, count, dataset in rows
        if count > 1 or (count == 0 and dataset not in LEGACY_DATASETS)
    ]
    return _result(
        f"Checked that each of {len(rows)} completed loads has one record describing which data "
        "it covers (older market-data loads may have none). Whether that description is right "
        "is not checked.",
        failures,
    )


def labels_match_stock_spells_in_release(conn: connection) -> Check:
    """Match each label to the stock identifier and last date in its named release."""
    rows = _rows(
        conn,
        """SELECT d.load_id, d.release_id, d.permno, d.last_date, count(*) OVER ()
           FROM sec.death_labels d JOIN ref.loads dl ON dl.load_id = d.load_id
           WHERE dl.status = 'complete' AND NOT EXISTS (
               SELECT 1 FROM market.stock_spans s
               WHERE s.release_id = d.release_id AND s.permno = d.permno
                 AND s.last_date = d.last_date
           )
           ORDER BY d.load_id, d.release_id, d.permno, d.last_date
           LIMIT %s""",
        (DETAIL_LIMIT,),
    )
    [(total,)] = _rows(
        conn,
        "SELECT count(*) FROM sec.death_labels d JOIN ref.loads dl USING (load_id) "
        "WHERE dl.status = 'complete'",
    )
    detail = (
        f"Matched {total} study labels to stock identifiers and last trading dates in their "
        "releases (nothing was checked if there are none). It does not check the labels' meaning."
    )
    if rows:
        missing = [
            f"load {load_id}, release {release}, stock {permno}, last date {last_date}"
            for load_id, release, permno, last_date, _ in rows
        ]
        detail += f" Could not match {_detail(missing, rows[0][-1])}."
    return {"ok": not rows, "detail": detail}


def cited_accessions_exist(conn: connection) -> Check:
    """Find cited filing identifiers in the headers or in one pass of the index.

    Only rows of completed loads count, on both sides: a half-finished or failed load must not
    make a citation look supported, and its labels are not part of the data in use.
    """
    rows = _rows(
        conn,
        """SELECT DISTINCT a.accession
           FROM sec.death_labels d
           JOIN ref.loads l ON l.load_id = d.load_id AND l.status = 'complete'
           LEFT JOIN LATERAL unnest(d.accessions) a(accession) ON true""",
    )
    if not rows:
        return _result("no labels loaded")
    pending = {accession for (accession,) in rows if accession is not None}
    cited = len(pending)
    if pending:
        headers = _rows(
            conn,
            """SELECT DISTINCT h.accession FROM sec.filing_headers h
               JOIN ref.loads l ON l.load_id = h.load_id AND l.status = 'complete'
               WHERE h.accession = ANY(%s)""",
            (sorted(pending),),
        )
        pending.difference_update(accession for (accession,) in headers)
    if pending:
        index = _rows(
            conn,
            # Written so that the expression index index_entries_accession_idx answers it.
            r"""SELECT DISTINCT accession FROM (
                    SELECT substring(e.filename from '/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$')
                           AS accession
                    FROM sec.index_entries e
                    JOIN ref.loads l ON l.load_id = e.load_id AND l.status = 'complete'
                    WHERE substring(e.filename from '/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$')
                          = ANY(%s)
                ) filings""",
            (sorted(pending),),
        )
        pending.difference_update(accession for (accession,) in index)
    detail = (
        f"Looked for {cited} cited filing identifiers in headers, then in quarterly index "
        "filenames; it does not check that a filing supports its label."
    )
    if not cited:
        detail = "No loaded label cites a filing identifier, so nothing was looked up."
    if pending:
        detail += f" Could not find {_detail(sorted(pending))}."
    return {"ok": not pending, "detail": detail}


def load_files_rehash(conn: connection) -> Check:
    """Compare recorded source files with fresh SHA-256 fingerprints of their bytes."""
    files = _recorded_files(conn)
    digests: dict[str, str] = {}
    failures = []
    for load_id, path, expected, _ in files:
        try:
            if path not in digests:
                _file_size(path)
                with (REPO_ROOT / path).open("rb") as source:
                    digests[path] = hashlib.file_digest(source, "sha256").hexdigest()
            if digests[path] != expected:
                failures.append(f"load {load_id} file {path} has a different fingerprint")
        except (OSError, ValueError) as exc:
            failures.append(f"load {load_id} file {path} could not be read ({safe_error(exc)})")
    return _result(
        f"Compared the recorded fingerprint with the contents of {len(files)} file records. "
        "Older sources and loads derived from another load have no file fingerprints checked here.",
        failures,
    )


def _index_counts(conn: connection, load_id: int, failures: list[str]) -> int:
    actual = dict(
        _rows(
            conn,
            "SELECT quarter, count(*) FROM sec.index_entries WHERE load_id = %s GROUP BY quarter",
            (load_id,),
        )
    )
    logged = dict(
        _rows(conn, "SELECT chunk_key, rows FROM ref.load_chunks WHERE load_id = %s", (load_id,))
    )
    for quarter in sorted(actual.keys() | logged.keys(), key=str):
        count, expected = actual.get(quarter, 0), logged.get(quarter, 0)
        if count != expected:
            failures.append(f"load {load_id} quarter {quarter} has {count} rows, logged {expected}")
    return sum(actual.values())


def _changed_label_rows(conn: connection, load_id: int) -> list[str]:
    """List labels whose stored fields no longer hash to the fingerprint saved at load time."""
    changed = []
    for permno, last_date, fields, saved in _rows(
        conn,
        "SELECT permno, last_date, source_fields, source_row_sha256 FROM sec.death_labels "
        "WHERE load_id = %s ORDER BY permno, last_date",
        (load_id,),
    ):
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != saved:
            changed.append(f"stock {permno} last date {last_date}")
    return changed


def _cache_fingerprint(conn: connection, load_id: int) -> str:
    rows = _rows(
        conn,
        "SELECT url_hash, content_sha256 FROM sec.fetched_files WHERE load_id = %s",
        (load_id,),
    )
    # Match cache.manifest_sha256, including its order and lack of a trailing newline.
    manifest = "\n".join(sorted(f"{url_hash}:{content}" for url_hash, content in rows))
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def content_sha256_matches_stored_rows(conn: connection) -> Check:
    """Count the actual stored rows and rebuild the saved-response list's fingerprint."""
    loads = _rows(
        conn,
        """SELECT load_id, dataset, rows_accepted, content_sha256 FROM ref.loads
           WHERE status = 'complete' ORDER BY load_id""",
    )
    compared, fingerprints, skips, failures = [], [], [], []
    for load_id, dataset, expected, digest in loads:
        name = f"load {load_id} ({dataset})"
        if dataset == "sec-index":
            actual = _index_counts(conn, load_id, failures)
            compared.append(f"{name} in sec.index_entries, also by quarter")
        else:
            table = STORED_TABLES.get((dataset or "").partition(":")[0])
            if table is None:
                failures.append(f"{name} has no known table to count")
                continue
            query = f"SELECT count(*) FROM {table} WHERE load_id = %s"  # noqa: S608 — fixed names
            [(actual,)] = _rows(conn, query, (load_id,))
            compared.append(f"{name} in {table}")
        if actual != expected:
            failures.append(f"{name} has {actual} stored rows, logged {expected}")

        if (dataset or "").startswith("sec-labels"):
            changed = _changed_label_rows(conn, load_id)
            compared[-1] += ", each label row against its own fingerprint"
            if changed:
                failures.append(
                    f"{name} has label rows that differ from their fingerprint: {_detail(changed)}"
                )
            skips.append(f"{name}, file fingerprint is covered by load_files_rehash")
        elif dataset == "sec-cache":
            fingerprints.append(name)
            if _cache_fingerprint(conn, load_id) != digest:
                failures.append(f"{name} has a different fingerprint for its saved response list")
        else:
            reason = (
                "hash is of the source file list"
                if dataset in ("sec-index", "sec-headers")
                else "hash is copied from its source load"
                if dataset == "crsp-derived-bars"
                else "hash is of the source file"
            )
            skips.append(f"{name}, {reason}")

    return _result(
        f"Counted stored rows for {_detail(compared)}. "
        f"Rebuilt saved-response fingerprints for {_detail(fingerprints)}. "
        f"Skipped stored-value comparisons for {_detail(skips)}. "
        "See load_files_rehash for source files with a recorded fingerprint.",
        failures,
    )


ORDINARY_CHECKS: tuple[Callable[[connection], Check], ...] = (
    load_row_counts_match_log,
    load_files_exist_with_recorded_size,
    no_incomplete_loads,
    every_complete_load_has_one_coverage_row,
    labels_match_stock_spells_in_release,
    cited_accessions_exist,
)
DEEP_CHECKS = (load_files_rehash, content_sha256_matches_stored_rows)


def run_checks(conn: connection, *, deep: bool = False) -> Report:
    """Run all requested checks, keeping later checks usable after a database error."""
    checks = {}
    for check in ORDINARY_CHECKS + (DEEP_CHECKS if deep else ()):
        try:
            checks[check.__name__] = check(conn)
        except Exception as exc:  # noqa: BLE001 — one check must not hide results of the others
            with suppress(Error):
                conn.rollback()
            detail = " ".join(safe_error(exc).splitlines())
            checks[check.__name__] = {"ok": False, "detail": detail}
    with suppress(Error):
        conn.rollback()  # end the read-only transaction so no snapshot is held
    return {"ok": all(result["ok"] for result in checks.values()), "checks": checks}


def record(conn: connection, report: Report, mode: str) -> int:
    """Insert and commit one verification report."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.integrity_checks (mode, ok, report) VALUES (%s, %s, %s) "
            "RETURNING check_id",
            (mode, report["ok"], Json(report)),
        )
        [check_id] = cur.fetchone()
    conn.commit()
    return check_id
