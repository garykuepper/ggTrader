"""Driver for dead-stock-early-filings: pilot or full SEC classification of 2000-2004 deaths."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ggTrader.utils.binding import binding, git_sha
from ggTrader.utils.io import atomic_write_text, write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import early_classify as ec  # noqa: E402  (found on sys.path through bootstrap)
import early_inputs  # noqa: E402
import early_names  # noqa: E402
import early_roster  # noqa: E402
import early_sec  # noqa: E402
import ef_dead_stock_first_test as fp  # noqa: E402
import pandas as pd  # noqa: E402
from ef_sec_client import SecNotFound  # noqa: E402

STUDY_DIR = study_dir(__file__)
STUDY_ID = "dead-stock-early-filings"
WINDOW = "early"
PIN_RELEASE_ID = 1
PIN_CRSP_LOAD = 1
PILOT_SEED = ec.PILOT_SEED
DEFAULT_PILOT_SIZE = 20
EXPECTED_ROSTER_ROWS = 251
EXPECTED_HELD_BY_RULE = 33
SEC_KEPT_FILING_END_QUARTER: tuple[int, int] = (2006, 4)
EXIT_REFUSED = 2
EXIT_CHECK_FAILED = 3
_HEADER_SCAN_BYTES = 262_144


def build_parser() -> argparse.ArgumentParser:
    """Command line: pilot or full mode, pinned release/load, optional output and cache paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="number of seeded roster rows in pilot mode (default 20)",
    )
    parser.add_argument("--release-id", type=int, required=True, help="pinned release (1)")
    parser.add_argument("--crsp-load", type=int, required=True, help="pinned CRSP load (1)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results JSON (default: _pilot.json or _early.json in study dir)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=early_sec.DEFAULT_CACHE_DIR,
        help="directory for cached SEC responses",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256() -> str:
    """One hash over the study's own code files and frozen copy records (tests excluded)."""
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for record_name in ("frozen_copies.json", "frozen_sec_copies.json"):
        record_path = here / record_name
        if record_path.is_file():
            digest.update(record_name.encode())
            digest.update(record_path.read_bytes())
    return digest.hexdigest()


def _quarter_start(year: int, quarter: int) -> date:
    return date(year, (quarter - 1) * 3 + 1, 1)


def _decode_header_text(raw_bytes: bytes) -> str:
    """Decode only the header prefix of a large EDGAR text filing when its closing tag is early."""
    prefix = raw_bytes[:_HEADER_SCAN_BYTES]
    lower = prefix.lower()
    if b"</sec-header>" in lower or b"<document>" in lower:
        return prefix.decode("utf-8", errors="replace")
    return raw_bytes.decode("utf-8", errors="replace")


def _in_window(filed: date, last_day: date) -> bool:
    return (
        last_day - timedelta(days=early_sec.BEFORE_DAYS)
        <= filed
        <= last_day + timedelta(days=early_sec.AFTER_DAYS)
    )


def _rank_candidate_cik(
    cik: int,
    last_day: date,
    cik_event_rows: dict[int, list[tuple[early_sec.IndexRow, str]]],
    cik_all_dates: dict[int, set[date]],
) -> tuple[int, int, int, int]:
    """Deterministic preference: in-window event filing, any window filing, closest date, CIK."""
    events = cik_event_rows.get(cik, ())
    has_event = any(_in_window(row.filed, last_day) for row, _conf in events)
    dates = cik_all_dates.get(cik, set())
    has_window_any = any(_in_window(d, last_day) for d in dates)
    min_dist = min((abs((d - last_day).days) for d in dates), default=10**9)
    return (-int(has_event), -int(has_window_any), min_dist, cik)


class SecLookup:
    """Cached, streaming SEC master-index search and header lookup for one named roster.

    Streams one quarterly ``master.zip`` archive at a time through ``EarlySecClient`` and keeps
    only rows of matched CIKs in memory. Fetches and parses filing headers for in-window
    8-K / 8-K/A / Form 25-type filings, then runs a single second pass for former names
    learned from those headers (tier ``former_name``, never promoted to ``checked``).
    """

    def __init__(
        self,
        client: early_sec.EarlySecClient,
        named_roster: pd.DataFrame,
        *,
        end_quarter: tuple[int, int] = SEC_KEPT_FILING_END_QUARTER,
    ) -> None:
        self._client = client
        self._primary_clean_to_ciks: dict[str, set[int]] = defaultdict(set)
        self._former_clean_to_header_ciks: dict[str, set[int]] = defaultdict(set)
        self._former_index_clean_to_ciks: dict[str, set[int]] = defaultdict(set)
        self._cik_learned_formers: dict[int, set[str]] = defaultdict(set)
        self._cik_company_names: dict[int, str] = {}
        self._cik_all_dates: dict[int, set[date]] = defaultdict(set)
        self._cik_event_rows: dict[int, list[tuple[early_sec.IndexRow, str]]] = defaultdict(list)
        self._seen_event_keys: set[tuple[int, str]] = set()
        self._headers: dict[str, early_sec.FilingHeader | None] = {}

        targets: list[tuple[str, date]] = []
        target_clean_names: set[str] = set()
        if not named_roster.empty and "company_name" in named_roster.columns:
            for _, row in named_roster.iterrows():
                raw_name = row.get("company_name")
                if raw_name is None or pd.isna(raw_name):
                    continue
                cleaned = early_sec.clean_company_name(str(raw_name))
                if not cleaned:
                    continue
                last_day = pd.Timestamp(row["last_bar"]).date()
                targets.append((cleaned, last_day))
                target_clean_names.add(cleaned)

        if not targets:
            self._quarters: list[tuple[int, int]] = []
            return

        all_row_quarters = [q for _, d in targets for q in early_sec.quarters_for(pd.Timestamp(d))]
        min_q = min(all_row_quarters)
        max_q = max(end_quarter, max(all_row_quarters))
        self._quarters = early_sec.quarters_covering(_quarter_start(*min_q), _quarter_start(*max_q))

        # Pass 1: stream one quarterly archive at a time; keep only matched CIKs.
        matched_ciks: set[int] = set()
        for year, quarter in self._quarters:
            self._scan_quarter(
                year,
                quarter,
                target_names=target_clean_names,
                name_to_ciks=self._primary_clean_to_ciks,
                tracked_ciks=matched_ciks,
                row_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            )

        # Fetch headers for in-window event filings of Pass-1 matched companies and collect
        # former names declared in those headers.
        search_formers: set[str] = set()
        for cleaned, last_day in targets:
            for cik in sorted(self._primary_clean_to_ciks.get(cleaned, ())):
                for ev_row, _row_conf in self._cik_event_rows.get(cik, ()):
                    if not _in_window(ev_row.filed, last_day):
                        continue
                    header = self._fetch_header(ev_row.file_path)
                    if header is None:
                        continue
                    for former in header.former_names:
                        clean_former = early_sec.clean_company_name(former)
                        if not clean_former:
                            continue
                        self._former_clean_to_header_ciks[clean_former].add(cik)
                        if clean_former not in self._primary_clean_to_ciks:
                            self._cik_learned_formers[cik].add(clean_former)
                            if clean_former not in target_clean_names:
                                search_formers.add(clean_former)

        # Pass 2: run once over the cached archives if headers revealed new former names to search.
        if search_formers:
            pass2_ciks: set[int] = set()
            for year, quarter in self._quarters:
                self._scan_quarter(
                    year,
                    quarter,
                    target_names=search_formers,
                    name_to_ciks=self._former_index_clean_to_ciks,
                    tracked_ciks=pass2_ciks,
                    row_confidence=early_sec.CONFIDENCE_FORMER_NAME,
                )
            for cleaned, last_day in targets:
                candidate_ciks: set[int] = set()
                for primary_cik in self._primary_clean_to_ciks.get(cleaned, ()):
                    for cfn in self._cik_learned_formers.get(primary_cik, ()):
                        candidate_ciks.update(self._former_index_clean_to_ciks.get(cfn, ()))
                for cik in sorted(candidate_ciks):
                    for ev_row, _row_conf in self._cik_event_rows.get(cik, ()):
                        if _in_window(ev_row.filed, last_day):
                            self._fetch_header(ev_row.file_path)

    def _scan_quarter(
        self,
        year: int,
        quarter: int,
        *,
        target_names: set[str],
        name_to_ciks: dict[str, set[int]],
        tracked_ciks: set[int],
        row_confidence: str,
    ) -> None:
        url = early_sec.master_index_url(year, quarter)
        try:
            archive_bytes = self._client.get(url)
        except SecNotFound:
            return
        name_cache: dict[str, str] = {}
        for row in early_sec.iter_master_index(archive_bytes):
            raw_name = row.company_name
            if raw_name:
                cleaned = name_cache.get(raw_name)
                if cleaned is None:
                    cleaned = early_sec.clean_company_name(raw_name)
                    name_cache[raw_name] = cleaned
                if cleaned and cleaned in target_names:
                    name_to_ciks[cleaned].add(row.cik)
                    tracked_ciks.add(row.cik)
                    self._cik_company_names.setdefault(row.cik, raw_name)
            if row.cik in tracked_ciks:
                self._cik_all_dates[row.cik].add(row.filed)
                if row.form.upper() in early_sec.RELEVANT_EVENT_FORMS:
                    key = (row.cik, row.file_path)
                    if key not in self._seen_event_keys:
                        self._seen_event_keys.add(key)
                        self._cik_event_rows[row.cik].append((row, row_confidence))

    def _fetch_header(self, file_path: str) -> early_sec.FilingHeader | None:
        if file_path in self._headers:
            return self._headers[file_path]
        url = early_sec.filing_url(file_path)
        try:
            raw = self._client.get(url)
        except SecNotFound:
            self._headers[file_path] = None
            return None
        header = early_sec.parse_filing_header(_decode_header_text(raw))
        self._headers[file_path] = header
        return header

    def _build_evidence(
        self, cik: int, last_day: date, match_confidence: str
    ) -> ec.CompanyEvidence:
        rows = sorted(
            [
                (r, conf)
                for r, conf in self._cik_event_rows.get(cik, ())
                if _in_window(r.filed, last_day)
            ],
            key=lambda item: (item[0].filed, item[0].form, item[0].file_path),
        )
        filings: list[ec.FilingEvidence] = []
        for row, row_conf in rows:
            effective_conf = (
                early_sec.CONFIDENCE_FORMER_NAME
                if early_sec.CONFIDENCE_FORMER_NAME in (match_confidence, row_conf)
                else match_confidence
            )
            filing = early_sec.Filing(
                form=row.form,
                company_name=row.company_name,
                cik=row.cik,
                filed=row.filed,
                file_path=row.file_path,
                confidence=effective_conf,
            )
            filings.append(
                ec.FilingEvidence(filing=filing, header=self._fetch_header(row.file_path))
            )
        return ec.CompanyEvidence(
            cik=cik,
            company_name=self._cik_company_names.get(cik, ""),
            match_confidence=match_confidence,
            window_filings=tuple(filings),
            all_filing_dates=tuple(sorted(self._cik_all_dates.get(cik, ()))),
        )

    def _has_qualifying_event(self, evidence: ec.CompanyEvidence, last_day: date) -> bool:
        eight_k_headers = [
            ev.header
            for ev in evidence.window_filings
            if ev.filing.form.upper() in early_sec.FORM8K_FORMS and ev.header is not None
        ]
        has_form25 = any(
            ev.filing.form.upper() in early_sec.FORM25_FORMS for ev in evidence.window_filings
        )
        label = early_sec.classify_company_events(
            eight_k_headers,
            has_form25=has_form25,
            filing_dates=evidence.all_filing_dates,
            last_bar=last_day,
        )
        return label in (
            early_sec.LABEL_BANKRUPTCY,
            early_sec.LABEL_ACQUISITION_TYPE,
            early_sec.LABEL_DELISTING_NOTICE_ONLY,
        )

    def lookup(
        self, company_name: str, last_trading_day: date | pd.Timestamp
    ) -> ec.CompanyEvidence | None:
        """Return ``CompanyEvidence`` for ``company_name`` around ``last_trading_day``, or None."""
        cleaned = early_sec.clean_company_name(company_name)
        if not cleaned:
            return None
        last_day = (
            last_trading_day.date()
            if isinstance(last_trading_day, pd.Timestamp)
            else last_trading_day
        )

        primary_ciks = self._primary_clean_to_ciks.get(cleaned)
        if primary_ciks:
            best_cik = min(
                primary_ciks,
                key=lambda c: _rank_candidate_cik(
                    c, last_day, self._cik_event_rows, self._cik_all_dates
                ),
            )
            primary_ev = self._build_evidence(best_cik, last_day, early_sec.CONFIDENCE_NAME_ONLY)
            if self._has_qualifying_event(primary_ev, last_day):
                return primary_ev
            # Try former names learned from this company's own headers (Pass 2).
            former_ciks: set[int] = set()
            for cfn in self._cik_learned_formers.get(best_cik, ()):
                former_ciks.update(self._former_index_clean_to_ciks.get(cfn, ()))
            former_ciks.discard(best_cik)
            if former_ciks:
                best_former = min(
                    former_ciks,
                    key=lambda c: _rank_candidate_cik(
                        c, last_day, self._cik_event_rows, self._cik_all_dates
                    ),
                )
                former_ev = self._build_evidence(
                    best_former, last_day, early_sec.CONFIDENCE_FORMER_NAME
                )
                if self._has_qualifying_event(former_ev, last_day):
                    return former_ev
            return primary_ev

        # No primary match: check if this stock's cleaned name was declared as a former name.
        matched_former_ciks = self._former_clean_to_header_ciks.get(
            cleaned, set()
        ) | self._former_index_clean_to_ciks.get(cleaned, set())
        if matched_former_ciks:
            best_cik = min(
                matched_former_ciks,
                key=lambda c: _rank_candidate_cik(
                    c, last_day, self._cik_event_rows, self._cik_all_dates
                ),
            )
            return self._build_evidence(best_cik, last_day, early_sec.CONFIDENCE_FORMER_NAME)

        return None

    def __call__(
        self, company_name: str, last_trading_day: date | pd.Timestamp
    ) -> ec.CompanyEvidence | None:
        return self.lookup(company_name, last_trading_day)

    def stats(self) -> dict[str, int]:
        """Return SEC network vs cache counters from the underlying ``EarlySecClient``."""
        return _client_stats(self._client)


def _client_stats(client: early_sec.EarlySecClient | None) -> dict[str, int]:
    if client is None:
        return {
            "network_requests": 0,
            "cache_hits": 0,
            "network_fetches_200": 0,
            "cache_hits_200": 0,
            "not_found_recorded": 0,
            "not_found_cache_hits": 0,
            "corrupt_cache_refetches": 0,
        }
    return {
        "network_requests": int(client.network_fetches + client.not_found_recorded),
        "cache_hits": int(client.cache_hits + client.not_found_hits),
        "network_fetches_200": int(client.network_fetches),
        "cache_hits_200": int(client.cache_hits),
        "not_found_recorded": int(client.not_found_recorded),
        "not_found_cache_hits": int(client.not_found_hits),
        "corrupt_cache_refetches": int(client.corrupt_cache_refetches),
    }


def _attach_roster_names(
    working_roster: pd.DataFrame,
    inputs: dict[str, Any],
    name_loader: Callable[[pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    if name_loader is not None:
        return ec.merge_names(working_roster, name_loader(working_roster))
    if inputs.get("names") is not None:
        return ec.merge_names(working_roster, inputs["names"])
    if inputs.get("mcgill_conn") is not None:
        return early_names.attach_names(inputs["mcgill_conn"], working_roster)
    from ggTrader.data.bars import connect_readonly

    conn = connect_readonly()
    try:
        return early_names.attach_names(conn, working_roster)
    finally:
        conn.close()


def _resolve_sec_client(
    args: argparse.Namespace,
    *,
    sec_client: early_sec.EarlySecClient | None,
    sec_lookup_factory: Callable[[pd.DataFrame, early_sec.EarlySecClient | None], Any] | None,
    contact_loader: Callable[[], str] | None,
) -> early_sec.EarlySecClient | None:
    if sec_client is not None:
        return sec_client
    if contact_loader is not None:
        contact = contact_loader()
        return early_sec.EarlySecClient(
            user_agent=contact,
            cache_dir=args.cache_dir,
            min_interval=early_sec.DEFAULT_MIN_INTERVAL,
        )
    if sec_lookup_factory is not None:
        return None
    contact = early_sec.load_contact_line()
    return early_sec.EarlySecClient(
        user_agent=contact,
        cache_dir=args.cache_dir,
        min_interval=early_sec.DEFAULT_MIN_INTERVAL,
    )


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    name_loader: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    sec_client: early_sec.EarlySecClient | None = None,
    sec_lookup_factory: Callable[[pd.DataFrame, early_sec.EarlySecClient | None], Any]
    | None = None,
    git_dirty: Callable[[], bool] | None = None,
    contact_loader: Callable[[], str] | None = None,
) -> int:
    """Run pilot or full early-window SEC classification and write JSON + CSV with provenance.

    Everything except ``argv`` is injectable for tests. Exit codes: 0 done, 2 refused before
    work (wrong pins, dirty tree, existing output, drift), 3 full-mode roster check failed.
    """
    command = list(sys.argv if argv is None else ["run.py", *argv])
    args = build_parser().parse_args(argv)

    if args.release_id != PIN_RELEASE_ID or args.crsp_load != PIN_CRSP_LOAD:
        print("refused: the release and CRSP load must be the pinned 1 and 1", file=sys.stderr)
        return EXIT_REFUSED
    if args.mode == "pilot" and args.pilot_size <= 0:
        print("refused: --pilot-size must be a positive integer", file=sys.stderr)
        return EXIT_REFUSED

    dirty = git_dirty() if git_dirty is not None else (loader is None and fp.git_tree_dirty(ROOT))
    if dirty:
        print("refused: the tree has uncommitted changes to tracked files", file=sys.stderr)
        return EXIT_REFUSED

    default_json_name = "_pilot.json" if args.mode == "pilot" else "_early.json"
    out_path = args.out or STUDY_DIR / default_json_name
    csv_name = "_pilot_labels.csv" if args.mode == "pilot" else "_early_labels.csv"
    csv_path = out_path.parent / csv_name

    if out_path.exists():
        print(f"refused: {out_path} already exists", file=sys.stderr)
        return EXIT_REFUSED
    if csv_path.exists():
        print(f"refused: {csv_path} already exists", file=sys.stderr)
        return EXIT_REFUSED

    start, end = early_inputs.WINDOWS[WINDOW]
    started = time.time()
    inputs = (
        loader(args)
        if loader is not None
        else early_inputs.read_inputs(WINDOW, args.release_id, args.crsp_load)
    )
    drifted = early_inputs.drift_in(inputs)
    if drifted:
        print(f"refused: Yahoo drift in the pinned release: {drifted[:5]}", file=sys.stderr)
        return EXIT_REFUSED

    full_roster = early_roster.build_death_roster(
        inputs["close"],
        inputs["eligible"],
        inputs["span_rows"],
        start=start,
        end=end,
        release_spans=inputs.get("release_spans"),
    )
    full_count = int(len(full_roster))
    full_held = int(full_roster["held_by_rule"].sum())

    if args.mode == "full" and (
        full_count != EXPECTED_ROSTER_ROWS or full_held != EXPECTED_HELD_BY_RULE
    ):
        print(
            f"check failed: expected {EXPECTED_ROSTER_ROWS} roster rows and "
            f"{EXPECTED_HELD_BY_RULE} held by the rule, got {full_count} rows and "
            f"{full_held} held",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED

    if args.mode == "pilot":
        working_roster = ec.select_pilot_rows(
            full_roster, pilot_size=args.pilot_size, seed=PILOT_SEED
        )
    else:
        working_roster = full_roster

    named_roster = _attach_roster_names(working_roster, inputs, name_loader)
    client = _resolve_sec_client(
        args,
        sec_client=sec_client,
        sec_lookup_factory=sec_lookup_factory,
        contact_loader=contact_loader,
    )
    if sec_lookup_factory is not None:
        sec_lookup = sec_lookup_factory(named_roster, client)
    else:
        assert client is not None
        sec_lookup = SecLookup(client, named_roster)

    classified = ec.classify_roster(working_roster, named_roster, sec_lookup)
    summary = ec.summarize_classifications(classified)
    if hasattr(sec_lookup, "stats") and callable(sec_lookup.stats):
        sec_requests = sec_lookup.stats()
    else:
        sec_requests = _client_stats(client)

    csv_text = classified.to_csv(index=False)
    atomic_write_text(csv_path, csv_text)

    data_prov = inputs.get("data_provenance") or {}
    candidates_prov = data_prov.get("candidates") if isinstance(data_prov, dict) else None
    spy_prov = data_prov.get("spy") if isinstance(data_prov, dict) else None
    data_release = (candidates_prov or {}).get("data_release")
    bars_sha256 = (candidates_prov or {}).get("bars_sha256")

    roster_counts: dict[str, Any] = {
        "full_roster_rows": full_count,
        "full_held_by_rule": full_held,
        "expected_roster_rows": EXPECTED_ROSTER_ROWS,
        "expected_held_by_rule": EXPECTED_HELD_BY_RULE,
        "selected_rows": int(len(working_roster)),
        "selected_held_by_rule": int(working_roster["held_by_rule"].sum()),
    }

    provenance_block = {
        **binding(ROOT),
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "window": WINDOW,
        "window_dates": [start, end],
        "mode": args.mode,
        "pilot_size": args.pilot_size if args.mode == "pilot" else None,
        "pilot_seed": PILOT_SEED if args.mode == "pilot" else None,
        "command": command,
        "code_sha256": _code_sha256(),
        "labels_csv": str(csv_path),
        "labels_csv_sha256": _sha256(csv_path),
        "data_release": data_release,
        "bars_sha256": bars_sha256,
        "spy_bars_sha256": (spy_prov or {}).get("bars_sha256"),
        "data_provenance": data_prov,
    }

    payload = {
        "study": STUDY_ID,
        "mode": args.mode,
        "window": WINDOW,
        "window_dates": [start, end],
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "git_head": git_sha(ROOT),
        "command": command,
        "provenance": provenance_block,
        "data_release": data_release,
        "bars_sha256": bars_sha256,
        "data_provenance": data_prov,
        "roster_counts": roster_counts,
        "summary": summary,
        "sec_requests": sec_requests,
        "wall_seconds": time.time() - started,
    }
    write_json(out_path, payload)
    print(f"wrote {out_path} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
