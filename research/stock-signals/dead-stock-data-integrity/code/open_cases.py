"""Phase 3 open-case matching, evidence classification, merging, and coverage under rule v2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import payout_table
import sec_filings
from sec_name_index import IndexedFiling

DEAL_FORMS: frozenset[str] = frozenset(
    {
        "DEFM14A",
        "SC TO-T",
        "SC TO-T/A",
        "SC 14D9",
        "SC 14D9/A",
        "SC 13E3",
        "SC 13E3/A",
    }
)
DEAL_DAYS_BEFORE = 365
DEAL_DAYS_AFTER = 30
CANDIDATE_DAYS_BEFORE = 365
CANDIDATE_DAYS_AFTER = 180
STILL_FILING_FIRST_DAY = 181
INDEX_LAST_DAY = date(2025, 3, 31)
PERIODIC_FORMS: frozenset[str] = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})

CLASS_DEAL_FORMS_ONLY = "deal_forms_only"
PHASE3_CLASSES: tuple[str, ...] = payout_table.EVIDENCE_CLASSES + (CLASS_DEAL_FORMS_ONLY,)
RULE_VERSION_V2 = "v2"
ADDED_COLUMNS: tuple[str, ...] = (
    "phase",
    "name_rule",
    "reason",
    "still_filing_check",
    "deal_form_accessions",
)
OUTPUT_COLUMNS: tuple[str, ...] = payout_table.TABLE_COLUMNS + ADDED_COLUMNS

STATUS_MATCHED = sec_filings.STATUS_MATCHED
STATUS_AMBIGUOUS = payout_table.CLASS_AMBIGUOUS
STATUS_NO_FILING = payout_table.CLASS_NO_FILING_MATCH
VALID_MATCH_STATUSES: frozenset[str] = frozenset(
    {STATUS_MATCHED, STATUS_AMBIGUOUS, STATUS_NO_FILING}
)

STILL_FILING_DONE = "done"
STILL_FILING_INCOMPLETE = "incomplete"
REASON_EVENTS_INCOMPLETE = "events_incomplete"
REASON_NO_DEAL_FORMS = "no_deal_forms"
REASON_STILL_FILING_PREFIX = "company_still_filing:"

PHASE_1 = "1"
PHASE_3 = "3"

LITERATURE_LOSS_VALUES: frozenset[float] = frozenset(
    float(entry["loss"]) for entry in payout_table.LITERATURE_FILLS.values()
)


@dataclass(frozen=True)
class OpenMatch:
    """Result of matching one open roster stock against pass-2 quarterly index filings."""

    status: str
    cik: int | None
    name_rule: str
    filer_name: str
    reason: str
    still_filing_check: str


def _as_date(value: Any) -> date:
    """Convert an ISO date string or ``date``/``datetime`` into a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _as_text(value: Any) -> str:
    """Format a row value as CSV-compatible text (empty string for missing/NaN values)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _stock_name(row: Mapping[str, Any]) -> str | None:
    """Return the raw non-blank stock name of a roster row, or ``None`` if unnamed."""
    if row.get("name_source") == "none":
        return None
    name = row.get("stock_name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name


def _exchcd_for_loss(value: Any) -> Any:
    """Parse ASCII integer CSV strings for ``payout_table.loss_decision`` while staying strict."""
    # payout_table.clean_exchange rejects strings, while CSV rows store exchcd as text like "1".
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isascii() and stripped.isdigit():
            return int(stripped)
        return None
    return value


def _format_loss(loss: float | None) -> str:
    """Format a suggested loss float as CSV text, matching ``_payouts.csv``."""
    if loss is None or (isinstance(loss, float) and math.isnan(loss)):
        return ""
    return str(float(loss))


def _parse_loss(value: Any) -> tuple[bool, float | None]:
    """Parse a text or numeric loss field into ``(present, finite_float_or_none)``."""
    text = _as_text(value).strip()
    if not text:
        return False, None
    try:
        parsed = float(text)
    except ValueError:
        return True, None
    if not math.isfinite(parsed):
        return True, None
    return True, parsed


def _parse_optional_cik(value: Any) -> int | None:
    """Parse an optional CIK value into an integer, or ``None`` when absent."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isascii() and text.isdigit():
        return int(text)
    return None


def name_rules(row: Mapping[str, Any], raw_name: str) -> frozenset[str]:
    """Return the subset of ``{"v1", "v1.1"}`` under which ``raw_name`` matches ``row``.

    Uses ``sec_filings._name_matches`` for v1 and ``sec_filings._name_matches_v11`` for v1.1 with
    ``raw_length = len(row["stock_name"])``. Unnamed rows match nothing.

    Args:
        row: Roster or payout row mapping containing ``stock_name``.
        raw_name: Raw company name from an SEC quarterly index line.

    Returns:
        Frozen set containing any of ``"v1"`` and ``"v1.1"`` that match.
    """
    stock_name = _stock_name(row)
    if stock_name is None or not isinstance(raw_name, str) or not raw_name.strip():
        return frozenset()
    raw_length = len(stock_name)
    matched: set[str] = set()
    # Guard against punctuation-only stock names whose v1 cleaned string is empty.
    if sec_filings.normalize_name(stock_name) and sec_filings._name_matches(
        stock_name, raw_length, raw_name
    ):
        matched.add(sec_filings.RULE_VERSION_V1)
    if sec_filings._name_matches_v11(stock_name, raw_length, raw_name):
        matched.add(sec_filings.RULE_VERSION_V11)
    return frozenset(matched)


def make_matcher(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[str], Mapping[str, frozenset[str]]]:
    """Build the pass-1 name matcher callable for ``sec_name_index.find_filers``.

    Precomputes normalized forms per roster row so streaming across 80 quarterly archives
    normalizes each distinct raw SEC company name once rather than once per roster row, while
    producing the exact same matches as ``name_rules``.

    Args:
        rows: Sequence of roster/payout row mappings, each keyed by ``row["symbol"]``.

    Returns:
        Callable mapping a raw SEC filer name to ``{symbol: frozenset(matched_rules)}`` with
        non-empty rule sets only.
    """
    v1_exact: dict[str, list[str]] = {}
    v1_prefix: list[tuple[str, str]] = []
    v11_exact: dict[str, list[str]] = {}
    v11_prefix: list[tuple[str, str]] = []

    for row in rows:
        stock_name = _stock_name(row)
        if stock_name is None:
            continue
        symbol = str(row["symbol"])
        is_truncated = len(stock_name) == sec_filings.TRUNCATED_NAME_LENGTH

        wanted_v1 = sec_filings.normalize_name(stock_name)
        if wanted_v1:
            if is_truncated:
                v1_prefix.append((symbol, wanted_v1))
            else:
                v1_exact.setdefault(wanted_v1, []).append(symbol)

        wanted_v11 = sec_filings.normalize_name_v11(stock_name)
        if wanted_v11:
            v11_exact.setdefault(wanted_v11, []).append(symbol)
            if is_truncated:
                wanted_untrimmed = sec_filings.normalize_name_v11_untrimmed(stock_name)
                if wanted_untrimmed:
                    v11_prefix.append((symbol, wanted_untrimmed))

    has_v11_prefix = bool(v11_prefix)

    def _match(raw_name: str) -> Mapping[str, frozenset[str]]:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return {}
        found: dict[str, set[str]] = {}

        cleaned_v1 = sec_filings.normalize_name(raw_name)
        if cleaned_v1:
            for sym in v1_exact.get(cleaned_v1, ()):
                found.setdefault(sym, set()).add(sec_filings.RULE_VERSION_V1)
            for sym, prefix in v1_prefix:
                if cleaned_v1.startswith(prefix):
                    found.setdefault(sym, set()).add(sec_filings.RULE_VERSION_V1)

        cleaned_v11 = sec_filings.normalize_name_v11(raw_name)
        if cleaned_v11:
            for sym in v11_exact.get(cleaned_v11, ()):
                found.setdefault(sym, set()).add(sec_filings.RULE_VERSION_V11)
        if has_v11_prefix:
            cleaned_untrimmed = sec_filings.normalize_name_v11_untrimmed(raw_name)
            if cleaned_untrimmed:
                for sym, prefix in v11_prefix:
                    if cleaned_untrimmed.startswith(prefix):
                        found.setdefault(sym, set()).add(sec_filings.RULE_VERSION_V11)

        return {sym: frozenset(rules) for sym, rules in found.items()}

    return _match


def match_open_case(
    row: Mapping[str, Any],
    filings: Mapping[int, Sequence[IndexedFiling]],
) -> OpenMatch:
    """Match one open roster row to a single SEC filer under frozen rules 1 and 3.

    Rule 1 requires name and in-window date (``exit_date - 365`` through ``exit_date + 180``,
    inclusive) on the same index line under v1 first; v1.1 runs only when v1 finds zero
    candidates. Rule 3 rejects a matched CIK that still files periodic reports (10-K/10-Q and
    amendments) from ``exit_date + 181`` through ``INDEX_LAST_DAY`` inclusive.

    Args:
        row: Roster or payout row mapping with ``stock_name`` and ``exit_date``.
        filings: Mapping from candidate integer CIK to its pass-2 ``IndexedFiling`` records.

    Returns:
        ``OpenMatch`` with ``status`` in ``{"matched", "ambiguous_match", "no_filing_match"}``.
    """
    if _stock_name(row) is None:
        return OpenMatch(
            status=STATUS_NO_FILING,
            cik=None,
            name_rule="",
            filer_name="",
            reason="",
            still_filing_check="",
        )

    exit_date = _as_date(row["exit_date"])
    window_start = exit_date - timedelta(days=CANDIDATE_DAYS_BEFORE)
    window_end = exit_date + timedelta(days=CANDIDATE_DAYS_AFTER)

    def _collect_candidates(rule: str) -> dict[int, list[IndexedFiling]]:
        hits: dict[int, list[IndexedFiling]] = {}
        for cik, cik_filings in filings.items():
            qualifying = [
                f
                for f in cik_filings
                if window_start <= f.filed <= window_end and rule in name_rules(row, f.name)
            ]
            if qualifying:
                hits[cik] = qualifying
        return hits

    # Rule 1 fallback order: v1.1 never runs if v1 found at least one candidate CIK.
    candidates = _collect_candidates(sec_filings.RULE_VERSION_V1)
    active_rule = sec_filings.RULE_VERSION_V1
    if not candidates:
        candidates = _collect_candidates(sec_filings.RULE_VERSION_V11)
        active_rule = sec_filings.RULE_VERSION_V11

    if not candidates:
        return OpenMatch(
            status=STATUS_NO_FILING,
            cik=None,
            name_rule="",
            filer_name="",
            reason="",
            still_filing_check="",
        )

    if len(candidates) > 1:
        sorted_ciks = sorted(candidates)
        return OpenMatch(
            status=STATUS_AMBIGUOUS,
            cik=None,
            name_rule="",
            filer_name="",
            reason=f"multiple_ciks: {', '.join(str(c) for c in sorted_ciks)}",
            still_filing_check="",
        )

    matched_cik, qualifying_lines = next(iter(candidates.items()))
    earliest_line = min(
        qualifying_lines,
        key=lambda f: (f.filed, f.form, f.name, f.accession),
    )

    # Rule 3: periodic reports on or after exit+181 up to the cached index end reject the match.
    still_start = exit_date + timedelta(days=STILL_FILING_FIRST_DAY)
    if still_start > INDEX_LAST_DAY:
        return OpenMatch(
            status=STATUS_MATCHED,
            cik=matched_cik,
            name_rule=active_rule,
            filer_name=earliest_line.name,
            reason="",
            still_filing_check=STILL_FILING_INCOMPLETE,
        )

    still_filing = any(
        f.form.strip() in PERIODIC_FORMS and still_start <= f.filed <= INDEX_LAST_DAY
        for f in filings[matched_cik]
    )
    if still_filing:
        return OpenMatch(
            status=STATUS_NO_FILING,
            cik=None,
            name_rule="",
            filer_name="",
            reason=f"{REASON_STILL_FILING_PREFIX} {matched_cik}",
            still_filing_check="",
        )

    return OpenMatch(
        status=STATUS_MATCHED,
        cik=matched_cik,
        name_rule=active_rule,
        filer_name=earliest_line.name,
        reason="",
        still_filing_check=STILL_FILING_DONE,
    )


def deal_form_filings(
    filings: Sequence[IndexedFiling],
    exit_date: date,
) -> list[IndexedFiling]:
    """Return deal-form filings dated from ``exit_date - 365`` to ``exit_date + 30`` inclusive.

    Args:
        filings: Sequence of ``IndexedFiling`` records for one CIK.
        exit_date: Stock's last trading day from ``_payouts.csv``.

    Returns:
        Matching deal-form filings sorted in ascending date order.
    """
    exit_dt = _as_date(exit_date)
    start = exit_dt - timedelta(days=DEAL_DAYS_BEFORE)
    end = exit_dt + timedelta(days=DEAL_DAYS_AFTER)
    deals = [f for f in filings if f.form.strip() in DEAL_FORMS and start <= f.filed <= end]
    deals.sort(key=lambda f: (f.filed, f.form, f.accession, f.name))
    return deals


def classify_open_case(
    row: Mapping[str, Any],
    match: OpenMatch,
    events: Sequence[Mapping[str, Any]],
    events_status: str,
    filings: Sequence[IndexedFiling],
) -> dict[str, str]:
    """Build one complete Phase 3 output row under frozen rules 4, 5, and 6.

    Args:
        row: Original 21-column row from ``_payouts.csv``.
        match: Outcome of ``match_open_case`` for this row.
        events: Window 8-K events fetched for the matched CIK (empty when unmatched).
        events_status: Retrieval status from ``fetch_events_with_status``.
        filings: Pass-2 quarterly index filings for the matched CIK (empty when unmatched).

    Returns:
        Dictionary with all 26 ``OUTPUT_COLUMNS`` as CSV-ready text strings.

    Raises:
        ValueError: If ``match.status`` is not a recognized ``OpenMatch`` status.
    """
    if match.status not in VALID_MATCH_STATUSES:
        raise ValueError(f"Unexpected OpenMatch status: {match.status!r}")

    out: dict[str, str] = {col: _as_text(row.get(col)) for col in payout_table.TABLE_COLUMNS}

    if match.status != STATUS_MATCHED:
        if _stock_name(row) is None and out["evidence_class"] == payout_table.CLASS_NAME_UNKNOWN:
            evidence = payout_table.CLASS_NAME_UNKNOWN
        else:
            evidence = match.status
        loss, basis, note = payout_table.loss_decision(
            evidence, _exchcd_for_loss(row.get("exchcd"))
        )
        out["evidence_class"] = evidence
        out["match_confidence"] = payout_table.CONFIDENCE_NONE
        out["match_rule_version"] = payout_table.RULE_VERSION_NONE
        out["suggested_loss"] = _format_loss(loss)
        out["basis"] = basis
        out["loss_note"] = note
        out["cross_check"] = payout_table.cross_check(evidence, out["price_label"])
        out["phase"] = PHASE_3
        out["name_rule"] = ""
        out["reason"] = match.reason
        out["still_filing_check"] = ""
        out["deal_form_accessions"] = ""
        return {col: out[col] for col in OUTPUT_COLUMNS}

    exit_dt = _as_date(row["exit_date"])
    deals = deal_form_filings(filings, exit_dt)
    deal_accessions = ";".join(dict.fromkeys(f.accession for f in deals))
    status_text = _as_text(events_status)

    if status_text not in payout_table.EVENTS_STATUS_COMPLETE:
        # Incomplete 8-K retrieval blocks both 8-K classification and rule 5 deal-form promotion.
        evidence = sec_filings.EVIDENCE_NONE
        reason = REASON_EVENTS_INCOMPLETE
    else:
        eight_k_class = sec_filings.evidence_class([dict(e) for e in events])
        if eight_k_class == sec_filings.EVIDENCE_NONE and deals:
            evidence = CLASS_DEAL_FORMS_ONLY
        else:
            evidence = eight_k_class
        reason = ""

    loss, basis, note = payout_table.loss_decision(evidence, _exchcd_for_loss(row.get("exchcd")))
    out["evidence_class"] = evidence
    # Rule 6: every rule 1 match stays name_only under rule v2, never confirmed.
    out["match_confidence"] = payout_table.CONFIDENCE_NAME_ONLY
    out["match_rule_version"] = RULE_VERSION_V2
    out["event_accessions"] = ";".join(str(e["accession"]) for e in events if e.get("accession"))
    out["events_status"] = status_text
    out["suggested_loss"] = _format_loss(loss)
    out["basis"] = basis
    out["loss_note"] = note
    out["cross_check"] = payout_table.cross_check(evidence, out["price_label"])
    out["phase"] = PHASE_3
    out["name_rule"] = match.name_rule
    out["reason"] = reason
    out["still_filing_check"] = match.still_filing_check
    out["deal_form_accessions"] = deal_accessions
    return {col: out[col] for col in OUTPUT_COLUMNS}


def classify_phase2_no_8k(
    row: Mapping[str, Any],
    filings: Sequence[IndexedFiling],
) -> dict[str, str]:
    """Apply frozen rules 2 and 5 to one of the five Phase 2 ``no_8k_found`` rows.

    Preserves ``match_confidence``, ``match_rule_version``, and all Form 25 and 8-K fields from
    Phase 2. Only promotes the class to ``deal_forms_only`` when the row's ``events_status`` is
    complete and at least one deal form appears in the exit window.

    Args:
        row: Original Phase 2 ``no_8k_found`` row from ``_payouts.csv``.
        filings: Pass-2 quarterly index filings for that row's Phase 2 CIK.

    Returns:
        Dictionary with all 26 ``OUTPUT_COLUMNS`` as CSV-ready text strings.

    Raises:
        ValueError: If ``row["evidence_class"]`` is not ``"no_8k_found"``.
    """
    if _as_text(row.get("evidence_class")) != sec_filings.EVIDENCE_NONE:
        raise ValueError(
            f"classify_phase2_no_8k requires a no_8k_found row, got {row.get('evidence_class')!r}"
        )

    out: dict[str, str] = {col: _as_text(row.get(col)) for col in payout_table.TABLE_COLUMNS}
    deals = deal_form_filings(filings, _as_date(row["exit_date"]))
    deal_accessions = ";".join(dict.fromkeys(f.accession for f in deals))

    if out["events_status"] not in payout_table.EVENTS_STATUS_COMPLETE:
        reason = REASON_EVENTS_INCOMPLETE
    elif deals:
        loss, basis, note = payout_table.loss_decision(
            CLASS_DEAL_FORMS_ONLY, _exchcd_for_loss(row.get("exchcd"))
        )
        out["evidence_class"] = CLASS_DEAL_FORMS_ONLY
        out["suggested_loss"] = _format_loss(loss)
        out["basis"] = basis
        out["loss_note"] = note
        out["cross_check"] = payout_table.cross_check(CLASS_DEAL_FORMS_ONLY, out["price_label"])
        reason = ""
    else:
        reason = REASON_NO_DEAL_FORMS

    out["phase"] = PHASE_3
    out["name_rule"] = ""
    out["reason"] = reason
    out["still_filing_check"] = ""
    out["deal_form_accessions"] = deal_accessions
    return {col: out[col] for col in OUTPUT_COLUMNS}


def merge_open_cases(
    phase1_rows: Sequence[Mapping[str, str]],
    phase3_rows: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Merge unchanged Phase 1 evidence rows with Phase 3 outcomes in original roster order.

    Every row whose original ``evidence_class`` is in ``payout_table.DOCUMENTED_CLASSES`` passes
    through with all 21 original field values untouched and ``phase = "1"``. Every other row must
    appear in ``phase3_rows`` keyed by ``symbol``.

    Args:
        phase1_rows: All rows of ``_payouts.csv`` in original order.
        phase3_rows: Mapping from ``symbol`` to Phase 3 output row for open cases only.

    Returns:
        List of 26-column text row dicts in the exact order of ``phase1_rows``.

    Raises:
        ValueError: If duplicate symbols appear, a Phase 3 row is supplied for an evidenced
            symbol, an open symbol is missing from ``phase3_rows``, or ``phase3_rows`` contains
            an unknown symbol.
    """
    merged: list[dict[str, str]] = []
    seen_symbols: set[str] = set()

    for row in phase1_rows:
        symbol = _as_text(row.get("symbol"))
        if not symbol:
            raise ValueError("Phase 1 row is missing a symbol")
        if symbol in seen_symbols:
            raise ValueError(f"Duplicate symbol in phase1_rows: {symbol!r}")
        seen_symbols.add(symbol)

        if _as_text(row.get("evidence_class")) in payout_table.DOCUMENTED_CLASSES:
            if symbol in phase3_rows:
                raise ValueError(f"Phase 3 row supplied for evidenced Phase 1 symbol {symbol!r}")
            preserved = {col: _as_text(row.get(col)) for col in payout_table.TABLE_COLUMNS}
            preserved["phase"] = PHASE_1
            preserved["name_rule"] = ""
            preserved["reason"] = ""
            preserved["still_filing_check"] = ""
            preserved["deal_form_accessions"] = ""
            merged.append({col: preserved[col] for col in OUTPUT_COLUMNS})
        else:
            if symbol not in phase3_rows:
                raise ValueError(f"Missing Phase 3 row for open symbol {symbol!r}")
            p3 = phase3_rows[symbol]
            p3_symbol = _as_text(p3.get("symbol"))
            if p3_symbol and p3_symbol != symbol:
                raise ValueError(f"Phase 3 row symbol {p3_symbol!r} does not match key {symbol!r}")
            combined = {col: _as_text(p3.get(col)) for col in OUTPUT_COLUMNS}
            combined["symbol"] = symbol
            if not combined["phase"]:
                combined["phase"] = PHASE_3
            merged.append(combined)

    unknown = sorted(set(phase3_rows) - seen_symbols)
    if unknown:
        raise ValueError(f"Phase 3 rows supplied for unknown symbols: {unknown}")

    return merged


def phase3_coverage(
    before: Sequence[Mapping[str, str]],
    after: Sequence[Mapping[str, str]],
    *,
    open_matches: Mapping[str, OpenMatch] | None = None,
) -> dict[str, Any]:
    """Summarize class counts, moves, Phase 3 matches, rejections, and best-case invariance.

    Because ``OUTPUT_COLUMNS`` (the CSV schema) omits ``filer_name`` and ``cik``, callers that
    want those populated on ``matches`` entries may pass ``open_matches`` (keyed by ``symbol``)
    or include ``filer_name``/``cik`` keys on the ``after`` row mappings.

    Args:
        before: Original rows from ``_payouts.csv``.
        after: Merged Phase 3 rows (same symbol set as ``before``).
        open_matches: Optional mapping from ``symbol`` to ``OpenMatch`` for Phase 3 rows.

    Returns:
        Summary dictionary with before/after class counts, ``moves``, ``deal_forms_only`` count
        apart, ``v1_1_only_matches``, ``matches``, ``rejected_still_filing``,
        ``still_filing_incomplete``, and ``best_case_unchanged``.

    Raises:
        ValueError: If ``before`` and ``after`` do not contain the exact same unique symbols.
    """
    before_by_symbol: dict[str, Mapping[str, str]] = {}
    for row in before:
        symbol = _as_text(row.get("symbol"))
        if not symbol or symbol in before_by_symbol:
            raise ValueError(f"Invalid or duplicate symbol in before rows: {symbol!r}")
        before_by_symbol[symbol] = row

    after_by_symbol: dict[str, Mapping[str, str]] = {}
    for row in after:
        symbol = _as_text(row.get("symbol"))
        if not symbol or symbol in after_by_symbol:
            raise ValueError(f"Invalid or duplicate symbol in after rows: {symbol!r}")
        after_by_symbol[symbol] = row

    if set(before_by_symbol) != set(after_by_symbol):
        raise ValueError("before and after must cover the exact same symbol set")

    before_by_class: dict[str, int] = {cls: 0 for cls in PHASE3_CLASSES}
    after_by_class: dict[str, int] = {cls: 0 for cls in PHASE3_CLASSES}
    moves: dict[str, int] = {}

    matches: list[dict[str, Any]] = []
    v1_1_only_matches: list[dict[str, Any]] = []
    rejected_still_filing: list[dict[str, str]] = []
    still_filing_incomplete = 0
    best_case_unchanged = True

    for row in before:
        symbol = _as_text(row.get("symbol"))
        after_row = after_by_symbol[symbol]

        src_class = _as_text(row.get("evidence_class"))
        dst_class = _as_text(after_row.get("evidence_class"))
        before_by_class[src_class] = before_by_class.get(src_class, 0) + 1
        after_by_class[dst_class] = after_by_class.get(dst_class, 0) + 1
        if src_class != dst_class:
            move_key = f"{src_class} -> {dst_class}"
            moves[move_key] = moves.get(move_key, 0) + 1

        phase = _as_text(after_row.get("phase"))
        rule_version = _as_text(after_row.get("match_rule_version"))
        reason = _as_text(after_row.get("reason"))
        check_status = _as_text(after_row.get("still_filing_check"))

        if check_status == STILL_FILING_INCOMPLETE:
            still_filing_incomplete += 1

        if phase == PHASE_3 and rule_version == RULE_VERSION_V2:
            match_obj = open_matches.get(symbol) if open_matches is not None else None
            if match_obj is not None and match_obj.status == STATUS_MATCHED:
                filer_name = match_obj.filer_name
                cik: int | None = match_obj.cik
                rule_label = _as_text(after_row.get("name_rule")) or match_obj.name_rule
            else:
                filer_name = _as_text(after_row.get("filer_name"))
                cik = _parse_optional_cik(after_row.get("cik"))
                rule_label = _as_text(after_row.get("name_rule"))
            entry = {
                "roster_name": _as_text(after_row.get("stock_name")),
                "filer_name": filer_name,
                "cik": cik,
                "name_rule": rule_label,
                "exit_date": _as_text(after_row.get("exit_date")),
            }
            matches.append(entry)
            if rule_label == sec_filings.RULE_VERSION_V11:
                v1_1_only_matches.append(entry)

        if phase == PHASE_3 and reason.startswith(REASON_STILL_FILING_PREFIX):
            rejected_still_filing.append(
                {
                    "roster_name": _as_text(after_row.get("stock_name")),
                    "reason": reason,
                    "exit_date": _as_text(after_row.get("exit_date")),
                }
            )

        # Check best-case invariance: every non-empty loss after must be 0.0 or a bankruptcy
        # literature fill, and no Phase 1 loss may change.
        after_has_loss, after_loss = _parse_loss(after_row.get("suggested_loss"))
        if after_has_loss:
            is_zero = after_loss == 0.0
            is_bankruptcy_lit = (
                dst_class == sec_filings.EVIDENCE_BANKRUPTCY
                and _as_text(after_row.get("basis")) == payout_table.BASIS_LITERATURE
                and after_loss in LITERATURE_LOSS_VALUES
            )
            if not (is_zero or is_bankruptcy_lit):
                best_case_unchanged = False

        before_has_loss, before_loss = _parse_loss(row.get("suggested_loss"))
        if src_class in payout_table.DOCUMENTED_CLASSES or before_has_loss:
            if before_has_loss != after_has_loss or before_loss != after_loss:
                best_case_unchanged = False

    return {
        "roster_rows": len(after),
        "before_by_class": before_by_class,
        "after_by_class": after_by_class,
        "moves": dict(sorted(moves.items())),
        "deal_forms_only": after_by_class[CLASS_DEAL_FORMS_ONLY],
        "v1_1_only_matches": v1_1_only_matches,
        "matches": matches,
        "rejected_still_filing": rejected_still_filing,
        "still_filing_incomplete": still_filing_incomplete,
        "best_case_unchanged": best_case_unchanged,
    }
