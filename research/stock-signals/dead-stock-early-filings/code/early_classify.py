"""Pure classification layer for the dead-stock-early-filings audit (no I/O, no network).

Combines the death roster (``early_roster``), McGill company names (``early_names``), and an
injected SEC lookup into the registered per-stock labels, confidence tiers, evidence strings,
suggested loss values, pilot sampling, and summary counts. All label precedence, era cutoffs,
180-day kept-filing logic, and confidence promotion rules are delegated to ``early_sec``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import early_roster
import early_sec
import numpy as np
import pandas as pd

PILOT_SEED = 20261125

REGISTERED_LABELS: tuple[str, ...] = (
    early_sec.LABEL_BANKRUPTCY,
    early_sec.LABEL_ACQUISITION_TYPE,
    early_sec.LABEL_DELISTING_NOTICE_ONLY,
    early_sec.LABEL_KEPT_FILING,
    early_sec.LABEL_NO_EVENT_FOUND,
    early_sec.LABEL_NO_COMPANY_MATCH,
    early_sec.LABEL_NAME_UNKNOWN,
)

CONFIDENCE_TIERS: tuple[str, ...] = (
    early_sec.CONFIDENCE_CHECKED,
    early_sec.CONFIDENCE_NAME_ONLY,
    early_sec.CONFIDENCE_FORMER_NAME,
)

OUTPUT_COLUMNS: list[str] = [
    "column",
    "permno",
    "symbol",
    "last_trading_day",
    "company_name",
    "name_source",
    "exchange",
    "rank_band",
    "held_by_rule",
    "label",
    "confidence",
    "evidence",
    "suggested_loss",
]

REQUIRED_ROSTER_COLUMNS: tuple[str, ...] = (
    "column",
    "permno",
    "symbol",
    "last_bar",
    "rank_band",
    "held_by_rule",
)

EXCHANGE_NAMES: dict[int, str] = {
    1: "NYSE",
    2: "AMEX",
    3: "Nasdaq",
}

_CANONICAL_RANK_BANDS: tuple[str, ...] = tuple(band for band, _, _ in early_roster.RANK_BANDS)


@dataclass(frozen=True, slots=True)
class FilingEvidence:
    """One in-window event filing (8-K / 8-K/A / Form 25) and its optional parsed header."""

    filing: early_sec.Filing
    header: early_sec.FilingHeader | None = None


@dataclass(frozen=True, slots=True)
class CompanyEvidence:
    """SEC lookup evidence returned for one named roster row."""

    cik: int
    company_name: str
    match_confidence: str
    window_filings: tuple[FilingEvidence, ...] = ()
    all_filing_dates: tuple[date, ...] = ()


SecLookupFn = Callable[[str, date], CompanyEvidence | None]


def _as_date(value: date | datetime | pd.Timestamp | str) -> date:
    """Normalise a Timestamp, datetime, date, or ISO string to ``datetime.date``."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _clean_text(value: Any) -> str | None:
    """Return a stripped non-empty string, or ``None`` for null/blank values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def exchange_label(exchcd: Any, exchange_text: Any = None) -> str:
    """Map a CRSP/McGill exchange code to a readable label (NYSE, AMEX, Nasdaq, unknown)."""
    explicit = _clean_text(exchange_text)
    if explicit is not None:
        return explicit
    if exchcd is None or pd.isna(exchcd):
        return "unknown"
    code = int(exchcd)
    return EXCHANGE_NAMES.get(code, f"exchcd_{code}")


def merge_names(roster: pd.DataFrame, names: pd.DataFrame | None) -> pd.DataFrame:
    """Attach ``company_name``, ``name_source``, ``exchcd`` (and optional ``exchange``) to roster.

    Accepts ``names=None`` (uses columns already on ``roster``), a full ``attach_names`` copy
    or minimal table keyed by unique ``column`` (preferred) or ``permno``, or a same-length
    row-aligned frame. Never mutates ``roster`` or ``names``.
    """
    out = roster.copy().reset_index(drop=True)
    if names is None:
        if "company_name" not in out.columns:
            out["company_name"] = pd.Series([None] * len(out), dtype=object)
        if "name_source" not in out.columns:
            out["name_source"] = pd.Series([""] * len(out), dtype=object)
        if "exchcd" not in out.columns:
            out["exchcd"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        return out

    extra_cols = [
        c for c in ("company_name", "name_source", "exchcd", "exchange") if c in names.columns
    ]
    if not extra_cols:
        return merge_names(out, None)

    if "column" in names.columns and names["column"].is_unique:
        by_col = names.set_index("column")
        for col in extra_cols:
            out[col] = out["column"].map(by_col[col])
    elif "permno" in names.columns and names["permno"].dropna().is_unique:
        by_permno = names.dropna(subset=["permno"]).set_index("permno")
        for col in extra_cols:
            out[col] = out["permno"].map(by_permno[col])
    elif len(names) == len(out):
        aligned = names.reset_index(drop=True)
        for col in extra_cols:
            out[col] = aligned[col]
    else:
        raise ValueError(
            "names frame must have unique 'column' or 'permno' keys or match roster length"
        )

    if "company_name" not in out.columns:
        out["company_name"] = pd.Series([None] * len(out), dtype=object)
    if "name_source" not in out.columns:
        out["name_source"] = pd.Series([""] * len(out), dtype=object)
    if "exchcd" not in out.columns:
        out["exchcd"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    else:
        out["exchcd"] = pd.Series(out["exchcd"], index=out.index, dtype="Int64")
    return out


def select_pilot_rows(
    roster: pd.DataFrame,
    *,
    pilot_size: int = 20,
    seed: int = PILOT_SEED,
) -> pd.DataFrame:
    """Pick the first ``pilot_size`` rows of a seeded shuffle of ``roster`` ordered by permno."""
    if pilot_size <= 0:
        raise ValueError(f"pilot_size must be positive, got {pilot_size}")
    if roster.empty:
        return roster.copy().reset_index(drop=True)
    sort_cols = [c for c in ("permno", "column") if c in roster.columns]
    ordered = roster.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(
        drop=True
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    take = min(pilot_size, len(ordered))
    return ordered.iloc[order[:take]].reset_index(drop=True)


def _in_window(filed: date, last_day: date) -> bool:
    """Return True if ``filed`` is within 30 days before to 180 days after ``last_day``."""
    return (
        last_day - timedelta(days=early_sec.BEFORE_DAYS)
        <= filed
        <= last_day + timedelta(days=early_sec.AFTER_DAYS)
    )


def _effective_base_confidence(company_conf: str, filing_conf: str) -> str:
    """Ensure a ``former_name`` match can never be promoted to ``name_only`` or ``checked``."""
    if early_sec.CONFIDENCE_FORMER_NAME in (company_conf, filing_conf):
        return early_sec.CONFIDENCE_FORMER_NAME
    return filing_conf or company_conf or early_sec.CONFIDENCE_NAME_ONLY


def _format_filing_evidence(ev: FilingEvidence) -> str:
    """Format one filing as a short deterministic evidence snippet (form, date, items)."""
    form = ev.filing.form
    filed_str = _as_date(ev.filing.filed).isoformat()
    if ev.header is not None and ev.header.items:
        items_str = ", ".join(sorted(ev.header.items))
        return f"{form} {filed_str} [{items_str}]"
    return f"{form} {filed_str}"


def _best_confidence(
    company_conf: str,
    qualifying: list[FilingEvidence],
) -> str:
    """Resolve confidence across qualifying filings using ``early_sec.resolve_confidence``."""
    if not qualifying:
        return company_conf
    resolved = [
        early_sec.resolve_confidence(
            _effective_base_confidence(company_conf, ev.filing.confidence),
            ev.filing,
            ev.header,
        )
        for ev in qualifying
    ]
    return max(resolved, key=lambda tier: early_sec.CONFIDENCE_RANK.get(tier, 0))


def _invoke_lookup(sec_lookup: Any, company_name: str, last_day: date) -> CompanyEvidence | None:
    """Call an injected ``sec_lookup`` callable or object with a ``.lookup`` method."""
    if callable(sec_lookup):
        return sec_lookup(company_name, last_day)
    if hasattr(sec_lookup, "lookup") and callable(sec_lookup.lookup):
        return sec_lookup.lookup(company_name, last_day)
    raise TypeError(
        "sec_lookup must be callable or define a .lookup(company_name, last_day) method"
    )


def _classify_one(
    company_name: str | None,
    last_day: date,
    sec_lookup: Any,
) -> tuple[str, str, str, float]:
    """Classify a single roster row into ``(label, confidence, evidence, suggested_loss)``."""
    if company_name is None:
        return early_sec.LABEL_NAME_UNKNOWN, "", "", float("nan")

    evidence = _invoke_lookup(sec_lookup, company_name, last_day)
    if evidence is None:
        return early_sec.LABEL_NO_COMPANY_MATCH, "", "", float("nan")

    in_window = [
        ev for ev in evidence.window_filings if _in_window(_as_date(ev.filing.filed), last_day)
    ]
    eight_k_evs = [ev for ev in in_window if ev.filing.form.upper() in early_sec.FORM8K_FORMS]
    form25_evs = [ev for ev in in_window if ev.filing.form.upper() in early_sec.FORM25_FORMS]

    header_entries = [ev.header for ev in eight_k_evs if ev.header is not None]
    label = early_sec.classify_company_events(
        header_entries,
        has_form25=bool(form25_evs),
        filing_dates=evidence.all_filing_dates,
        last_bar=last_day,
    )

    if label in (early_sec.LABEL_BANKRUPTCY, early_sec.LABEL_ACQUISITION_TYPE):
        qualifying = sorted(
            [
                ev
                for ev in eight_k_evs
                if ev.header is not None and early_sec.label_filing_header(ev.header) == label
            ],
            key=lambda ev: (_as_date(ev.filing.filed), ev.filing.form, ev.filing.file_path),
        )
        confidence = _best_confidence(evidence.match_confidence, qualifying)
        evidence_str = "; ".join(_format_filing_evidence(ev) for ev in qualifying)
    elif label == early_sec.LABEL_DELISTING_NOTICE_ONLY:
        qualifying_8k = [
            ev
            for ev in eight_k_evs
            if ev.header is not None
            and early_sec.label_filing_header(ev.header) == early_sec.LABEL_DELISTING_NOTICE_ONLY
        ]
        qualifying = sorted(
            [*form25_evs, *qualifying_8k],
            key=lambda ev: (_as_date(ev.filing.filed), ev.filing.form, ev.filing.file_path),
        )
        confidence = _best_confidence(evidence.match_confidence, qualifying)
        evidence_str = "; ".join(_format_filing_evidence(ev) for ev in qualifying)
    elif label == early_sec.LABEL_KEPT_FILING:
        confidence = evidence.match_confidence
        cutoff = last_day + timedelta(days=early_sec.AFTER_DAYS)
        later = sorted(d for d in (_as_date(x) for x in evidence.all_filing_dates) if d > cutoff)
        if not later:
            evidence_str = f"CIK {evidence.cik}: filing after >180d"
        elif later[0] == later[-1]:
            evidence_str = f"CIK {evidence.cik}: filed {later[0].isoformat()} (>180d)"
        else:
            evidence_str = (
                f"CIK {evidence.cik}: {len(later)} filings >180d "
                f"({later[0].isoformat()}..{later[-1].isoformat()})"
            )
    else:
        confidence = evidence.match_confidence
        if eight_k_evs:
            ordered = sorted(
                eight_k_evs,
                key=lambda ev: (_as_date(ev.filing.filed), ev.filing.form, ev.filing.file_path),
            )
            evidence_str = f"CIK {evidence.cik}: " + "; ".join(
                _format_filing_evidence(ev) for ev in ordered[:3]
            )
        else:
            evidence_str = f"CIK {evidence.cik}: no qualifying filing"

    suggested_loss = 0.0 if label == early_sec.LABEL_ACQUISITION_TYPE else float("nan")
    return label, confidence, evidence_str, suggested_loss


def classify_roster(
    roster: pd.DataFrame,
    names: pd.DataFrame | None,
    sec_lookup: Any,
) -> pd.DataFrame:
    """Classify every stock in ``roster`` using ``names`` and the injected ``sec_lookup``.

    Returns a DataFrame with one row per roster row and exact ``OUTPUT_COLUMNS``:
    ``column``, ``permno``, ``symbol``, ``last_trading_day``, ``company_name``, ``name_source``,
    ``exchange``, ``rank_band``, ``held_by_rule``, ``label``, ``confidence``, ``evidence``,
    ``suggested_loss`` (``0.0`` only for ``acquisition_type``, else ``NaN`` / empty in CSV).
    """
    missing = [col for col in REQUIRED_ROSTER_COLUMNS if col not in roster.columns]
    if missing:
        raise ValueError(f"roster is missing required columns {missing}")

    if roster.empty:
        return pd.DataFrame(
            {
                "column": pd.Series(dtype=object),
                "permno": pd.Series(dtype="Int64"),
                "symbol": pd.Series(dtype=object),
                "last_trading_day": pd.Series(dtype=object),
                "company_name": pd.Series(dtype=object),
                "name_source": pd.Series(dtype=object),
                "exchange": pd.Series(dtype=object),
                "rank_band": pd.Series(dtype=object),
                "held_by_rule": pd.Series(dtype=bool),
                "label": pd.Series(dtype=object),
                "confidence": pd.Series(dtype=object),
                "evidence": pd.Series(dtype=object),
                "suggested_loss": pd.Series(dtype="float64"),
            },
            columns=OUTPUT_COLUMNS,
        )

    merged = merge_names(roster, names)
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        last_day = _as_date(row["last_bar"])
        company_name = _clean_text(row.get("company_name"))
        name_source = _clean_text(row.get("name_source")) or ""
        exchange = exchange_label(row.get("exchcd"), row.get("exchange"))

        label, confidence, evidence_str, suggested_loss = _classify_one(
            company_name, last_day, sec_lookup
        )
        records.append(
            {
                "column": str(row["column"]),
                "permno": int(row["permno"]) if pd.notna(row["permno"]) else pd.NA,
                "symbol": str(row["symbol"]),
                "last_trading_day": last_day.isoformat(),
                "company_name": company_name if company_name is not None else "",
                "name_source": name_source,
                "exchange": exchange,
                "rank_band": str(row["rank_band"]),
                "held_by_rule": bool(row["held_by_rule"]),
                "label": label,
                "confidence": confidence,
                "evidence": evidence_str,
                "suggested_loss": suggested_loss,
            }
        )

    out = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    out["permno"] = out["permno"].astype("Int64")
    out["held_by_rule"] = out["held_by_rule"].astype(bool)
    out["suggested_loss"] = out["suggested_loss"].astype("float64")
    return out


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().to_dict() if not frame.empty else {}
    return {label: int(counts.get(label, 0)) for label in REGISTERED_LABELS}


def _confidence_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {**{tier: 0 for tier in CONFIDENCE_TIERS}, "none": 0}
    series = frame["confidence"].fillna("").astype(str)
    counts = series.value_counts().to_dict()
    result = {tier: int(counts.get(tier, 0)) for tier in CONFIDENCE_TIERS}
    result["none"] = int(counts.get("", 0))
    return result


def _group_breakdown(
    frame: pd.DataFrame,
    key_series: pd.Series,
    *,
    ordered_keys: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    keys = list(ordered_keys) if ordered_keys is not None else []
    observed = sorted({str(k) for k in key_series})
    for k in observed:
        if k not in keys:
            keys.append(k)
    result: dict[str, dict[str, Any]] = {}
    for k in keys:
        sub = frame[key_series.astype(str) == k]
        if sub.empty and ordered_keys is None:
            continue
        result[k] = {
            "total": int(len(sub)),
            "by_label": _label_counts(sub),
        }
    return result


def _subset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    years = (
        frame["last_trading_day"].astype(str).str.slice(0, 4)
        if not frame.empty
        else pd.Series(dtype=object)
    )
    exchanges = frame["exchange"].astype(str) if not frame.empty else pd.Series(dtype=object)
    rank_bands = frame["rank_band"].astype(str) if not frame.empty else pd.Series(dtype=object)
    return {
        "total": int(len(frame)),
        "by_label": _label_counts(frame),
        "by_confidence": _confidence_counts(frame),
        "by_year": _group_breakdown(frame, years),
        "by_exchange": _group_breakdown(frame, exchanges),
        "by_rank_band": _group_breakdown(frame, rank_bands, ordered_keys=_CANONICAL_RANK_BANDS),
    }


def summarize_classifications(classified: pd.DataFrame) -> dict[str, Any]:
    """Compute traceable counts by label, confidence, year, exchange, rank band, and held flag."""
    summary = _subset_summary(classified)
    held = (
        classified[classified["held_by_rule"].astype(bool)] if not classified.empty else classified
    )
    not_held = (
        classified[~classified["held_by_rule"].astype(bool)] if not classified.empty else classified
    )
    summary["by_held_by_rule"] = {
        "held": {"total": int(len(held)), "by_label": _label_counts(held)},
        "not_held": {"total": int(len(not_held)), "by_label": _label_counts(not_held)},
    }
    summary["held_by_rule"] = _subset_summary(held)
    return summary


__all__ = [
    "CONFIDENCE_TIERS",
    "EXCHANGE_NAMES",
    "OUTPUT_COLUMNS",
    "PILOT_SEED",
    "REGISTERED_LABELS",
    "REQUIRED_ROSTER_COLUMNS",
    "CompanyEvidence",
    "FilingEvidence",
    "SecLookupFn",
    "classify_roster",
    "exchange_label",
    "merge_names",
    "select_pilot_rows",
    "summarize_classifications",
]
