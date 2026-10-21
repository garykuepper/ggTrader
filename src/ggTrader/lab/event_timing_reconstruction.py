"""Public-time Form 4 replay and metadata reconciliation for the timing audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd


def stored_key(lot: dict, owner: str) -> tuple:
    """Match the actual stored PK, including transaction_code and shares."""
    try:
        shares: Any = Decimal(str(lot.get("shares"))).normalize()
    except InvalidOperation:
        shares = lot.get("shares")
    return (
        lot["accession"],
        str(int(owner)),
        str(lot.get("transaction_date")),
        lot.get("transaction_code"),
        shares,
    )


def reconcile_lots(lots: list[dict], stored: list[dict]) -> dict:
    """Distinguish raw duplicates and distinct lots colliding under the stored key."""
    keys: dict[tuple, list[dict]] = {}
    for lot in lots:
        for owner in lot.get("owners", []):
            if owner is not None:
                keys.setdefault(stored_key(lot, owner), []).append(lot)
    stored_keys = Counter(stored_key(r, str(r["insider_cik"])) for r in stored)
    collisions = []
    exact_duplicates = []
    for key, group in keys.items():
        fingerprints = Counter(lot["raw_lot_sha256"] for lot in group)
        if len(group) > 1:
            if len(fingerprints) == 1:
                exact_duplicates.append(
                    {"key": list(map(str, key)), "lot_ids": [r["lot_id"] for r in group]}
                )
            else:
                collisions.append(
                    {"key": list(map(str, key)), "lot_ids": [r["lot_id"] for r in group]}
                )
    missing = [
        r["lot_id"]
        for group in keys.values()
        for r in group
        if stored_keys[stored_key(r, next(o for o in r["owners"] if o is not None))] == 0
    ]
    return {
        "raw_lots": len(lots),
        "stored_rows": len(stored),
        "collisions": collisions,
        "exact_duplicate_candidates": exact_duplicates,
        "missing_stored_lots": sorted(set(missing)),
        "raw_lot_ids": [r["lot_id"] for r in lots],
        "predicate": "PASS" if all(lot.get("raw_lot_sha256") for lot in lots) else "UNVERIFIED",
        "representation": "raw lots retained; stored losses are Phase 2 build requirements",
    }


def known_time(lot: dict) -> pd.Timestamp | None:
    """Observed public time or a separately labelled conservative date-only execution bound."""
    value = lot.get("availability") or lot.get("decision_eligibility_time")
    return pd.Timestamp(value) if value is not None else None


def cluster_snapshot(
    lots: list[dict], decision: pd.Timestamp, amendments: list[dict]
) -> tuple[list[dict], list[str]]:
    """Reconstruct the 14-day, three-owner signal using only then-public source lots."""
    issues = []
    visible = [
        r
        for r in lots
        if not r.get("quarantine") and known_time(r) is not None and known_time(r) <= decision
    ]
    active = {r["lot_id"]: r for r in visible}
    amendment_times = {r["accession"]: known_time(r) for r in visible if r["form"] == "4/A"}
    for action in amendments:
        time = known_time(action)
        if time is not None and time <= decision:
            amendment_times[action["accession"]] = time
    for accession, public_time in sorted(
        amendment_times.items(), key=lambda pair: (pair[1], pair[0])
    ):
        links = [a for a in amendments if a.get("accession") == accession]
        if len(links) != 1 or not links[0].get("source_refs") or links[0].get("linkage_unverified"):
            issues.append(f"amendment linkage unresolved: {accession}")
            active = {k: r for k, r in active.items() if r["accession"] != accession}
            continue
        link = links[0]
        if link.get("no_signal_input_change"):
            active = {k: r for k, r in active.items() if r["accession"] != accession}
            continue
        replacements = link.get("replaces_lot_ids", [])
        if not replacements or any(
            k not in active or known_time(active[k]) >= public_time for k in replacements
        ):
            issues.append(f"amendment target unavailable or not earlier: {accession}")
            active = {k: r for k, r in active.items() if r["accession"] != accession}
            continue
        for key in replacements:
            active.pop(key)
    usable = []
    for row in active.values():
        if row["transaction_code"] != "P" or row["plan_state"] == "PRESENT-TRUE":
            continue
        if row["plan_state"] not in ("PRESENT-FALSE", "ABSENT"):
            issues.append(f"non-plan status unknown: {row['lot_id']}")
            continue
        owners = row.get("owners", [])
        if not owners or any(o is None for o in owners):
            issues.append(f"reporting owner missing: {row['lot_id']}")
            continue
        usable.append(row)
    events = []
    groups = {(r["symbol"], str(r.get("issuer_cik", "")), r.get("security_id", "")) for r in usable}
    for symbol, issuer_cik, security_id in sorted(groups):
        window: list[dict] = []
        rows = sorted(
            (
                r
                for r in usable
                if (r["symbol"], str(r.get("issuer_cik", "")), r.get("security_id", ""))
                == (symbol, issuer_cik, security_id)
            ),
            key=lambda r: (str(r["transaction_date"]), r["lot_id"]),
        )
        for row in rows:
            date = pd.Timestamp(row["transaction_date"])
            window = [r for r in window if (date - pd.Timestamp(r["transaction_date"])).days <= 14]
            window.append(row)
            owners = {str(o) for r in window for o in r["owners"]}
            if len(owners) >= 3:
                members = sorted(r["lot_id"] for r in window)
                precise = all(r.get("availability") for r in window)
                availability = (
                    max(pd.Timestamp(r["availability"]) for r in window) if precise else None
                )
                closes = [r["tradeable_close"] for r in window if r.get("tradeable_close")]
                events.append(
                    {
                        "event_id": f"{symbol}|{'|'.join(members)}",
                        "symbol": symbol,
                        "issuer_cik": issuer_cik,
                        "security_id": security_id,
                        "availability": availability.isoformat()
                        if availability is not None
                        else None,
                        "formed_at": max(known_time(r) for r in window).isoformat(),
                        "tradeable_close": max(closes) if len(closes) == len(window) else None,
                        "precision": "timestamp" if precise else "date_only",
                        "lot_ids": members,
                        "owners": sorted(owners),
                        "absent_plan_lots": [
                            r["lot_id"] for r in window if r["plan_state"] == "ABSENT"
                        ],
                        "owner_independence": "Distinct CIKs; related-party independence unknown",
                        "purchase_code_scope": "P includes private purchases; existing filter",
                        "quarantine": None,
                    }
                )
                window = []
    return events, sorted(set(issues))


def replay_clusters(
    lots: list[dict],
    amendments: list[dict],
    decision: pd.Timestamp,
    deadline: Callable[[], bool] | None = None,
) -> tuple[list[dict], list[str]]:
    """Freeze emitted signals while replaying each security's disclosures in public-time order."""
    ledger: dict[str, dict] = {}
    issues = []
    by_symbol: dict[str, list[dict]] = {}
    for row in lots:
        if (
            not row.get("quarantine")
            and known_time(row) is not None
            and known_time(row) <= decision
        ):
            by_symbol.setdefault(row["symbol"], []).append(row)
    for action in amendments:
        if (
            action.get("symbol")
            and known_time(action) is not None
            and known_time(action) <= decision
        ):
            by_symbol.setdefault(action["symbol"], [])
    for symbol, rows in sorted(by_symbol.items()):
        visible = []
        ids = {r["lot_id"] for r in rows}
        accessions = {r["accession"] for r in rows}
        actions = [
            a
            for a in amendments
            if a.get("symbol") == symbol
            or a["accession"] in accessions
            or ids.intersection(a.get("replaces_lot_ids", []))
        ]
        changes: dict[pd.Timestamp, list[dict]] = {}
        for row in rows:
            changes.setdefault(known_time(row), []).append(row)
        for action in actions:
            if known_time(action) is not None and known_time(action) <= decision:
                changes.setdefault(known_time(action), [])
        for time, group in sorted(changes.items()):
            if deadline is not None and not deadline():
                return list(ledger.values()), sorted(set(issues + ["runtime budget exhausted"]))
            visible.extend(group)
            snapshot, errors = cluster_snapshot(visible, time, actions)
            issues.extend(errors)
            current = [a for a in actions if known_time(a) == time]
            closes = [r["tradeable_close"] for r in group + current if r.get("tradeable_close")]
            for event in snapshot:
                if (
                    event["event_id"] not in ledger
                    and current
                    and any(set(event["lot_ids"]) & set(old["lot_ids"]) for old in ledger.values())
                ):
                    issues.extend(
                        f"amendment re-signalling definition unresolved: {a['accession']}"
                        for a in current
                    )
                event["formed_at"] = max(time, pd.Timestamp(event["formed_at"])).isoformat()
                if closes and event.get("tradeable_close"):
                    event["tradeable_close"] = max([event["tradeable_close"], *closes])
                if current:
                    precise = event.get("availability") and all(
                        a.get("availability") for a in current
                    )
                    event["availability"] = (
                        max(time, pd.Timestamp(event["availability"])).isoformat()
                        if precise
                        else None
                    )
                    event["precision"] = "timestamp" if precise else "date_only"
                ledger.setdefault(event["event_id"], event)
    return sorted(ledger.values(), key=lambda e: (e["formed_at"], e["event_id"])), sorted(
        set(issues)
    )


def evaluate_prefix_invariance(
    lots: list[dict],
    amendments: list[dict],
    clock: Any,
    complete_by_month: dict[str, bool],
    deadline: Callable[[], bool] | None = None,
) -> dict:
    """Compare full-history ledger with each independently truncated A9 information set."""
    decisions = []
    annual: dict[int, list[str]] = {}
    horizon = max(
        (
            known_time(r)
            for r in lots + amendments
            if not r.get("quarantine") and known_time(r) is not None
        ),
        default=clock.normalize_to_ny("2026-04-30 16:00"),
    )
    full, full_issues = replay_clusters(lots, amendments, horizon, deadline)
    for period in pd.period_range("2015-01", "2026-04", freq="M"):
        schedule = clock.get_schedule(str(period.start_time.date()), str(period.end_time.date()))
        decision = schedule.iloc[-1]["market_close"]
        if not complete_by_month.get(str(period), False) or deadline is not None and not deadline():
            row = {
                "month": str(period),
                "decision": decision.isoformat(),
                "predicate": "UNVERIFIED",
                "reason": "incomplete source inventory/raw inputs or runtime cap",
            }
        else:
            historical = [e for e in full if pd.Timestamp(e["formed_at"]) <= decision]
            truncated = [r for r in lots if known_time(r) is not None and known_time(r) <= decision]
            prefix_actions = [
                a for a in amendments if known_time(a) is None or known_time(a) <= decision
            ]
            prefix, issues = replay_clusters(truncated, prefix_actions, decision, deadline)
            equal = historical == prefix
            unexercised = bool(truncated) and not historical
            predicate = "FAIL" if not equal else "UNVERIFIED" if issues or unexercised else "PASS"
            row = {
                "month": str(period),
                "decision": decision.isoformat(),
                "predicate": predicate,
                "full_outputs": historical,
                "prefix_outputs": prefix,
                "issues": issues,
                "reason": "identical historical ledger"
                if historical
                else "nonempty inputs produced no cluster"
                if unexercised
                else "verified-empty prefix compared",
            }
        decisions.append(row)
        annual.setdefault(period.year, []).append(row["predicate"])
    return {
        "decisions": decisions,
        "full_history_issues": full_issues,
        "annual": {
            year: (
                "FAIL" if "FAIL" in values else "PASS" if set(values) == {"PASS"} else "UNVERIFIED"
            )
            for year, values in annual.items()
        },
    }


def filing_lags(lots: list[dict], clock: Any) -> dict:
    """Count NYSE sessions after transaction through filing on quarantined-free raw lots."""
    rows = [r for r in lots if not r.get("quarantine")]
    if not rows:
        return {"observation_unit": "raw transaction lot", "by_filing_year": {}}
    sessions = clock.get_schedule("1990-01-01", max(str(r["filing_date"]) for r in rows)).index
    result = {}
    for year in sorted({pd.Timestamp(r["filing_date"]).year for r in rows}):
        group = [r for r in rows if pd.Timestamp(r["filing_date"]).year == year]
        values = pd.Series(
            [
                int(
                    sessions.searchsorted(pd.Timestamp(r["filing_date"]), side="right")
                    - sessions.searchsorted(pd.Timestamp(r["transaction_date"]), side="right")
                )
                for r in group
            ]
        )
        result[year] = {
            "count": len(values),
            "median_sessions": float(values.median()),
            "p95_sessions": float(values.quantile(0.95)),
            "max_sessions": int(values.max()),
            "share_over_two_sessions": float((values > 2).mean()),
            "lot_ids": [r["lot_id"] for r in group],
        }
    return {"observation_unit": "raw transaction lot", "by_filing_year": result}


def load_boundaries(lots: list[dict], clock: Any) -> list[dict]:
    """Inventory disclosures the existing transaction-date loader would omit at each A9 close."""
    result = []
    clean = [r for r in lots if not r.get("quarantine") and known_time(r) is not None]
    for period in pd.period_range("2015-01", "2026-04", freq="M"):
        decision = clock.get_schedule(
            str(period.start_time.date()), str(period.end_time.date())
        ).iloc[-1]["market_close"]
        for hold in (126, 189, 252):
            boundary = (decision - pd.Timedelta(days=hold + 14 + 30)).date()
            omitted = [
                r["lot_id"]
                for r in clean
                if known_time(r) <= decision
                and pd.Timestamp(r["transaction_date"]).date() < boundary
                and known_time(r).date() >= boundary
            ]
            result.append(
                {
                    "decision": decision.isoformat(),
                    "hold_days": hold,
                    "transaction_date_load_boundary": str(boundary),
                    "omitted_disclosure_lot_ids": omitted,
                    "raw_public_time_inputs_retain_them": True,
                }
            )
    return result
