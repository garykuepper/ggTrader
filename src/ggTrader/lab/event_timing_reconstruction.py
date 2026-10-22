"""Public-time Form 4 replay and metadata reconciliation for the timing audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd


def stored_key(lot: dict, owner: str) -> tuple | None:
    """Match the actual stored PK, including transaction_code and shares."""
    if not str(owner).isdigit() or not lot.get("accession"):
        return None
    try:
        shares: Any = Decimal(str(lot.get("shares"))).normalize()
    except (InvalidOperation, ValueError):
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
    unassignable = []
    for lot in lots:
        if not lot.get("raw_lot_sha256") or not lot.get("lot_id") or not lot.get("owners"):
            unassignable.append({"lot": lot, "reason": "missing raw fingerprint/identity"})
        for owner in lot.get("owners", []):
            key = stored_key(lot, owner)
            if key is None:
                unassignable.append({"lot": lot, "owner": owner, "reason": "invalid owner/key"})
            else:
                keys.setdefault(key, []).append(lot)
    stored_keys = Counter()
    for row in stored:
        key = stored_key(row, row.get("insider_cik"))
        if key is None:
            unassignable.append({"stored_row": row, "reason": "invalid stored owner/key"})
        else:
            stored_keys[key] += 1
    collisions = []
    exact_duplicates = []
    for key, group in keys.items():
        fingerprints = Counter(lot.get("raw_lot_sha256") for lot in group)
        if len(group) > 1:
            if len(fingerprints) == 1:
                exact_duplicates.append(
                    {"key": list(map(str, key)), "lot_ids": [r.get("lot_id") for r in group]}
                )
            else:
                collisions.append(
                    {"key": list(map(str, key)), "lot_ids": [r.get("lot_id") for r in group]}
                )
    missing = [
        {"lot_id": r.get("lot_id"), "owner": key[1]}
        for key, group in keys.items()
        for r in group
        if stored_keys[key] == 0 and r.get("lot_id")
    ]
    return {
        "raw_lots": len(lots),
        "stored_rows": len(stored),
        "collisions": collisions,
        "exact_duplicate_candidates": exact_duplicates,
        "missing_stored_lots": sorted({r["lot_id"] for r in missing}),
        "missing_stored_owner_lots": [
            {"lot_id": lot_id, "owner": owner}
            for lot_id, owner in sorted({(r["lot_id"], r["owner"]) for r in missing})
        ],
        "raw_lot_ids": [r.get("lot_id") for r in lots],
        "unassignable_rows": unassignable,
        "predicate": "UNVERIFIED" if unassignable else "PASS",
        "representation": "raw lots retained; stored losses are Phase 2 build requirements",
    }


def known_time(lot: dict) -> pd.Timestamp | None:
    """Observed public time or a separately labelled conservative date-only execution bound."""
    value = lot.get("availability") or lot.get("decision_eligibility_time")
    return pd.Timestamp(value) if value is not None else None


def cluster_snapshot(
    lots: list[dict],
    decision: pd.Timestamp,
    amendments: list[dict],
    deadline: Callable[[], bool] | None = None,
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
        if deadline is not None and not deadline():
            return [], issues + ["runtime budget exhausted"]
        if row.get("reconstruction_issue"):
            issues.append(f"invalid raw input: {row.get('lot_id', 'unassignable')}")
            continue
        if row["transaction_code"] != "P" or row["plan_state"] == "PRESENT-TRUE":
            continue
        if row["plan_state"] not in ("PRESENT-FALSE", "ABSENT"):
            issues.append(f"non-plan status unknown: {row['lot_id']}")
            continue
        owners = row.get("owners", [])
        if not owners or any(not str(o).isdigit() for o in owners):
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
            if deadline is not None and not deadline():
                return events, issues + ["runtime budget exhausted"]
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
    progress: dict | None = None,
    save_progress: Callable[[], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Freeze emitted signals while replaying each security's disclosures in public-time order."""
    ledger: dict[str, dict] = {}
    issues = []
    by_symbol: dict[str, list[dict]] = {}
    for row in lots:
        if deadline is not None and not deadline():
            return [], ["runtime budget exhausted"]
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
        saved = progress.get(symbol) if progress is not None else None
        if saved is not None and saved.get("complete"):
            ledger.update({e["event_id"]: e for e in saved["events"]})
            issues.extend(saved["issues"])
            continue
        last_time = pd.Timestamp(saved["through"]) if saved and saved.get("through") else None
        if saved is not None:
            ledger.update({e["event_id"]: e for e in saved["events"]})
            issues.extend(saved["issues"])
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
            if last_time is not None and time <= last_time:
                continue
            snapshot, errors = cluster_snapshot(visible, time, actions, deadline)
            if "runtime budget exhausted" in errors:
                return list(ledger.values()), sorted(set(issues + errors))
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
            if progress is not None:
                progress[symbol] = {
                    "complete": False,
                    "through": time.isoformat(),
                    "events": [e for e in ledger.values() if e["symbol"] == symbol],
                    "issues": list(issues),
                }
                if save_progress is not None:
                    save_progress()
        if progress is not None:
            progress[symbol] = {
                "complete": True,
                "events": [e for e in ledger.values() if e["symbol"] == symbol],
                "issues": list(issues),
            }
            if save_progress is not None:
                save_progress()
    return sorted(ledger.values(), key=lambda e: (e["formed_at"], e["event_id"])), sorted(
        set(issues)
    )


def evaluate_prefix_invariance(
    lots: list[dict],
    amendments: list[dict],
    clock: Any,
    complete_by_month: dict[str, bool],
    deadline: Callable[[], bool] | None = None,
    progress: dict | None = None,
    save_progress: Callable[[], None] | None = None,
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
    if progress is None:
        progress = {}
    full, full_issues = replay_clusters(
        lots, amendments, horizon, deadline, progress.setdefault("full", {}), save_progress
    )
    for period in pd.period_range("2015-01", "2026-04", freq="M"):
        saved = progress.get(str(period), {})
        if saved.get("complete"):
            row = saved["row"]
            decisions.append(row)
            annual.setdefault(period.year, []).append(row["predicate"])
            continue
        expired = deadline is not None and not deadline()
        decision = (
            None
            if expired
            else clock.get_schedule(
                str(period.start_time.date()), str(period.end_time.date())
            ).iloc[-1]["market_close"]
        )
        if not complete_by_month.get(str(period), False) or expired:
            row = {
                "month": str(period),
                "decision": decision.isoformat() if decision is not None else None,
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
            predicate = "FAIL" if not equal else "UNVERIFIED" if issues else "PASS"
            row = {
                "month": str(period),
                "decision": decision.isoformat(),
                "predicate": predicate,
                "full_outputs": historical,
                "prefix_outputs": prefix,
                "issues": issues,
                "reason": "identical historical ledger"
                if historical
                else "verified-empty prefix compared",
            }
        decisions.append(row)
        annual.setdefault(period.year, []).append(row["predicate"])
        if deadline is None or deadline():
            progress[str(period)] = {"complete": True, "row": row}
            if save_progress is not None:
                save_progress()
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


def filing_lags(
    lots: list[dict],
    clock: Any,
    deadline: Callable[[], bool] | None = None,
    progress: dict | None = None,
    save_progress: Callable[[], None] | None = None,
) -> dict:
    """Count NYSE sessions after transaction through filing on quarantined-free raw lots."""
    if deadline is not None and not deadline():
        return {
            "observation_unit": "raw transaction lot",
            "by_filing_year": {},
            "pending": "runtime budget exhausted",
        }
    rows = [r for r in lots if not r.get("quarantine")]
    if not rows:
        return {"observation_unit": "raw transaction lot", "by_filing_year": {}}
    sessions = clock.get_schedule("1990-01-01", max(str(r["filing_date"]) for r in rows)).index
    result = {}
    for year in sorted({pd.Timestamp(r["filing_date"]).year for r in rows}):
        if deadline is not None and not deadline():
            return {
                "observation_unit": "raw transaction lot",
                "by_filing_year": result,
                "pending": "runtime budget exhausted",
            }
        if progress is not None and str(year) in progress:
            result[year] = progress[str(year)]
            continue
        group = [r for r in rows if pd.Timestamp(r["filing_date"]).year == year]
        values = pd.Series(
            sessions.searchsorted(pd.DatetimeIndex([r["filing_date"] for r in group]), side="right")
            - sessions.searchsorted(
                pd.DatetimeIndex([r["transaction_date"] for r in group]), side="right"
            )
        )
        result[year] = {
            "count": len(values),
            "median_sessions": float(values.median()),
            "p95_sessions": float(values.quantile(0.95)),
            "max_sessions": int(values.max()),
            "share_over_two_sessions": float((values > 2).mean()),
            "lot_ids": [r["lot_id"] for r in group],
        }
        if progress is not None:
            progress[str(year)] = result[year]
            if save_progress is not None:
                save_progress()
    return {"observation_unit": "raw transaction lot", "by_filing_year": result}


def load_boundaries(
    lots: list[dict],
    clock: Any,
    deadline: Callable[[], bool] | None = None,
    progress: dict | None = None,
    save_progress: Callable[[], None] | None = None,
) -> list[dict]:
    """Inventory disclosures the existing transaction-date loader would omit at each A9 close."""
    result = []
    clean = [r for r in lots if not r.get("quarantine") and known_time(r) is not None]
    for period in pd.period_range("2015-01", "2026-04", freq="M"):
        if deadline is not None and not deadline():
            break
        if progress is not None and str(period) in progress:
            result.extend(progress[str(period)])
            continue
        start = len(result)
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
        if progress is not None:
            progress[str(period)] = result[start:]
            if save_progress is not None:
                save_progress()
    return result
