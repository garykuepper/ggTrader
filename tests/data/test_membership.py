"""Fast tests of ``ggTrader.data.membership``: selection, fixtures, the DB source (in memory), age.

No database is needed: the DB source is exercised through an in-memory store, and the one test
that builds a real ``PgMembershipStore`` points it at a closed local port.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import shutil
import traceback
import warnings
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ggTrader.cli import cmd_membership
from ggTrader.cli.cmd_db import register_db_parser
from ggTrader.data import membership
from ggTrader.data.membership import (
    DbMembershipSource,
    FixtureMembershipSource,
    MembershipIntegrityError,
    MembershipSourceNotSelected,
    MembershipSpecError,
    MembershipStaleError,
    MembershipStaleWarning,
    MembershipUnavailable,
    PgMembershipStore,
    check_membership_age,
    load_manifest,
    membership_source,
    parse_spec,
    redact,
)
from ggTrader.data.universe_algebra import Interval
from ggTrader.utils.paths import REPO_ROOT
from tests.data.membership_fakes import (
    BASE_TIME,
    FailingStore,
    FakeMembershipStore,
    synthetic_history,
)

V1 = "sp500_v1_2026-06-02"
V1_GZ_SHA = "b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472"
V1_RAW_SHA = "bc9c2aaed27e70406a247cd3f75513e786b0f173c70c3e05c1e243bf25498787"
V1_CONTENT_SHA = "256e022dc1580a82aaef38d303a772611469bc6a705374704a79333eefcef71c"
V1_FILE = REPO_ROOT / "data" / "universe" / "sp500_constituents_history.csv.gz"
MANIFEST_DIR = REPO_ROOT / "data" / "universe" / "membership"
D = date.fromisoformat


@pytest.fixture(autouse=True)
def _clean_selection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with nothing selected and no database configuration."""
    for name in (
        "GGT_MEMBERSHIP",
        "GGT_MEMBERSHIP_DB_URL",
        "POSTGRES_CONNECTION_STRING",
        "DB_HOST",
        "DB_USER",
        "DB_PASS",
        "DB_NAME",
        "DB_PORT",
    ):
        monkeypatch.delenv(name, raising=False)


# ---- the frozen vintage-1 file and its manifest ---------------------------------------------


def test_the_vintage_one_file_bytes_have_not_changed() -> None:
    """The frozen fixture is pinned by bytes: any edit to the file fails here (decision D4)."""
    data = V1_FILE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == V1_GZ_SHA
    assert hashlib.sha256(gzip.decompress(data)).hexdigest() == V1_RAW_SHA


def test_manifest_lists_the_vintage_one_alias_with_the_pinned_hashes() -> None:
    entry = load_manifest()[V1]
    assert (entry.path, entry.gz_sha256, entry.raw_sha256, entry.content_sha256) == (
        "data/universe/sp500_constituents_history.csv.gz",
        V1_GZ_SHA,
        V1_RAW_SHA,
        V1_CONTENT_SHA,
    )
    assert entry.db_load_id_on_encom == 1
    assert (entry.coverage_start, entry.coverage_end) == (D("1996-01-02"), D("2026-06-02"))
    assert entry.n_snapshot_dates == 2712


def test_every_manifest_entry_exists_and_matches_its_hashes() -> None:
    for name, entry in load_manifest().items():
        source = FixtureMembershipSource(name)  # checks gz, raw and content hashes and coverage
        assert source.vintage().content_sha256 == entry.content_sha256


def _manifest_events() -> dict[str, dict[str, str]]:
    return json.loads((MANIFEST_DIR / "MANIFEST.json").read_text(encoding="utf-8"))["events"]


def test_no_unlisted_file_sits_in_the_manifest_folder() -> None:
    listed = {(REPO_ROOT / e.path).resolve() for e in load_manifest().values()}
    listed |= {(REPO_ROOT / e["path"]).resolve() for e in _manifest_events().values()}
    stray = [
        str(p.relative_to(MANIFEST_DIR))
        for p in MANIFEST_DIR.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json" and p.resolve() not in listed
    ]
    assert stray == []


def test_every_listed_events_file_matches_its_hash_and_parses() -> None:
    from ggTrader.data.membership_events import parse_events_csv

    for name, entry in _manifest_events().items():
        raw = (REPO_ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
        events = parse_events_csv(raw)  # every event carries a url, a date and an announcement
        assert all(e.evidence_url.startswith("https://") and e.evidence_sha256 for e in events)
        assert len(entry["expected_content_sha256"]) == 64


def test_manifest_file_is_valid_json_with_sorted_keys() -> None:
    text = (MANIFEST_DIR / "MANIFEST.json").read_text(encoding="utf-8")
    document = json.loads(text)
    assert list(document["fixtures"]) == sorted(document["fixtures"])


def _scratch_tree(tmp_path: Path) -> Path:
    """A copy of the manifest and the v1 file under ``tmp_path`` laid out like the repo."""
    folder = tmp_path / "data" / "universe" / "membership"
    folder.mkdir(parents=True)
    shutil.copy(MANIFEST_DIR / "MANIFEST.json", folder / "MANIFEST.json")
    shutil.copy(V1_FILE, tmp_path / "data" / "universe" / V1_FILE.name)
    return tmp_path


def _edit_manifest(root: Path, **changes: object) -> None:
    path = root / "data" / "universe" / "membership" / "MANIFEST.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fixtures"][V1].update(changes)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_a_scratch_copy_of_the_tree_loads(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    assert FixtureMembershipSource(V1, root=root).vintage().content_sha256 == V1_CONTENT_SHA


def test_a_changed_fixture_file_is_refused(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    target = root / "data" / "universe" / V1_FILE.name
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(MembershipIntegrityError, match="gz_sha256"):
        FixtureMembershipSource(V1, root=root)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("raw_sha256", "raw sha256"),
        ("content_sha256", "content sha256"),
        ("n_snapshot_dates", "coverage or date count"),
        ("coverage", "coverage or date count"),
    ],
)
def test_a_manifest_that_disagrees_with_the_file_is_refused(
    tmp_path: Path, field: str, match: str
) -> None:
    root = _scratch_tree(tmp_path)
    bad: object = {
        "raw_sha256": "0" * 64,
        "content_sha256": "1" * 64,
        "n_snapshot_dates": 2711,
        "coverage": {"start": "1996-01-03", "end": "2026-06-02"},
    }[field]
    _edit_manifest(root, **{field: bad})
    with pytest.raises(MembershipIntegrityError, match=match):
        FixtureMembershipSource(V1, root=root)


def test_a_missing_fixture_file_or_manifest_is_unavailable(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    (root / "data" / "universe" / V1_FILE.name).unlink()
    with pytest.raises(MembershipUnavailable, match="cannot read"):
        FixtureMembershipSource(V1, root=root)
    with pytest.raises(MembershipUnavailable, match="manifest"):
        FixtureMembershipSource(V1, root=tmp_path / "nowhere")


def test_a_malformed_manifest_entry_is_refused(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    _edit_manifest(root, gz_sha256="not-a-hash")
    with pytest.raises(MembershipIntegrityError, match="not a sha256"):
        load_manifest(root)


def test_unknown_fixture_name_lists_the_known_ones() -> None:
    with pytest.raises(MembershipSpecError, match=V1):
        FixtureMembershipSource("sp500_v9")


# ---- selection ------------------------------------------------------------------------------


def test_nothing_selected_raises_with_instructions() -> None:
    with pytest.raises(MembershipSourceNotSelected, match="GGT_MEMBERSHIP") as err:
        membership_source()
    assert "db:<load_id>" in str(err.value)


def test_an_empty_spec_and_an_empty_variable_count_as_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GGT_MEMBERSHIP", "")
    with pytest.raises(MembershipSourceNotSelected):
        membership_source("")


def test_the_environment_variable_selects_when_no_spec_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GGT_MEMBERSHIP", f"fixture:{V1}")
    assert membership_source().vintage().fixture == V1


def test_an_explicit_spec_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeMembershipStore()
    store.add(synthetic_history())
    monkeypatch.setenv("GGT_MEMBERSHIP", f"fixture:{V1}")
    assert membership_source("db", store=store).vintage().kind == "db"


@pytest.mark.parametrize(
    ("spec", "parsed"),
    [
        ("db", ("db", None)),
        ("db:1", ("db", 1)),
        ("db:42", ("db", 42)),
        (f"fixture:{V1}", ("fixture", V1)),
    ],
)
def test_parse_spec(spec: str, parsed: tuple[str, object]) -> None:
    assert parse_spec(spec) == parsed


@pytest.mark.parametrize(
    "spec",
    ["latest", "DB", "db:", "db:0", "db:-1", "db:abc", "db:1.5", "fixture:", "fixture:a/b",
     "file:x.csv", "fixture:a b", "db latest"],
)  # fmt: skip
def test_bad_specs_are_refused(spec: str) -> None:
    with pytest.raises(MembershipSpecError):
        membership_source(spec)


# ---- the database source, in memory ---------------------------------------------------------


def test_latest_resolves_to_one_load_id_once_at_construction() -> None:
    store = FakeMembershipStore()
    first = store.add(synthetic_history(1))
    source = membership_source("db", store=store)
    assert source.vintage().load_id == first == 1
    second = store.add(synthetic_history(2))  # a load that lands after the source was built
    assert source.vintage().load_id == 1
    assert source.members_asof(D("2000-01-03")) == list(synthetic_history(1)[D("2000-01-03")])
    assert store.read_calls == [None]  # one read, with "latest"
    assert membership_source("db", store=store).vintage().load_id == second


def test_a_pinned_vintage_reads_exactly_that_load() -> None:
    store = FakeMembershipStore()
    store.add(synthetic_history(1))
    store.add(synthetic_history(2))
    pinned = membership_source("db:1", store=store)
    assert pinned.vintage().load_id == 1
    assert store.read_calls == [1]
    assert (
        pinned.vintage().content_sha256
        != membership_source("db", store=store).vintage().content_sha256
    )


def test_a_missing_vintage_is_unavailable_not_silently_replaced() -> None:
    store = FakeMembershipStore()
    store.add(synthetic_history(1))
    with pytest.raises(MembershipUnavailable, match="no load 9"):
        membership_source("db:9", store=store)


def test_an_unreadable_database_raises_and_never_reads_a_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a fixture was read after the database failed")

    monkeypatch.setattr(membership, "load_manifest", forbidden)
    monkeypatch.setattr(FixtureMembershipSource, "__init__", forbidden)
    monkeypatch.setenv("GGT_MEMBERSHIP", "db")
    with pytest.raises(MembershipUnavailable):
        membership_source(store=FailingStore())
    with pytest.raises(MembershipUnavailable):
        DbMembershipSource("latest", store=FailingStore())


def test_the_real_store_with_no_database_configured_is_unavailable() -> None:
    with pytest.raises(MembershipUnavailable, match="cannot read membership"):
        membership_source("db")


@pytest.mark.parametrize("how", ["environment", "argument"])
def test_an_unreachable_database_error_carries_no_credentials_or_host(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    url = "postgresql+psycopg2://scratch_user:s3cr3tpw@127.0.0.1:1/nodb"  # nothing listens on :1
    if how == "environment":
        monkeypatch.setenv("GGT_MEMBERSHIP_DB_URL", url)
    store = PgMembershipStore(url) if how == "argument" else None
    with pytest.raises(MembershipUnavailable) as err:
        membership_source("db", store=store)
    shown = "".join(traceback.format_exception(err.value)) + repr(err.value) + str(err.value)
    for secret in ("s3cr3tpw", "scratch_user", "127.0.0.1"):
        assert secret not in shown
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_redact_removes_url_user_password_and_host() -> None:
    url = "postgresql+psycopg2://alice:p%40ss@db.example.org:5433/ggtrader"
    text = f"fatal: {url} / alice / p@ss / p%40ss / db.example.org / postgresql://x:y@h/db"
    cleaned = redact(text, url)
    for secret in ("alice", "p@ss", "p%40ss", "db.example.org", "x:y@h"):
        assert secret not in cleaned
    assert redact("no url here", None) == "no url here"


def _tampered(store: FakeMembershipStore, **overrides: object) -> FakeMembershipStore:
    stored = store.vintages[1]
    store.vintages[1] = replace(stored, **overrides)  # type: ignore[arg-type]
    return store


def _one_vintage() -> FakeMembershipStore:
    store = FakeMembershipStore()
    store.add(synthetic_history(3))
    return store


def test_a_changed_interval_fails_the_content_hash_check() -> None:
    store = _one_vintage()
    stored = store.vintages[1]
    first = stored.intervals[0]
    moved = Interval(first.symbol_raw, first.valid_from + timedelta(days=1), first.valid_to)
    bad = _tampered(store, intervals=[moved, *stored.intervals[1:]])
    with pytest.raises(MembershipIntegrityError, match="content sha256"):
        DbMembershipSource(1, store=bad)


@pytest.mark.parametrize("key", ["content_sha256", "n_intervals", "n_snapshot_dates"])
def test_a_load_row_that_disagrees_with_the_rows_is_refused(key: str) -> None:
    store = _one_vintage()
    load = dict(store.vintages[1].load)
    load[key] = "f" * 64 if key == "content_sha256" else load[key] + 1
    bad = _tampered(store, load=load)
    with pytest.raises(MembershipIntegrityError):
        DbMembershipSource(1, store=bad)


def test_a_wrong_member_count_or_coverage_is_refused() -> None:
    store = _one_vintage()
    day = next(iter(store.vintages[1].dates))
    dates = {**store.vintages[1].dates, day: 1}
    with pytest.raises(MembershipIntegrityError, match="member count"):
        DbMembershipSource(1, store=_tampered(store, dates=dates))
    store = _one_vintage()
    load = {**store.vintages[1].load, "coverage_end": D("2099-01-01")}
    with pytest.raises(MembershipIntegrityError, match="coverage"):
        DbMembershipSource(1, store=_tampered(store, load=load))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1", "newest"])
def test_a_bad_vintage_argument_is_refused(bad: object) -> None:
    with pytest.raises(MembershipSpecError):
        DbMembershipSource(bad, store=_one_vintage())  # type: ignore[arg-type]


def test_the_db_vintage_record_carries_the_load_row_and_the_verification_time() -> None:
    store = FakeMembershipStore()
    checked = BASE_TIME + timedelta(days=30)
    store.add(synthetic_history(4), retrieved_at=BASE_TIME, verified_at=checked)
    vintage = membership_source("db", store=store).vintage()
    load = store.vintages[1].load
    assert (vintage.kind, vintage.load_id, vintage.fixture) == ("db", 1, None)
    assert vintage.content_sha256 == load["content_sha256"]
    assert vintage.raw_sha256 == load["raw_sha256"]
    assert (vintage.coverage_start, vintage.coverage_end) == (
        load["coverage_start"],
        load["coverage_end"],
    )
    assert (vintage.retrieved_at, vintage.verified_at) == (BASE_TIME, checked)


def test_instants_are_normalised_to_utc() -> None:
    pacific = timezone(timedelta(hours=-7))
    store = FakeMembershipStore()
    store.add(synthetic_history(4), retrieved_at=datetime(2026, 6, 8, 17, tzinfo=pacific))
    vintage = DbMembershipSource(1, store=store).vintage()
    assert vintage.retrieved_at == datetime(2026, 6, 9, tzinfo=timezone.utc)
    assert vintage.retrieved_at.utcoffset() == timedelta(0)


# ---- lookups ---------------------------------------------------------------------------------


@functools.cache
def _small() -> FixtureMembershipSource:
    """The vintage-1 fixture source, built once (the source is read-only)."""
    return FixtureMembershipSource(V1)


def test_history_table_has_the_shape_of_load_sp500_history() -> None:
    frame = _small().history()
    assert list(frame.columns) == ["tickers"]
    assert frame.index.name == "date"
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_monotonic_increasing
    assert all(t == sorted(t) for t in frame["tickers"].head(50))
    assert len(frame) == 2712
    assert _small().snapshot_dates()[0] == D("1996-01-02")


def test_history_is_built_once_per_source() -> None:
    source = _small()
    assert source.history() is source.history()


def test_asof_before_the_first_snapshot_raises_like_the_old_reader() -> None:
    with pytest.raises(ValueError, match="predates"):
        _small().members_asof(D("1996-01-01"))
    with pytest.raises(ValueError, match="predates"):
        _small().all_members_between(D("1990-01-01"), D("2000-01-01"))
    with pytest.raises(ValueError, match="NaT"):
        _small().members_asof(pd.NaT)


def test_snapshot_asof_names_the_snapshot_in_force() -> None:
    source = _small()
    day, tickers = source.snapshot_asof(D("2020-12-21"))  # TSLA joined that day
    assert day == D("2020-12-21") and "TSLA" in tickers
    day, tickers = source.snapshot_asof(D("2020-12-20"))
    assert day < D("2020-12-21") and "TSLA" not in tickers
    day, _ = source.snapshot_asof(D("2030-01-01"))  # carried forward past the end
    assert day == source.coverage_end == D("2026-06-02")


def test_timestamps_are_read_as_utc_days_like_the_old_reader() -> None:
    source = _small()
    naive = pd.Timestamp("2020-12-21 00:00:00")
    pacific = pd.Timestamp("2020-12-20 20:00:00", tz="US/Pacific")  # = 2020-12-21 04:00 UTC
    late_pacific = pd.Timestamp("2020-12-20 15:00:00", tz="US/Pacific")  # = 2020-12-20 23:00 UTC
    assert "TSLA" in source.members_asof(naive)
    assert "TSLA" in source.members_asof(pacific)
    assert "TSLA" not in source.members_asof(late_pacific)


def test_all_members_between_is_the_start_snapshot_plus_changes_in_the_window() -> None:
    source = _small()
    union = source.all_members_between(D("2020-01-01"), D("2021-12-31"))
    assert set(source.members_asof(D("2020-01-01"))) <= set(union)
    assert set(source.members_asof(D("2021-12-31"))) <= set(union)
    assert union == sorted(set(union))
    assert source.all_members_between(D("2021-12-31"), D("2020-01-01")) == source.members_asof(
        D("2021-12-31")
    )


# ---- provenance and age ----------------------------------------------------------------------


def test_provenance_has_the_fields_of_spec_3_5() -> None:
    record = _small().provenance(D("2026-11-08"))
    assert record["membership_sha256"] == V1_CONTENT_SHA
    block = record["membership"]
    assert block == {
        "kind": "fixture",
        "load_id": None,
        "fixture": V1,
        "raw_sha256": V1_RAW_SHA,
        "coverage_start": "1996-01-02",
        "coverage_end": "2026-06-02",
        "retrieved_at": "2026-06-09T00:00:00+00:00",
        "verified_at": "2026-06-09T00:00:00+00:00",
        "age_since_verified_days": 152,
    }
    json.dumps(record)  # JSON-able as it stands
    assert _small().provenance()["membership"]["age_since_verified_days"] is None


def test_db_provenance_names_the_load_id() -> None:
    store = _one_vintage()
    block = membership_source("db:1", store=store).provenance()["membership"]
    assert (block["kind"], block["load_id"], block["fixture"]) == ("db", 1, None)


def test_provenance_refuses_a_timestamp_as_the_as_of_day() -> None:
    with pytest.raises(TypeError, match="datetime.date"):
        _small().provenance(pd.Timestamp("2026-11-08"))  # type: ignore[arg-type]


def test_age_of_the_vintage_one_fixture_today_matches_the_spec() -> None:
    result = check_membership_age(_small().vintage(), D("2026-11-08"))
    assert (result.age_since_verified_days, result.age_since_change_days) == (152, 159)
    assert result.verdict == "ok" and result.reasons == ()


@pytest.mark.parametrize(
    ("asof", "verdict"),
    [
        ("2026-06-19", "ok"),  # 10 days: at the limit, not above it
        ("2026-06-20", "warn"),  # 11 days
        ("2026-07-24", "warn"),  # 45 days
        ("2026-07-25", "refuse"),  # 46 days
    ],
)
def test_verified_age_thresholds_are_strictly_greater_than(asof: str, verdict: str) -> None:
    result = check_membership_age(
        _small().vintage(), D(asof), verified_warn_days=10, verified_refuse_days=45
    )
    assert result.verdict == verdict
    assert bool(result.reasons) == (verdict != "ok")


def test_the_lag_bound_uses_the_last_change_date() -> None:
    vintage = _small().vintage()  # last change 2026-06-02
    assert check_membership_age(vintage, D("2026-08-31"), lag_warn_days=90).verdict == "ok"
    assert check_membership_age(vintage, D("2026-09-01"), lag_warn_days=90).verdict == "warn"
    refused = check_membership_age(vintage, D("2026-11-30"), lag_warn_days=90, lag_refuse_days=180)
    assert refused.verdict == "refuse" and "last change date" in refused.reasons[0]


def test_refuse_outranks_warn_and_both_rules_report() -> None:
    result = check_membership_age(
        _small().vintage(),
        D("2026-11-08"),
        verified_warn_days=10,
        verified_refuse_days=400,
        lag_warn_days=10,
        lag_refuse_days=100,
    )
    assert result.verdict == "refuse" and len(result.reasons) == 2


def test_a_vintage_verified_after_the_as_of_day_is_never_stale() -> None:
    result = check_membership_age(
        _small().vintage(), D("2024-01-02"), verified_warn_days=0, verified_refuse_days=0
    )
    assert result.age_since_verified_days < 0 and result.verdict == "ok"


def test_an_unchecked_threshold_never_fires() -> None:
    assert check_membership_age(_small().vintage(), D("2040-01-01")).verdict == "ok"


def test_warn_above_refuse_is_a_mistake() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        check_membership_age(
            _small().vintage(), D("2026-11-08"), verified_warn_days=50, verified_refuse_days=10
        )


def test_enforce_raises_on_refuse_and_warns_on_warn() -> None:
    vintage = _small().vintage()
    with pytest.raises(MembershipStaleError, match=V1):
        check_membership_age(vintage, D("2026-11-08"), verified_refuse_days=45, enforce=True)
    with pytest.warns(MembershipStaleWarning):
        check_membership_age(vintage, D("2026-11-08"), verified_warn_days=45, enforce=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_membership_age(vintage, D("2026-06-10"), verified_warn_days=45, enforce=True)


def test_the_age_check_refuses_a_datetime_as_of_day() -> None:
    with pytest.raises(TypeError, match="datetime.date"):
        check_membership_age(_small().vintage(), datetime(2026, 11, 8))  # type: ignore[arg-type]


def test_a_recorded_unchanged_check_moves_verified_at_forward() -> None:
    later = BASE_TIME + timedelta(days=140)
    store = FakeMembershipStore()
    store.add(synthetic_history(5), retrieved_at=BASE_TIME, verified_at=later)
    vintage = DbMembershipSource(store=store).vintage()
    result = check_membership_age(vintage, D("2026-11-08"), verified_refuse_days=45)
    assert result.age_since_verified_days == (D("2026-11-08") - later.date()).days == 12
    assert result.verdict == "ok"


# ---- CLI: verify --against-fixture -------------------------------------------------------------


def _verify_args(*extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["db", "membership", "verify", "--load-id", "1", *extra])


def test_verify_defaults_to_the_loads_own_bytes_and_accepts_a_file_or_fixture_name() -> None:
    assert _verify_args().against_fixture is None
    assert _verify_args().against_file is None  # no option = the vintage's own archived bytes
    assert _verify_args("--against-file", "x.csv").against_file == Path("x.csv")
    assert _verify_args("--against-fixture", V1).against_fixture == V1


def test_verify_takes_a_file_or_a_fixture_not_both() -> None:
    with pytest.raises(SystemExit):
        _verify_args("--against-file", "x.csv", "--against-fixture", V1)


def test_verify_with_an_unknown_fixture_fails_before_touching_the_database(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the database was opened")

    monkeypatch.setattr(cmd_membership, "connect", forbidden)
    assert cmd_membership.run_membership(_verify_args("--against-fixture", "nope")) == 1
    assert "nope" in capsys.readouterr().err


# ---- implicit selection and the process caches (slice M3) -------------------------------------


def test_selected_source_needs_a_spec_or_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    membership.clear_source_cache()
    with pytest.raises(MembershipSourceNotSelected, match="GGT_MEMBERSHIP"):
        membership.selected_source()
    with pytest.raises(MembershipSourceNotSelected):
        membership.resolve_source(None)
    monkeypatch.setenv("GGT_MEMBERSHIP", f"fixture:{V1}")
    assert membership.selected_source().vintage().fixture == V1


def test_selected_source_is_built_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    membership.clear_source_cache()
    monkeypatch.setenv("GGT_MEMBERSHIP", f"fixture:{V1}")
    first = membership.selected_source()
    assert membership.selected_source() is first
    assert membership.selected_source(f"fixture:{V1}") is first  # a spec string names the same one
    assert membership.resolve_source(first) is first  # an explicit source is never replaced
    membership.clear_source_cache()
    assert membership.selected_source() is not first


def test_selected_source_never_falls_back_to_a_fixture_when_db_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    membership.clear_source_cache()
    monkeypatch.setenv("GGT_MEMBERSHIP", "db")
    monkeypatch.setenv("GGT_MEMBERSHIP_DB_URL", "postgresql://u:pw@127.0.0.1:1/none")
    with pytest.raises(MembershipUnavailable) as err:
        membership.selected_source()
    assert "pw" not in str(err.value)


def test_equal_content_shares_one_history_frame_across_sources() -> None:
    membership.clear_source_cache()
    store = FakeMembershipStore()
    store.add(synthetic_history())
    database = DbMembershipSource("latest", store=store)
    other = DbMembershipSource("latest", store=store)
    assert database.history() is other.history()  # keyed by content_sha256
    assert database.history() is database.history()


def test_the_command_line_helper_selects_pins_and_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    membership.clear_source_cache()
    parser = argparse.ArgumentParser()
    membership.add_membership_argument(parser)
    with pytest.raises(MembershipSourceNotSelected):
        membership.source_from_args(parser.parse_args([]))
    args = parser.parse_args(
        ["--membership", f"fixture:{V1}", "--expect-membership-sha", V1_CONTENT_SHA]
    )
    assert membership.source_from_args(args).vintage().content_sha256 == V1_CONTENT_SHA
    wrong = parser.parse_args(
        ["--membership", f"fixture:{V1}", "--expect-membership-sha", "0" * 64]
    )
    with pytest.raises(MembershipIntegrityError, match="expected"):
        membership.source_from_args(wrong)
    oracle = argparse.ArgumentParser()
    membership.add_membership_argument(oracle, default=membership.ORACLE_SPEC)
    assert membership.source_from_args(oracle.parse_args([])).vintage().fixture == V1
    assert membership.ORACLE_SPEC == f"fixture:{V1}"
