"""Fast, offline tests of the weekly job's source fetchers and parsers.

The wikitext is built from the real pages' structure (``tests/data/membership_weekly_fakes.py``);
every HTTP call goes through a fake ``get``. Nothing opens a socket.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from ggTrader.data import membership_sources as ms
from ggTrader.data.membership_sources import (
    SourceStructureError,
    fetch_sources,
    parse_chino_changes,
    parse_chino_components,
    parse_current_list,
    parse_ledger,
    split_cells,
)
from tests.data.membership_weekly_fakes import (
    NOW,
    REAL_HEADER,
    World,
    base_ours,
    changes_csv,
    components_csv,
    current_json,
    current_wikitext,
    ledger_json,
    ledger_wikitext,
    parse_json,
    press_ref,
    reply,
)

ROWS = [
    # OKTA's row carries its announcement date (the real page's known defect)
    (
        "October 30, 2026",
        "OKTA",
        "Okta",
        "KVUE",
        "Kenvue",
        "Acquired.",
        press_ref("October 30, 2026"),
    ),
    (
        "October 6, 2026",
        "TWLO",
        "Twilio",
        "WBD",
        "Warner",
        "Acquired.",
        press_ref("October 1, 2026", "a"),
    ),
    ("October 6, 2026", "", "", "CTVA", "Corteva", "Cap changes.", '<ref name="a" />'),
    ("October 1, 2026", "VYLR", "Vylor", "", "", "Spin-off.", '<ref name="a" />'),
    ("September 21, 2026", "BE", "Bloom", "TAP", "Molson", "Cap changes.", ""),
]


# --------------------------------------------------------------------------- the ledger


def test_the_ledger_parses_rows_in_every_editing_style():
    ledger = parse_ledger(ledger_json(ROWS))
    assert ledger.revid == 111 and len(ledger.rows) == len(ROWS) + 110
    first = ledger.rows[0]
    assert (first.effective, first.added, first.removed) == (date(2026, 10, 30), "OKTA", "KVUE")
    assert first.added_name == "Okta" and first.removed_name == "Kenvue"
    by_key = {(r.effective, r.added, r.removed): r for r in ledger.rows[:5]}
    # a removal with no addition and an addition with no removal are both valid rows
    assert (date(2026, 10, 6), "", "CTVA") in by_key
    assert (date(2026, 10, 1), "VYLR", "") in by_key
    assert ledger.newest_effective == date(2026, 10, 30) and ledger.warnings == []


def test_a_stray_pipe_after_a_ticker_and_two_cells_on_one_line_are_read_correctly():
    ledger = parse_ledger(ledger_json(ROWS))
    # row index 2 of the page uses the third style: "ADD |" and "REM|| [[Name]]" on one line
    third = ledger.rows[2]
    assert third.removed == "CTVA" and third.removed_name == "Corteva" and third.added == ""
    filler = [r for r in ledger.rows if r.added.startswith("ZA")][2]
    assert filler.added == filler.added.strip() and "|" not in filler.added + filler.removed


def test_announcement_dates_come_from_refs_including_reused_names():
    ledger = parse_ledger(ledger_json(ROWS))
    by_key = {(r.effective, r.added, r.removed): r for r in ledger.rows[:5]}
    okta = by_key[(date(2026, 10, 30), "OKTA", "KVUE")]
    assert okta.announcement == okta.effective  # dated on its announcement date
    twlo = by_key[(date(2026, 10, 6), "TWLO", "WBD")]
    assert twlo.announcement == date(2026, 10, 1)
    # the reused ref name resolves to the definition's date
    assert by_key[(date(2026, 10, 6), "", "CTVA")].announcement == date(2026, 10, 1)
    assert by_key[(date(2026, 10, 1), "VYLR", "")].announcement == date(2026, 10, 1)
    assert by_key[(date(2026, 9, 21), "BE", "TAP")].announcement is None  # no refs at all


def test_a_row_without_a_refs_cell_has_six_cells_and_still_parses():
    text = ledger_wikitext(ROWS)
    assert "\n|-\n" in text
    ledger = parse_ledger(parse_json(text))
    assert any(r.announcement is None for r in ledger.rows)


@pytest.mark.parametrize(
    ("old", "new", "word"),
    [
        ("| Effective Date", "| Date", "columns"),  # renamed
        ('! rowspan="2" | Reason\n', "", "columns"),  # missing
        ('! rowspan="2" | Refs', '! rowspan="2" | Refs\n! rowspan="2" | Notes', "columns"),  # extra
        (
            "! Ticker || Security || Ticker || Security",
            "! Symbol || Name || Symbol || Name",
            "columns",
        ),
    ],
)
def test_a_changed_column_is_detected(old, new, word):
    header = REAL_HEADER.replace(old, new)
    assert header != REAL_HEADER
    with pytest.raises(SourceStructureError) as caught:
        parse_ledger(parse_json(ledger_wikitext(ROWS, header=header)))
    assert caught.value.code == "columns_changed" and word in str(caught.value)


def test_a_merged_header_row_and_a_missing_table_are_detected():
    one_row = REAL_HEADER.replace("|-\n! Ticker || Security || Ticker || Security", "")
    with pytest.raises(SourceStructureError, match="header rows"):
        parse_ledger(parse_json(ledger_wikitext(ROWS, header=one_row)))
    renamed = REAL_HEADER.replace('id="changes"', 'id="renamed"')
    with pytest.raises(SourceStructureError) as caught:
        parse_ledger(parse_json(ledger_wikitext(ROWS, header=renamed)))
    assert caught.value.code == "table_not_found"


def test_an_api_error_reply_and_a_tiny_table_are_not_a_ledger():
    with pytest.raises(SourceStructureError) as caught:
        parse_ledger(json.dumps({"error": {"code": "maxlag"}}).encode())
    assert caught.value.code == "api_error" and "maxlag" in str(caught.value)
    tiny = '{| class="wikitable" id="changes"\n' + REAL_HEADER.split("\n", 1)[1] + "\n|}\n"
    with pytest.raises(SourceStructureError) as caught:
        parse_ledger(parse_json(tiny))
    assert caught.value.code == "too_few_rows"


def test_a_bad_newest_row_is_an_error_but_a_bad_old_row_is_only_a_warning():
    bad_new = [("Sometime soon", "XX", "n", "YY", "m", "r", ""), *ROWS]
    with pytest.raises(SourceStructureError) as caught:
        parse_ledger(ledger_json(bad_new))
    assert caught.value.code == "bad_row" and "does not parse" in str(caught.value)
    text = ledger_wikitext(ROWS) + ""
    old_bad = text.replace("ZA100", "ZA 100!")  # an old row near the end of the page
    ledger = parse_ledger(parse_json(old_bad))
    assert len(ledger.warnings) == 1 and "bad ticker" in ledger.warnings[0]


def test_the_cell_splitter_ignores_pipes_inside_templates_links_and_refs():
    cells = split_cells("|A\n|{{T|x||y}}\n|[[a|b]] || c\n|<ref>p||q</ref>\n")
    assert [c.strip() for c in cells] == ["A", "{{T|x||y}}", "[[a|b]]", "c", "<ref>p||q</ref>"]


# --------------------------------------------------------------------------- the current list


def test_the_current_list_reads_symbols_from_all_template_spellings():
    tickers = [t.replace(".", "-") for t in base_ours()]
    current = parse_current_list(current_json(tickers))
    assert current.revid == 222 and set(current.tickers) == set(tickers)
    assert "BRK-B" in current.tickers and "BRK.B" not in current.tickers


def test_a_changed_current_list_is_detected():
    tickers = [t.replace(".", "-") for t in base_ours()]
    renamed = ms_header = current_wikitext(tickers).replace("[[Ticker symbol|Symbol]]", "Ticker")
    with pytest.raises(SourceStructureError) as caught:
        parse_current_list(parse_json(renamed))
    assert caught.value.code == "columns_changed" and ms_header
    short = current_wikitext(tickers[:100])
    with pytest.raises(SourceStructureError) as caught:
        parse_current_list(parse_json(short))
    assert caught.value.code == "too_few_rows"
    dropped = current_wikitext(tickers).replace("|| Industrials \n", "", 1)
    with pytest.raises(SourceStructureError, match="cells"):
        parse_current_list(parse_json(dropped))


# --------------------------------------------------------------------------- chinobing


def test_the_components_file_gives_its_newest_snapshot_and_tolerates_repeated_dates():
    snap = parse_chino_components(components_csv(base_ours(), date(2026, 11, 8)))
    assert snap.day == date(2026, 11, 8) and len(snap.tickers) == 499
    assert "BRK-B" in snap.tickers and snap.n_rows == 121  # one repeated date


def test_the_components_file_refuses_a_wrong_header_dates_going_backwards_and_few_rows():
    good = components_csv(base_ours(), date(2026, 11, 8)).decode()
    with pytest.raises(SourceStructureError, match="header"):
        parse_chino_components(good.replace("date,tickers", "day,members", 1).encode())
    lines = good.splitlines()
    lines[50], lines[51] = lines[51], lines[50]
    with pytest.raises(SourceStructureError, match="backwards"):
        parse_chino_components("\n".join(lines).encode() + b"\n")
    with pytest.raises(SourceStructureError, match="snapshot rows"):
        parse_chino_components("\n".join(good.splitlines()[:20]).encode() + b"\n")


def test_the_changes_file_parses_python_style_lists():
    changes = parse_chino_changes(
        changes_csv(
            [(date(2026, 10, 8), ["TWLO", "VYLR"], ["WBD"]), (date(2026, 11, 5), ["OKTA"], [])]
        )
    )
    assert changes[-1].added == ("OKTA",) and changes[-1].removed == ()
    assert changes[-2].day == date(2026, 10, 8) and changes[-2].added == ("TWLO", "VYLR")
    with pytest.raises(SourceStructureError, match="header"):
        parse_chino_changes(b"date,added\n")
    broken = changes_csv([]).decode() + '2026-01-01,"not a list","[]"\n'
    with pytest.raises(SourceStructureError, match="cannot be parsed"):
        parse_chino_changes(broken.encode())


# --------------------------------------------------------------------------- fetching


def test_fetch_sources_reads_four_files_and_two_metadata_calls_politely():
    world = World(base_ours())
    fetched = fetch_sources(get=world, now=lambda: NOW)
    assert set(fetched) == set(ms.FETCHED_SOURCES) and all(s.ok for s in fetched.values())
    assert len(world.calls) == 6  # 2 wiki pages, their edit times, 2 CSVs, the repo's last commit
    assert len({(u, tuple(sorted(p.items()))) for u, p, _ in world.calls}) == 6
    for _, params, headers in world.calls:
        assert headers["User-Agent"] == ms.USER_AGENT and "ggTrader" in ms.USER_AGENT
        if params.get("action"):
            assert params["maxlag"] == str(ms.MAX_LAG_SECONDS) and params["format"] == "json"
    parse_calls = [p for _, p, _ in world.calls if p.get("action") == "parse"]
    assert {p["page"] for p in parse_calls} == {ms.LEDGER_PAGE, ms.CURRENT_PAGE}
    assert all(p["prop"] == "wikitext|revid" for p in parse_calls)
    assert fetched[ms.LEDGER].last_change == date(2026, 11, 3)
    assert fetched[ms.CURRENT].last_change == date(2026, 11, 8)
    assert fetched[ms.CHANGES].last_change == date(2026, 11, 8)


@pytest.mark.parametrize(
    ("key", "failure", "reason"),
    [
        ("ledger", requests.ConnectionError("no route"), "unreachable"),
        ("current", reply(b"gone", 404), "http_status"),
        ("components", reply(b"", 200), "empty"),
        (
            "changes",
            reply(b"x" * 10).__class__(200, b"x" * 10, {"content-length": "99"}),
            "truncated",
        ),
    ],
)
def test_one_failing_source_is_recorded_and_the_others_still_arrive(key, failure, reason):
    world = World(base_ours())
    world.errors[key] = failure
    fetched = fetch_sources(get=world, now=lambda: NOW)
    names = {
        "ledger": ms.LEDGER,
        "current": ms.CURRENT,
        "components": ms.COMPONENTS,
        "changes": ms.CHANGES,
    }
    broken = fetched[names[key]]
    assert not broken.ok and broken.reason == reason and broken.raw is None
    assert all(s.ok for n, s in fetched.items() if n != names[key])


def test_a_wiki_api_error_object_and_non_json_are_failures():
    world = World(base_ours())
    world.errors["ledger"] = reply(json.dumps({"error": {"code": "ratelimited"}}).encode())
    world.errors["current"] = reply(b"<html>blocked</html>")
    fetched = fetch_sources(get=world, now=lambda: NOW)
    assert fetched[ms.LEDGER].reason == "api_error" and "ratelimited" in fetched[ms.LEDGER].error
    assert fetched[ms.CURRENT].reason == "not_json"


def test_failed_metadata_calls_leave_the_last_change_unknown_and_say_why():
    world = World(base_ours())
    world.errors["revisions"] = reply(b"slow down", 429)
    world.errors["commits"] = requests.Timeout("timed out")
    fetched = fetch_sources(get=world, now=lambda: NOW)
    assert fetched[ms.LEDGER].ok and fetched[ms.LEDGER].last_change is None
    assert "429" in fetched[ms.LEDGER].detail["last_change_error"]
    assert fetched[ms.COMPONENTS].last_change is None
    assert "unreachable" in fetched[ms.COMPONENTS].detail["last_change_error"]


def test_an_unexpected_exception_in_one_fetch_cannot_abort_the_others():
    world = World(base_ours())
    world.errors["ledger"] = ValueError("boom")  # not a requests error
    fetched = fetch_sources(get=world, now=lambda: NOW)
    assert fetched[ms.LEDGER].reason == "internal" and fetched[ms.CURRENT].ok


def test_an_oversized_source_is_refused_while_streaming():
    big = reply(b"x" * (ms.MAX_DOWNLOAD_BYTES + 5 * 65536))
    world = World(base_ours())
    world.errors["current"] = big
    fetched = fetch_sources(get=world, now=lambda: NOW)
    failed = [f for f in fetched.values() if not f.ok]
    assert [f.reason for f in failed] == ["too_large"]
    assert big.served <= ms.MAX_DOWNLOAD_BYTES + 65536 and big.closed
