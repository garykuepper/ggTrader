"""Offline tests of the S&P DJI press-release source (``membership_press``) and its use by the
weekly job: parsing, fetching, the official date in a proposal and the warnings.

Page shapes follow the live PR Newswire pages (checked 2026-11-09); nothing here opens a socket.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from ggTrader.data import membership_press as mp
from ggTrader.data.membership_proposal import build_proposal
from ggTrader.data.membership_sources import SourceStructureError
from tests.data.membership_weekly_fakes import (
    MIDCAP_CARD,
    NOW,
    FakeWeeklyStore,
    RecordingNotifier,
    World,
    base_ours,
    press_listing_html,
    press_release_html,
    reply,
)
from tests.data.test_membership_weekly import codes, fabricated_mismatch, read_proposal, run

OKTA_SLUG = "okta-set-to-join-sp-500-others-to-join-sp-midcap-400-302925405"
OKTA_CARD = ("Oct 30, 2026, 17:51 ET", OKTA_SLUG, "Okta Set to Join S&amp;P 500; Others")
OLD_CARD = ("Jul 31, 2026, 17:51 ET", "ferguson-set-to-join-sp-500-302840407", "Ferguson")


def release(release_id: str = "302925405", published: date = date(2026, 10, 30)):
    return mp.PressRelease(
        mp.PRESS_BASE + f"/news-releases/x-{release_id}.html", "t", published, release_id
    )


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Nov 4, 2026", date(2026, 11, 4)),
        ("Sept 21, 2026", date(2026, 9, 21)),
        ("Nov 06, 2026, 18:09 ET", date(2026, 11, 6)),
        ("Dec. 1, 2026", date(2026, 12, 1)),
        ("Feb 30, 2026", None),
        ("Xyz 4, 2026", None),
        ("soon", None),
    ],
)
def test_press_dates(text, expected):
    assert mp.parse_press_date(text) == expected


def test_the_listing_keeps_only_sp500_releases_newest_first_and_dedupes():
    raw = press_listing_html([MIDCAP_CARD, OLD_CARD, OKTA_CARD, OKTA_CARD])
    found = mp.parse_press_listing(raw)
    assert [(r.release_id, r.published) for r in found] == [
        ("302925405", date(2026, 10, 30)),
        ("302840407", date(2026, 7, 31)),
    ]
    assert found[0].url == mp.PRESS_BASE + f"/news-releases/{OKTA_SLUG}.html"
    assert found[0].source_name == "sp_press_302925405" and "&" in found[0].title


def test_a_listing_with_no_cards_or_an_unreadable_card_is_a_structure_error():
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_listing(b"<html>maintenance</html>")
    assert err.value.code == "cards_not_found"
    bad = press_listing_html([("sometime", OKTA_SLUG, "Okta")])
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_listing(bad)
    assert err.value.code == "bad_card"


def test_the_release_table_gives_only_the_sp500_additions_and_deletions():
    raw = press_release_html(
        [
            ("Nov 4, 2026", "S&amp;P 500", "Addition", "Okta", "OKTA"),
            ("Nov 4, 2026", "S&P 500", "Deletion", "Kenvue", "KVUE"),
            ("Nov 4, 2026", "S&amp;P MidCap 400", "Addition", "Glaukos", "GKOS"),
            ("Sept 21, 2026", "S&P 100", "Deletion", "NIKE", "NKE"),
        ]
    )
    got = mp.parse_press_release(raw, release())
    assert [(c.effective, c.action, c.ticker, c.company) for c in got] == [
        (date(2026, 11, 4), "add", "OKTA", "Okta"),
        (date(2026, 11, 4), "remove", "KVUE", "Kenvue"),
    ]
    assert all(c.published == date(2026, 10, 30) and c.url.endswith("-302925405.html") for c in got)


def test_a_release_that_only_touches_other_indices_gives_no_rows():
    raw = press_release_html([("Nov 4, 2026", "S&P MidCap 400", "Addition", "Glaukos", "GKOS")])
    assert mp.parse_press_release(raw, release()) == []


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"<html><body>no body here</body></html>", "body_not_found"),
        (
            press_release_html([("Nov 4, 2026", "S&P 500", "Addition", "A", "A")], header=False),
            "table_not_found",
        ),
        (press_release_html([("Nov 4, 2026", "S&P 500", "Replaced", "A", "A")]), "bad_row"),
        (press_release_html([("someday", "S&P 500", "Addition", "A", "A")]), "bad_row"),
        (press_release_html([("Nov 4, 2026", "S&P 500", "Addition", "A", "")]), "bad_row"),
    ],
)
def test_a_changed_release_page_is_a_structure_error(raw, code):
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(raw, release())
    assert err.value.code == code


def test_a_correction_release_replaces_the_earlier_date():
    first = mp.PressChange(date(2026, 11, 4), "add", "OKTA", "Okta", date(2026, 10, 30), "u1")
    fixed = mp.PressChange(date(2026, 11, 5), "add", "okta", "Okta", date(2026, 11, 2), "u2")
    other = mp.PressChange(date(2026, 11, 4), "remove", "KVUE", "Kenvue", date(2026, 10, 30), "u1")
    got = mp.latest_changes([fixed, first, other])
    assert [(c.action, c.ticker, c.effective) for c in got] == [
        ("remove", "KVUE", date(2026, 11, 4)),
        ("add", "okta", date(2026, 11, 5)),
    ]


# --------------------------------------------------------------------------- fetching


def test_fetch_press_fetches_only_recent_sp500_releases_within_the_cap():
    world = World(base_ours())
    world.press_listing = press_listing_html([MIDCAP_CARD, OLD_CARD, OKTA_CARD])
    page = press_release_html([("Nov 4, 2026", "S&P 500", "Addition", "Okta", "OKTA")])
    world.press_releases = {"302925405": page, "302840407": page}
    out = mp.fetch_press(get=world, now=lambda: NOW)
    assert out.listing.ok and [r.release_id for r, _ in out.releases] == ["302925405"]
    assert out.releases[0][1].ok and out.releases[0][1].last_change == date(2026, 10, 30)
    assert len(world.calls) == 2  # the old (July) release is outside the 75-day look-back
    capped = mp.fetch_press(get=world, now=lambda: NOW, lookback_days=400, max_releases=1)
    assert len(capped.releases) == 1


def test_fetch_press_never_raises_when_a_request_fails():
    world = World(base_ours())
    world.press_listing = press_listing_html([OKTA_CARD])
    world.press_releases = {}
    out = mp.fetch_press(get=world, now=lambda: NOW)
    assert out.releases[0][1].ok is False and out.releases[0][1].reason == "http_status"

    def boom(*_a, **_k):
        raise RuntimeError("socket on fire")

    dead = mp.fetch_press(get=boom, now=lambda: NOW)
    assert dead.listing.ok is False and dead.releases == []
    garbled = World(base_ours())
    garbled.press_listing = b"<html>no cards</html>"
    assert mp.fetch_press(get=garbled, now=lambda: NOW).releases == []


# --------------------------------------------------------------------------- the proposal dates


def _proposal(press, *, ledger=(), chino=()):
    return build_proposal(
        adds=["ZZNEW"],
        removes=[],
        our_raw={},
        ledger=ledger,
        chino=chino,
        floor=date(2026, 11, 1),
        today=date(2026, 11, 9),
        our_count=500,
        press=press,
    )


def _change(day=date(2026, 11, 5), action="add", published=date(2026, 11, 2)):
    return mp.PressChange(day, action, "ZZNEW", "New", published, "https://pr/x-1.html")


def test_the_official_date_alone_dates_a_change_but_still_needs_a_human():
    row = _proposal([_change()]).rows[0]
    assert (row.effective, row.date_source, row.needs_confirmation) == (
        date(2026, 11, 5),
        "press",
        True,
    )
    assert row.press_date == date(2026, 11, 5) and "https://pr/x-1.html" in row.note


def test_an_agreeing_second_source_clears_the_flag_and_is_named():
    from ggTrader.data.membership_sources import ChinoChange

    chino = [ChinoChange(date(2026, 11, 5), ("ZZNEW",), ())]
    row = _proposal([_change()], chino=chino).rows[0]
    assert (row.date_source, row.needs_confirmation) == ("chinobing+press", False)


def test_a_disagreeing_source_is_noted_and_the_flag_stays():
    from ggTrader.data.membership_sources import ChinoChange

    chino = [ChinoChange(date(2026, 11, 7), ("ZZNEW",), ())]
    row = _proposal([_change()], chino=chino).rows[0]
    assert (row.effective, row.date_source, row.needs_confirmation) == (
        date(2026, 11, 5),
        "press",
        True,
    )
    assert "chinobing date 2026-11-07 differs" in row.note


def test_a_future_official_date_always_needs_a_human():
    from ggTrader.data.membership_sources import ChinoChange

    chino = [ChinoChange(date(2026, 11, 12), ("ZZNEW",), ())]
    row = _proposal([_change(day=date(2026, 11, 12))], chino=chino).rows[0]
    assert row.needs_confirmation is True and "future" in row.note


def test_without_press_rows_the_proposal_is_unchanged():
    assert _proposal([]).rows[0].date_source == "none"
    assert _proposal([_change(day=date(2026, 9, 1))]).rows[0].date_source == "none"  # too old


# --------------------------------------------------------------------------- the weekly job


def _announce(world, rows, card=OKTA_CARD):
    world.press_listing = press_listing_html([card])
    world.press_releases = {"302925405": press_release_html(rows)}


def test_an_announced_change_that_our_list_has_is_quiet_and_archived(tmp_path):
    ours = base_ours()
    world = World(ours)
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001"),
            ("Oct 20, 2026", "S&P 500", "Deletion", "Gone", "GONE"),
        ],
    )
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "ok" and result.findings == []
    _, rows = store.runs[0]
    by_name = {r.source: r for r in rows}
    assert by_name["sp_press_listing"].last_change == date(2026, 10, 30)
    page = by_name["sp_press_302925405"]
    assert page.raw == world.press_releases["302925405"] and page.last_change == date(2026, 10, 30)
    assert page.detail["n_sp500_rows"] == 2
    assert len(notifier.messages) == 1 and "sp_press_302925405" not in notifier.messages[0]
    assert "sp_press_listing ok" in notifier.messages[0]


def test_an_announced_change_missing_from_our_list_is_a_warning_not_a_refusal(tmp_path):
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Addition", "Ghost", "GHOST"),
            ("Oct 20, 2026", "S&P 500", "Deletion", "Kept", "X0002"),
            ("Dec 1, 2026", "S&P 500", "Addition", "Later", "LATER"),  # not yet effective
        ],
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and result.exit_code == 0
    assert codes(result) == {"press_change_not_in_list"}
    text = " ".join(f.text for f in result.findings)
    assert "GHOST added" in text and "X0002 removed" in text and "LATER" not in text
    assert result.check_id == 1 and result.spy.outcome == "matched"


def test_a_later_announced_change_of_the_same_ticker_supersedes_the_earlier_one(tmp_path):
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 10, 2026", "S&P 500", "Addition", "Flip", "FLIP"),
            ("Oct 20, 2026", "S&P 500", "Deletion", "Flip", "FLIP"),
        ],
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert result.status == "ok"  # FLIP was added, then removed: absent from our list is right


@pytest.mark.parametrize(
    "failure", ["listing_down", "page_down", "page_changed", "listing_changed"]
)
def test_a_press_failure_only_warns_and_the_match_is_still_recorded(failure, tmp_path):
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    if failure == "listing_down":
        world.errors["press_listing"] = reply(b"down", 503)
    elif failure == "page_down":
        world.press_releases = {}
    elif failure == "page_changed":
        world.press_releases["302925405"] = b"<html><body class='x'>redesign</body></html>"
    else:
        world.press_listing = b"<html>redesign</html>"
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and result.exit_code == 0
    assert codes(result) <= {"press_unavailable", "press_structure"} and codes(result)
    assert result.spy.outcome == "matched" and result.check_id == 1
    assert len(notifier.messages) == 1 and "note" in notifier.messages[0]
    assert store.runs[0][0].outcome == "matched"


def test_the_official_dates_clear_the_flags_in_a_proposal(tmp_path):
    rows = [
        ("Nov 2, 2026", "S&P 500", "Addition", "Agree", "ZZAGR"),
        ("Nov 2, 2026", "S&P 500", "Deletion", "Ten", "X0010"),
        ("Nov 5, 2026", "S&P 500", "Addition", "New", "ZZNEW"),
        ("Nov 5, 2026", "S&P 500", "Deletion", "Nine", "X0009"),
        ("Nov 6, 2026", "S&P 500", "Addition", "Ann", "ZZANN"),
        ("Nov 6, 2026", "S&P 500", "Deletion", "Thirty", "X0030"),
    ]
    world, _, result, store, notifier = fabricated_mismatch(
        tmp_path, tweak=lambda w: _announce(w, rows)
    )
    assert result.status == "refuse"
    got = {
        (r["action"], r["ticker"]): (r["effective_date"], r["date_source"], r["needs_confirmation"])
        for r in read_proposal(result.proposal_path)
    }
    assert got[("add", "ZZAGR")] == ("2026-11-02", "ledger+chinobing+press", "N")
    # the ledger says 11-04 and chinobing observed 11-05: the release settles it at 11-05
    assert got[("add", "ZZNEW")] == ("2026-11-05", "chinobing+press", "N")
    assert got[("remove", "X0009")] == ("2026-11-05", "chinobing+press", "N")
    # the ledger row carries its announcement date; the release agrees with chinobing's 11-06
    assert got[("add", "ZZANN")] == ("2026-11-06", "chinobing+press", "N")
    # no release dates the rename pair: still chinobing alone
    assert got[("add", "NEWCO")] == ("2026-11-03", "chinobing", "Y")
    assert "ledger+chinobing+press" in notifier.messages[0]


def test_the_dry_run_store_and_notifier_stay_offline():
    # guard: the helpers used above never reach a database or Telegram
    assert FakeWeeklyStore(base_ours()).runs == [] and RecordingNotifier().messages == []
    assert NOW == datetime(2026, 11, 9, 14, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- review fixes


def test_a_hostile_numeric_reference_reads_as_a_changed_page_not_an_exception():
    huge = "&#" + "9" * 5000 + ";"
    # in the date text: it reaches html.unescape (a title is cut at 1,000 characters before that)
    bad_stamp = press_listing_html([(f"Oct 30, 2026 {huge}", OKTA_SLUG, "Okta")])
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_listing(bad_stamp)
    assert err.value.code == "internal"
    page = press_release_html([("Nov 4, 2026", "S&P 500", "Addition", f"Okta {huge}", "OKTA")])
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(page, release())
    assert err.value.code == "internal"


def test_a_hostile_page_only_warns_in_the_weekly_job_and_the_match_is_recorded(tmp_path):
    world = World(base_ours())
    world.press_listing = press_listing_html(
        [("Oct 30, 2026 &#" + "9" * 5000 + ";", OKTA_SLUG, "Okta")]
    )
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and codes(result) == {"press_structure"}
    assert store.runs[0][0].outcome == "matched"


def test_the_correction_with_the_higher_release_id_wins_even_on_the_same_day():
    first = mp.PressChange(
        date(2026, 11, 5), "add", "OKTA", "Okta", date(2026, 10, 30), "u1", "302925405"
    )
    fixed = mp.PressChange(
        date(2026, 11, 4), "add", "OKTA", "Okta", date(2026, 10, 30), "u2", "302925999"
    )
    assert [c.effective for c in mp.latest_changes([fixed, first])] == [date(2026, 11, 4)]


def test_the_proposal_uses_the_newest_release_on_the_same_day():
    first = mp.PressChange(
        date(2026, 11, 5), "add", "ZZNEW", "New", date(2026, 11, 2), "u1", "302925405"
    )
    fixed = mp.PressChange(
        date(2026, 11, 4), "add", "ZZNEW", "New", date(2026, 11, 2), "u2", "302925999"
    )
    assert _proposal([fixed, first]).rows[0].effective == date(2026, 11, 4)


def test_dotted_and_dashed_spellings_are_one_ticker_in_corrections():
    first = mp.PressChange(
        date(2026, 10, 20), "add", "BRK.B", "Berkshire", date(2026, 10, 1), "u1", "302900001"
    )
    fixed = mp.PressChange(
        date(2026, 11, 12), "add", "BRK-B", "Berkshire", date(2026, 11, 2), "u2", "302900002"
    )
    assert [c.effective for c in mp.latest_changes([first, fixed])] == [date(2026, 11, 12)]


def test_a_future_opposite_change_does_not_hide_a_current_discrepancy(tmp_path):
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Addition", "Ghost", "GHOST"),
            ("Dec 1, 2026", "S&P 500", "Deletion", "Ghost", "GHOST"),  # not yet effective
        ],
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert codes(result) == {"press_change_not_in_list"}
    assert "GHOST added" in " ".join(f.text for f in result.findings)


def test_a_truncated_sp500_row_is_a_structure_error():
    cells = "<tr><td>Nov 4, 2026</td><td>S&amp;P 500</td><td>Addition</td><td>Okta</td></tr>"
    page = press_release_html([]).replace(b"</table>", cells.encode() + b"</table>")
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(page, release())
    assert err.value.code == "bad_row"


# --------------------------------------------------------------------------- review round 2


def test_a_card_with_a_logo_before_its_heading_is_still_read():
    card = (
        '<div class="card"><a class="newsreleaseconsolidatelink display-outline w-100" '
        f'href="/news-releases/{OKTA_SLUG}.html"><div class="img-parent-fixed"><img src="x"></div>'
        '<h3 class="no-top-margin"><small>Oct 30, 2026, 17:51 ET</small> Okta Set to Join '
        "S&amp;P 500</h3></a></div>"
    )
    found = mp.parse_press_listing(f"<html>{card}</html>".encode())
    assert [r.release_id for r in found] == ["302925405"] and found[0].title.endswith("S&P 500")


def test_a_card_the_parser_cannot_read_is_a_structure_error_not_a_silent_drop():
    good = press_listing_html([OKTA_CARD]).decode()
    odd = '<a class="newsreleaseconsolidatelink" href="https://elsewhere/x">odd</a>'
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_listing((good + odd).encode())
    assert err.value.code == "bad_card"


def test_parsing_time_stays_linear_on_a_page_of_unclosed_cards():
    import time

    card = (
        '<a class="newsreleaseconsolidatelink x" href="/news-releases/a-sp-500-302925405.html">'
        "<h3><small>Oct 30, 2026</small> title "
    )
    started = time.perf_counter()
    found = mp.parse_press_listing((card * 6000).encode())  # about 700 KB, no closing tags
    assert [r.release_id for r in found] == ["302925405"]  # the same release, deduplicated
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("text", ["Mayo 3, 2026", "May 3, 20269", "Nov 4, 99999"])
def test_loose_month_words_and_long_years_are_not_dates(text):
    assert mp.parse_press_date(text) is None


def test_a_correction_that_moves_the_date_earlier_is_not_lost_to_the_slack_cut():
    # floor 2026-11-20 -> cut 11-06; the correction (11-05) must not leave the stale 11-10 alive
    first = mp.PressChange(
        date(2026, 11, 10), "add", "ZZNEW", "New", date(2026, 11, 2), "u1", "302900100"
    )
    fixed = mp.PressChange(
        date(2026, 11, 5), "add", "ZZNEW", "New", date(2026, 11, 3), "u2", "302900200"
    )
    row = build_proposal(
        adds=["ZZNEW"],
        removes=[],
        our_raw={},
        ledger=[],
        chino=[],
        floor=date(2026, 11, 20),
        today=date(2026, 11, 21),
        our_count=500,
        press=[first, fixed],
    ).rows[0]
    assert row.press_date is None and row.date_source == "none"  # neither date is used


def test_a_same_day_delete_and_re_add_of_one_ticker_raises_no_warning(tmp_path):
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Deletion", "Same", "X0003"),
            ("Oct 20, 2026", "S&P 500", "Addition", "Same", "X0003"),
        ],
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert result.status == "ok" and result.findings == []


# --------------------------------------------------------------------------- review round 3


def _timed(fn):
    import time

    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def test_an_oversized_release_id_is_a_structure_error_not_an_exception(tmp_path):
    slug = "okta-set-to-join-sp-500-" + "9" * 5000
    page = press_listing_html([("Oct 30, 2026, 17:51 ET", slug, "Okta")])
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_listing(page)
    assert err.value.code == "cards_not_found"  # an id this long is not a card at all
    world = World(base_ours())
    world.press_listing = page
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and codes(result) == {"press_structure"}
    assert store.runs[0][0].outcome == "matched"


def test_opening_tags_without_a_closing_bracket_cost_linear_time():
    tag = '<a class="newsreleaseconsolidatelink" href="/news-releases/x-sp-500-302925405.html" '
    page = ("<html>" + tag * 8000).encode()  # no '>' anywhere

    def parse():
        with pytest.raises(SourceStructureError):
            mp.parse_press_listing(page)

    assert _timed(parse) < 1.0


def test_thousands_of_angle_brackets_in_a_title_cost_linear_time():
    title = "<" * 200_000
    page = press_listing_html([("Oct 30, 2026, 17:51 ET", OKTA_SLUG, title)])
    assert _timed(lambda: mp.parse_press_listing(page)) < 1.0


def test_a_release_with_an_absurd_number_of_rows_is_refused():
    rows = [("Nov 4, 2026", "S&P 500", "Addition", f"C{i}", f"T{i}") for i in range(300)]
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(press_release_html(rows), release())
    assert err.value.code == "too_many_rows"


def test_comparing_the_most_rows_a_release_may_hold_is_fast(tmp_path):
    rows = [("Oct 20, 2026", "S&P 500", "Addition", f"C{i}", f"T{i:04d}") for i in range(200)]
    world = World(base_ours())
    _announce(world, rows)

    def go():
        result, _, _ = run(world, tmp_path=tmp_path)
        assert codes(result) == {"press_change_not_in_list"}

    assert _timed(go) < 5.0


# --------------------------------------------------------------------------- size cap


def test_a_press_page_over_the_size_cap_is_refused_and_never_stored(tmp_path):
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.press_releases["302925405"] = b"<html>" + b"x" * (mp.MAX_PAGE_BYTES + 1)
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and codes(result) == {"press_unavailable"}
    rows = {r.source: r for r in store.runs[0][1]}
    assert rows["sp_press_302925405"].status == "failed" and rows["sp_press_302925405"].raw is None
    assert rows["sp_press_302925405"].detail["reason"] == "too_large"
    assert store.runs[0][0].outcome == "matched"


def test_the_size_cap_is_exact_at_the_boundary():
    for extra, ok in ((0, True), (1, False)):
        world = World(base_ours())
        base = press_listing_html([OKTA_CARD])
        world.press_listing = base + b" " * (mp.MAX_PAGE_BYTES - len(base) + extra)
        out = mp.fetch_press(get=world, now=lambda: NOW)
        assert out.listing.ok is ok
        if ok:
            assert len(out.listing.raw) == mp.MAX_PAGE_BYTES
        else:
            assert out.listing.reason == "too_large" and out.listing.raw is None


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def test_control_characters_in_a_page_never_reach_the_database_rows(tmp_path):
    nul = "\x00"
    world = World(base_ours())
    world.press_listing = press_listing_html([("Oct 30, 2026, 17:51 ET", OKTA_SLUG, f"Ok{nul}ta")])
    world.press_releases = {
        "302925405": press_release_html(
            [("Oct 20, 2026", "S&P 500", "Addition", f"Co{nul}mpany", "X0001")]
        )
    }
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "ok"
    for row in store.runs[0][1]:
        assert not any(nul in text for text in _strings([row.url, row.detail])), row.source
    assert all(nul not in c.company for c in result.press)


def test_a_nul_in_an_address_or_ticker_is_a_changed_page_not_a_stored_string(tmp_path):
    nul = "\x00"
    world = World(base_ours())
    world.press_listing = press_listing_html(
        [("Oct 30, 2026, 17:51 ET", f"x-sp-500-{nul}-302925405", "Okta")]
    )
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and codes(result) == {"press_structure"}
    assert all(nul not in text for r in store.runs[0][1] for text in _strings([r.url, r.detail]))
    row = ("Oct 20, 2026", "S&P 500", "Addition", "Co", f"X{nul}1")
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(press_release_html([row]), release())
    assert err.value.code == "bad_row"


def test_eight_maximum_size_hostile_pages_cost_only_a_few_seconds():
    page = b'<html><body><div class="release-body container">' + b"<" * (mp.MAX_PAGE_BYTES - 60)
    assert len(page) <= mp.MAX_PAGE_BYTES

    def parse_all():
        for _ in range(8):
            with pytest.raises(SourceStructureError):
                mp.parse_press_release(page, release())

    assert _timed(parse_all) < 5.0


# --------------------------------------------------------------------------- review round 5


class _Balanced(__import__("html.parser", fromlist=["HTMLParser"]).HTMLParser):
    """Telegram's HTML subset: every <b> / <code> must be closed, nothing else is allowed."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.bad = False

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.bad = True


def _well_formed(text: str) -> bool:
    parser = _Balanced()
    parser.feed(text)
    parser.close()
    import re

    bare = re.sub(r"&(amp|lt|gt|quot|#x27);", "", text)
    return not parser.bad and not parser.stack and "&" not in bare


def test_a_flood_of_press_warnings_still_gives_a_valid_message_with_the_commands(tmp_path):
    rows = [("Aug 3, 2026", "S&P 500", "Addition", f"C{i}", f"ZQ{i:03d}") for i in range(60)]
    _, _, result, store, notifier = fabricated_mismatch(
        tmp_path, tweak=lambda w: _announce(w, rows)
    )
    message = notifier.messages[0]
    assert len(message) <= 4096 and _well_formed(message)
    assert "and 5" in message and "more (see the log)" in message  # 60 warnings -> 8 shown + count
    assert "ggt db membership proposal --id 1 --out proposed_events.csv" in message
    assert "--commit</code>" in message  # the commands survive: findings were capped, not the tail


@pytest.mark.parametrize(
    "tail",
    ["x" * 5000, "<code>" + "y" * 5000, "&amp;" * 1500, "<b>" + "z" * 3490 + "<co"],
)
def test_an_over_long_message_is_cut_into_well_formed_html(tail):
    from ggTrader.data import membership_weekly as mw

    cut = mw._fit_message("head\n" + tail)
    assert len(cut) <= mw.MAX_MESSAGE_CHARS and cut.endswith("see the log and the file)")
    assert _well_formed(cut)


def test_more_table_rows_than_any_real_release_is_refused_even_for_other_indices():
    tiny = "<tr><td>x</td></tr>" * (mp.MAX_TABLE_ROWS + 1)  # small enough to fit the parsed window
    page = b'<html><div class="release-body container"><table>' + tiny.encode() + b"</table>"
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(page, release())
    assert err.value.code == "too_many_rows"


def test_the_published_limits_are_the_documented_ones():
    assert (mp.MAX_PAGE_BYTES, mp.MAX_BODY_CHARS, mp.MAX_TABLE_ROWS) == (1_000_000, 300_000, 2000)
    assert (mp.MAX_SP500_ROWS, mp.MAX_RELEASES, mp.LOOKBACK_DAYS) == (200, 8, 75)


def test_a_long_title_is_cut_and_c1_controls_are_removed():
    title = "Okta\x85\x9f " + "<" * 50 + "t" * 5000
    page = press_listing_html([("Oct 30, 2026, 17:51 ET", OKTA_SLUG, title)])
    found = mp.parse_press_listing(page)
    assert len(found[0].title) <= mp.MAX_TITLE_CHARS
    assert "\x85" not in found[0].title and "\x9f" not in found[0].title


def test_bad_row_problems_never_keep_a_whole_hostile_cell():
    row = ("Nov 4, 2026", "S&P 500", "Addition", "Co", "A" * 100_000)
    with pytest.raises(SourceStructureError) as err:
        mp.parse_press_release(press_release_html([row]), release())
    assert err.value.code == "bad_row"
    assert all(len(p) <= 200 for p in err.value.problems) and len(err.value.problems) <= 3
