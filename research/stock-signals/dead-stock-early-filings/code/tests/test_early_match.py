"""Tests for early-window company-name matching v1.1 (brief amendment 3)."""

from __future__ import annotations

import pytest
from early_match import (
    ABBREVIATIONS,
    ADDED_ABBREVIATIONS,
    MATCH_RULE1,
    MATCH_RULES_2_TO_4,
    PLURAL_TO_SINGULAR,
    SUFFIX_WORDS,
    TRUNCATED_NAME_LENGTH,
    US_STATE_CODES,
    V11_ABBREVIATIONS,
    RosterNameIndex,
    classify_match,
    name_matches,
    normalize_name,
    normalize_name_untrimmed,
    roster_name_key,
)

pytestmark = pytest.mark.study


def test_v11_equal_after_cleaning() -> None:
    assert name_matches("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP")
    assert name_matches("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP /NC/")
    assert name_matches(
        "NATL PROCESSING INC", len("NATL PROCESSING INC"), "NATIONAL PROCESSING INC"
    )
    assert name_matches("DALLAS SEMICOND", len("DALLAS SEMICOND"), "DALLAS SEMICONDUCTOR CORP")
    assert name_matches("UNICOM CP", len("UNICOM CP"), "UNICOM CORP")
    assert name_matches("CONOCO INC", len("CONOCO INC"), "CONOCO INC /DE/")
    assert name_matches("CONOCO INC", len("CONOCO INC"), "CONOCO INC")

    # Earlier study's v1.1 rules: slash tags, share-class markers, plural folding, trailing suffix.
    assert normalize_name("CYTEC INDUSTRIES INC/DE/") == "CYTEC INDUSTRY"
    assert normalize_name("ACME CORP/NEW") == "ACME"
    assert normalize_name("AT&T/EDISON") == "AT T EDISON"
    assert normalize_name("ALPHARMA INC-A") == "ALPHARMA"
    assert normalize_name("ALPHARMA INC, CLASS B") == "ALPHARMA"
    assert normalize_name("ESTERLINE TECH") == normalize_name("Esterline Technologies Corp")
    assert normalize_name("ACME COMPANIES") == "ACME"
    assert normalize_name_untrimmed("ACME COMPANIES") == "ACME COMPANY"
    assert normalize_name("Inc.") == "INC"


def test_added_short_forms() -> None:
    expected = {
        "ENRG": ("ENERGY", "NORTH ENERGY CORP"),
        "CPTL": ("CAPITAL", "NORTH CAPITAL CORP"),
        "GP": ("GROUP", "NORTH GROUP INC"),
        "SEMICOND": ("SEMICONDUCTOR", "NORTH SEMICONDUCTOR CORP"),
        "MANG": ("MANAGEMENT", "NORTH MANAGEMENT INC"),
        "MGMT": ("MANAGEMENT", "NORTH MANAGEMENT INC"),
        "BROS": ("BROTHERS", "NORTH BROTHERS INC"),
        "ASSUR": ("ASSURANCE", "NORTH ASSURANCE CO"),
        "FIN": ("FINANCIAL", "NORTH FINANCIAL CORP"),
        "1ST": ("FIRST", "FIRST NORTH CORP"),
        "HOLD": ("HOLDINGS", "NORTH HOLDINGS INC"),
        "COR": ("CORP", "NORTH CORP"),
        "PAC": ("PACIFIC", "NORTH PACIFIC CORP"),
        "ENTERTAIN": ("ENTERTAINMENT", "NORTH ENTERTAINMENT INC"),
        "TECHNOL": ("TECHNOLOGY", "NORTH TECHNOLOGY CORP"),
        "COMMS": ("COMMUNICATIONS", "NORTH COMMUNICATIONS INC"),
        "NTWK": ("NETWORK", "NORTH NETWORK INC"),
        "NTWKS": ("NETWORKS", "NORTH NETWORKS INC"),
        "SOFTWAR": ("SOFTWARE", "NORTH SOFTWARE INC"),
        "DEVELOPME": ("DEVELOPMENT", "NORTH DEVELOPMENT CORP"),
        "PHARMACEUTI": ("PHARMACEUTICALS", "NORTH PHARMACEUTICALS INC"),
    }
    assert "LAB" not in ABBREVIATIONS
    assert set(ADDED_ABBREVIATIONS) == set(expected)
    assert set(V11_ABBREVIATIONS).isdisjoint(set(ADDED_ABBREVIATIONS))
    assert ABBREVIATIONS == {**V11_ABBREVIATIONS, **ADDED_ABBREVIATIONS}

    for short_form, (expanded, filer_name) in expected.items():
        assert ABBREVIATIONS[short_form] == expanded
        roster = f"{short_form} NORTH" if short_form == "1ST" else f"NORTH {short_form}"
        assert len(roster) != TRUNCATED_NAME_LENGTH
        assert name_matches(roster, len(roster), filer_name), (
            f"short form {short_form!r} failed to match {filer_name!r}"
        )


def test_cutoff_prefix_rule_only_for_16_characters() -> None:
    roster_16 = "MIDAMERICAN ENRG"
    assert len(roster_16) == 16
    assert name_matches(roster_16, len(roster_16), "MIDAMERICAN ENERGY HOLDINGS CO")
    assert name_matches(roster_16, len(roster_16), "MIDAMERICAN ENERGY CO")
    # A 16-character name also matches an equal cleaned filer name.
    assert name_matches(roster_16, len(roster_16), "MIDAMERICAN ENERGY")

    # A 15-character roster name does NOT match a longer filer name by prefix.
    roster_15 = "MIDAMERICAN ENR"
    assert len(roster_15) == 15
    assert not name_matches(roster_15, len(roster_15), "MIDAMERICAN ENERGY HOLDINGS CO")
    assert not name_matches(roster_15, len(roster_15), "MIDAMERICAN ENERGY CO")

    roster_15_clean = "ALPHA ENERGY CO"
    assert len(roster_15_clean) == 15
    assert not name_matches(roster_15_clean, len(roster_15_clean), "ALPHA ENERGY HOLDINGS CO")

    # Cut-off corporate suffix at 16 characters still matches via untrimmed prefix.
    cutoff_inc = "ATHLON ENERGY IN"
    assert len(cutoff_inc) == 16
    assert name_matches(cutoff_inc, len(cutoff_inc), "Athlon Energy Inc.")
    assert not name_matches("ATHLON ENERGY I", 15, "Athlon Energy Inc.")


def test_state_code_dropped_on_roster_side_only() -> None:
    assert len(US_STATE_CODES) == 50
    assert "DC" not in US_STATE_CODES

    # 3 words after cleaning (including the state code): NC dropped, then CORP dropped.
    assert roster_name_key("WACHOVIA CP NC") == "WACHOVIA"
    assert roster_name_key("WACHOVIA CORP NC") == "WACHOVIA"
    assert name_matches("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP")
    assert name_matches("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP /NC/")

    # Two-word roster name ending in a state code keeps it.
    assert roster_name_key("PHARMACIA OH") == "PHARMACIA OH"
    assert not name_matches("PHARMACIA OH", len("PHARMACIA OH"), "PHARMACIA")
    assert not name_matches("PHARMACIA OH", len("PHARMACIA OH"), "PHARMACIA CORP")
    assert name_matches("PHARMACIA OH", len("PHARMACIA OH"), "PHARMACIA OH")

    # Filer name is never state-stripped.
    assert normalize_name("WACHOVIA CORP NC") == "WACHOVIA CORP NC"
    assert not name_matches("WACHOVIA CORP", len("WACHOVIA CORP"), "WACHOVIA CORP NC")
    assert not name_matches("ALPHA BETA", len("ALPHA BETA"), "ALPHA BETA OH")


def test_no_false_equality() -> None:
    assert not name_matches("COLUMBIA ENERGY", len("COLUMBIA ENERGY"), "COLUMBIA GAS")
    assert not name_matches(
        "AMERICAN GENERAL CORP", len("AMERICAN GENERAL CORP"), "AMERICAN GENERAL FINANCE"
    )
    # Empty / whitespace / punctuation-only names never match (even at length 16).
    assert not name_matches("", 0, "ALPHA CORP")
    assert not name_matches("   ", 3, "   ")
    assert not name_matches("ALPHA CORP", len("ALPHA CORP"), "")
    assert not name_matches("ALPHA CORP", len("ALPHA CORP"), "   ")
    assert not name_matches(" " * 16, 16, "ALPHA CORP")
    assert not name_matches("-" * 16, 16, "ALPHA CORP")
    # Prefix direction is roster -> filer only (never filer shorter than roster).
    assert not name_matches("AMYLIN PHARMACEU", 16, "AMYLIN")


def test_classify_match_distinguishes_rule1_from_rules_2_to_4() -> None:
    # Rule 1 matches (base v1.1 table, slash tags, corporate suffixes, exact equality).
    assert (
        classify_match("NATL PROCESSING INC", len("NATL PROCESSING INC"), "NATIONAL PROCESSING INC")
        == MATCH_RULE1
    )
    assert classify_match("UNICOM CP", len("UNICOM CP"), "UNICOM CORP") == MATCH_RULE1
    assert classify_match("CONOCO INC", len("CONOCO INC"), "CONOCO INC /DE/") == MATCH_RULE1
    # A 16-char name that already matches exactly under Rule 1 stays Rule 1.
    assert (
        classify_match("LATE RENAMED INC", len("LATE RENAMED INC"), "LATE RENAMED CORP")
        == MATCH_RULE1
    )
    # Trailing CO on a 3-word name is a Rule 1 corporate suffix, not a Rule 4 state code.
    assert (
        classify_match("EXAMPLE ENERGY CO", len("EXAMPLE ENERGY CO"), "EXAMPLE ENERGY CO")
        == MATCH_RULE1
    )

    # Rule 2 (16-char cut-off prefix without Rule 1 exact match).
    assert (
        classify_match("NORTHSTAR WIDGET", 16, "NORTHSTAR WIDGET HOLDINGS INC")
        == MATCH_RULES_2_TO_4
    )
    assert classify_match("ATHLON ENERGY IN", 16, "Athlon Energy Inc.") == MATCH_RULES_2_TO_4
    # Rule 2 + Rule 3.
    assert (
        classify_match("MIDAMERICAN ENRG", 16, "MIDAMERICAN ENERGY HOLDINGS CO")
        == MATCH_RULES_2_TO_4
    )
    # Rule 3 (added short form).
    assert (
        classify_match("DALLAS SEMICOND", len("DALLAS SEMICOND"), "DALLAS SEMICONDUCTOR CORP")
        == MATCH_RULES_2_TO_4
    )
    # Rule 4 (roster-side trailing state code).
    assert (
        classify_match("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP")
        == MATCH_RULES_2_TO_4
    )
    assert (
        classify_match("WACHOVIA CP NC", len("WACHOVIA CP NC"), "WACHOVIA CORP /NC/")
        == MATCH_RULES_2_TO_4
    )


def test_name_index_agrees_with_name_matches() -> None:
    roster_names = [
        "WACHOVIA CP NC",
        "PHARMACIA OH",
        "NATL PROCESSING INC",
        "DALLAS SEMICOND",
        "UNICOM CP",
        "CONOCO INC",
        "MIDAMERICAN ENRG",  # 16 chars
        "MIDAMERICAN ENR",  # 15 chars
        "ATHLON ENERGY IN",  # 16 chars
        "NORTHSTAR WIDGET",  # 16 chars
        "COLUMBIA ENERGY",
        "AMERICAN GENERAL CORP",
        "ACME",  # short non-16
        "ACME ENERGY HOLD",  # 16 chars
        "ACME ENERGY HLDG",  # 16 chars (nested prefix test)
        "ACME CORP       ",  # 16 chars with spaces -> untrimmed "ACME CORP" (nested prefix test)
        "ALPHARMA INC-A",
        "ESTERLINE TECH",
        "FIRST BANK OH",
        "   ",
        "",
    ]
    filer_names = [
        "WACHOVIA CORP",
        "WACHOVIA CORP /NC/",
        "WACHOVIA CORP NC",
        "PHARMACIA",
        "PHARMACIA OH",
        "NATIONAL PROCESSING INC",
        "DALLAS SEMICONDUCTOR CORP",
        "UNICOM CORP",
        "CONOCO INC /DE/",
        "CONOCO INC",
        "MIDAMERICAN ENERGY HOLDINGS CO",
        "MIDAMERICAN ENERGY CO",
        "MIDAMERICAN ENERGY",
        "Athlon Energy Inc.",
        "NORTHSTAR WIDGET CORP",
        "NORTHSTAR WIDGET HOLDINGS INC",
        "COLUMBIA GAS",
        "COLUMBIA ENERGY GROUP",
        "AMERICAN GENERAL FINANCE",
        "AMERICAN GENERAL CORP",
        "ACME CORP",
        "ACME CORPORATION",
        "ACME CORP INTERNATIONAL",
        "ACME ENERGY HOLDING CO",
        "ACME ENERGY HLDG INC",
        "ACME ENERGY HOLDINGS INC",
        "ALPHARMA INC",
        "Esterline Technologies Corp",
        "FIRST BANK",
        "FIRST BANK /OH/",
        "FIRST BANK OH",
        "UNRELATED COMPANY LTD",
        "",
        "   ",
    ]

    index = RosterNameIndex(roster_names)
    for filer in filer_names:
        expected_set = {r for r in roster_names if name_matches(r, len(r), filer)}
        assert index.match(filer) == expected_set, f"Mismatch on filer {filer!r}"

        expected_rules = {
            r: rule for r in roster_names if (rule := classify_match(r, len(r), filer)) is not None
        }
        assert index.match_rules(filer) == expected_rules, f"Rule mismatch on filer {filer!r}"

    # Verify explicit (key, roster_name, raw_length) items with identical text, different lengths.
    explicit = RosterNameIndex.from_items(
        [
            ("k16", "ALPHA BETA", 16),
            ("k10", "ALPHA BETA", 10),
        ]
    )
    assert explicit.match("ALPHA BETA GAMMA INC") == {"k16"}
    assert explicit.match("ALPHA BETA INC") == {"k16", "k10"}
    assert explicit.match_rules("ALPHA BETA INC") == {"k16": MATCH_RULE1, "k10": MATCH_RULE1}
    assert explicit.match_rules("ALPHA BETA GAMMA INC") == {"k16": MATCH_RULES_2_TO_4}


def test_tables_are_well_formed() -> None:
    assert all(k == k.upper() and " " not in k for k in ABBREVIATIONS)
    assert all(k == k.upper() and " " not in k for k in PLURAL_TO_SINGULAR)
    assert all(w == w.upper() and " " not in w for w in SUFFIX_WORDS)
    assert all(len(c) == 2 and c.isalpha() and c == c.upper() for c in US_STATE_CODES)


def test_lab_withdrawal_matches_earlier_v11_behavior_direct_and_indexed() -> None:
    """Amendment 4 regression: LAB withdrawn so LAB/LABS match earlier v1.1 behaviour."""
    roster_16 = "GREAT PLAINS LAB"
    filer_prefix = "GREAT PLAINS LABS SERVICES INC"
    assert len(roster_16) == TRUNCATED_NAME_LENGTH
    assert normalize_name_untrimmed(roster_16) == "GREAT PLAINS LAB"
    assert normalize_name_untrimmed(filer_prefix) == "GREAT PLAINS LAB SERVICE INC"
    assert name_matches(roster_16, len(roster_16), filer_prefix)
    assert classify_match(roster_16, len(roster_16), filer_prefix) == MATCH_RULES_2_TO_4

    # Brief amendment 4 withdraws LAB -> LABORATORIES, keeping the earlier v1.1 rule as-is:
    # LABS folds to LAB while LABORATORIES folds to LABORATORY, so this pair does not match.
    roster_labs = "NORTH LABS"
    filer_laboratories = "NORTH LABORATORIES INC"
    assert normalize_name(roster_labs) == "NORTH LAB"
    assert normalize_name(filer_laboratories) == "NORTH LABORATORY"
    assert not name_matches(roster_labs, len(roster_labs), filer_laboratories)
    assert classify_match(roster_labs, len(roster_labs), filer_laboratories) is None

    # Direct matcher and indexed lookup agree on both counterexamples.
    index = RosterNameIndex([roster_16, roster_labs])
    assert index.match(filer_prefix) == {roster_16}
    assert index.match_rules(filer_prefix) == {roster_16: MATCH_RULES_2_TO_4}
    assert index.match(filer_laboratories) == set()
    assert index.match_rules(filer_laboratories) == {}
