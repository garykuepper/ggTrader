"""Pure company-name cleaning and matching v1.1 for dead-stock-early-filings (no I/O).

Implements the five matching rules of the third dated amendment (2026-11-25) to the frozen
study brief, copying the earlier study's v1.1 tables and regexes without importing from it:

1. Earlier study's v1.1 rule: uppercase, trailing EDGAR slash tag and share-class removal,
   punctuation to spaces, whole-word abbreviation expansion (``V11_ABBREVIATIONS``),
   whole-word plural folding (``PLURAL_TO_SINGULAR``), trailing corporate-form removal
   (``SUFFIX_WORDS``, last word always kept); equal cleaned names match.
2. 16-character cut-off prefix rule: a roster name of exact raw length 16 matches a filer
   whose untrimmed cleaned name starts with the roster's untrimmed cleaned name.
3. Added short forms (``ADDED_ABBREVIATIONS``), expanded on both sides.
4. Trailing US state postal code dropped on the roster side only when the cleaned name has
   at least three words (including the state code), applied before trailing corporate words.
5. Guards: matches that rely on rules 2 to 4 are classified as ``MATCH_RULES_2_TO_4`` so
   callers keep them at tier ``name_only`` and never promote them to ``checked``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

# McGill company names are cut off at 16 characters.
TRUNCATED_NAME_LENGTH = 16

# Match classifications for confidence-tier gating (amendment 3 rule 5).
MATCH_RULE1 = "rule1"
MATCH_RULES_2_TO_4 = "rules_2_to_4"

# Trailing corporate-form words dropped after abbreviation expansion and plural folding.
SUFFIX_WORDS: frozenset[str] = frozenset(
    {"INC", "CORP", "CO", "COMPANY", "LTD", "LLC", "LP", "PLC"}
)

# Earlier study's v1.1 whole-word short forms (rule 1, copied from dead-stock-data-integrity).
V11_ABBREVIATIONS: dict[str, str] = {
    "CP": "CORP",
    "CORPORATION": "CORP",
    "INCORPORATED": "INC",
    "INDS": "INDUSTRIES",
    "INTL": "INTERNATIONAL",
    "HLDGS": "HOLDINGS",
    "GRP": "GROUP",
    "TECH": "TECHNOLOGY",
    "TECHS": "TECHNOLOGIES",
    "SYS": "SYSTEMS",
    "PHARMACEUTICAL": "PHARMACEUTICALS",
    "DRG": "DRUG",
    "STRS": "STORES",
    "AMERN": "AMERICAN",
    "NATL": "NATIONAL",
    "PHARMA": "PHARMACEUTICALS",
    "PHARMS": "PHARMACEUTICALS",
    "NAT": "NATURAL",
    "ELEC": "ELECTRIC",
    "MED": "MEDICAL",
    "CONTL": "CONTINENTAL",
    "EXPLOR": "EXPLORATION",
    "RES": "RESOURCES",
    "PETE": "PETROLEUM",
    "COMM": "COMMUNICATIONS",
    "SOLU": "SOLUTIONS",
    "SOL": "SOLUTIONS",
    "PAYMNT": "PAYMENT",
    "SOUTHN": "SOUTHERN",
    "FINL": "FINANCIAL",
    "INS": "INSURANCE",
}

# Added short forms from brief amendment 3 rule 3 (expanded on both sides).
ADDED_ABBREVIATIONS: dict[str, str] = {
    "ENRG": "ENERGY",
    "CPTL": "CAPITAL",
    "GP": "GROUP",
    "SEMICOND": "SEMICONDUCTOR",
    "MANG": "MANAGEMENT",
    "MGMT": "MANAGEMENT",
    "BROS": "BROTHERS",
    "ASSUR": "ASSURANCE",
    "FIN": "FINANCIAL",
    "1ST": "FIRST",
    "HOLD": "HOLDINGS",
    "COR": "CORP",
    "PAC": "PACIFIC",
    "ENTERTAIN": "ENTERTAINMENT",
    "TECHNOL": "TECHNOLOGY",
    "COMMS": "COMMUNICATIONS",
    "NTWK": "NETWORK",
    "NTWKS": "NETWORKS",
    "SOFTWAR": "SOFTWARE",
    "DEVELOPME": "DEVELOPMENT",
    "PHARMACEUTI": "PHARMACEUTICALS",
    "LAB": "LABORATORIES",
}

# Full whole-word abbreviation table (v1.1 base + amendment 3 rule 3 additions).
ABBREVIATIONS: dict[str, str] = {**V11_ABBREVIATIONS, **ADDED_ABBREVIATIONS}

# Fixed whole-word plural-to-singular folding table (copied from dead-stock-data-integrity).
PLURAL_TO_SINGULAR: dict[str, str] = {
    "TECHNOLOGIES": "TECHNOLOGY",
    "HOLDINGS": "HOLDING",
    "SYSTEMS": "SYSTEM",
    "PHARMACEUTICALS": "PHARMACEUTICAL",
    "COMPANIES": "COMPANY",
    "LABS": "LAB",
    "LABORATORIES": "LABORATORY",
    "INDUSTRIES": "INDUSTRY",
    "SERVICES": "SERVICE",
    "PRODUCTS": "PRODUCT",
    "RESOURCES": "RESOURCE",
    "STORES": "STORE",
}

# Trailing EDGAR state or status slash tag (e.g. "/DE/", "/NC/", "/NEW"); interior slashes stay.
EDGAR_SLASH_TAG_PATTERN = r"/[A-Z]{2,4}/?$"
_EDGAR_SLASH_TAG_RE = re.compile(EDGAR_SLASH_TAG_PATTERN)

# Trailing share-class marker (-A/-B/-C, CL A, CLASS A, etc.) stripped before suffix stripping.
SHARE_CLASS_PATTERN = r"(?:\s*-\s*(?:CL(?:ASS)?\.?\s+)?[ABC]|,?\s+CL(?:ASS)?\.?\s+[ABC])$"
_SHARE_CLASS_RE = re.compile(SHARE_CLASS_PATTERN)

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

# The 50 US state postal codes (amendment 3 rule 4, roster side only).
US_STATE_CODES: frozenset[str] = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    }
)


def _base_words(name: str) -> list[str]:
    """Uppercase, drop trailing EDGAR slash tag and share class, and split on punctuation."""
    if not isinstance(name, str):
        return []
    text = name.strip().upper()
    if not text:
        return []
    without_tag = _EDGAR_SLASH_TAG_RE.sub("", text).strip()
    if without_tag:
        text = without_tag
    without_class = _SHARE_CLASS_RE.sub("", text).strip()
    if without_class:
        text = without_class
    return _NON_ALNUM_RE.sub(" ", text).split()


def _expand_words(base: list[str], abbreviations: Mapping[str, str]) -> list[str]:
    """Expand whole-word abbreviations and fold fixed plurals to singular."""
    return [
        PLURAL_TO_SINGULAR.get(expanded, expanded)
        for word in base
        for expanded in (abbreviations.get(word, word),)
    ]


def _strip_trailing_suffixes(words: list[str]) -> list[str]:
    """Drop trailing corporate-form words while always keeping at least one word."""
    out = list(words)
    while len(out) > 1 and out[-1] in SUFFIX_WORDS:
        out.pop()
    return out


def normalize_name_untrimmed(name: str) -> str:
    """Clean a company name with v1.1 + rule-3 steps, keeping trailing corporate words."""
    base = _base_words(name)
    if not base:
        return ""
    return " ".join(_expand_words(base, ABBREVIATIONS))


def normalize_name(name: str) -> str:
    """Clean a company name with v1.1 + rule-3 steps, dropping trailing corporate words."""
    base = _base_words(name)
    if not base:
        return ""
    return " ".join(_strip_trailing_suffixes(_expand_words(base, ABBREVIATIONS)))


def normalize_name_rule1(name: str) -> str:
    """Clean a company name under rule 1 only (earlier study's v1.1 table, no rules 2-4)."""
    base = _base_words(name)
    if not base:
        return ""
    return " ".join(_strip_trailing_suffixes(_expand_words(base, V11_ABBREVIATIONS)))


def roster_name_key(name: str) -> str:
    """Clean a roster name, applying amendment rule 4's roster-side trailing state-code drop.

    If the cleaned word list (before trailing corporate-word removal) has at least three
    words (including the state code) and its last word is one of the 50 US state postal
    codes, drop that last word before stripping trailing corporate words.
    """
    base = _base_words(name)
    if not base:
        return ""
    words = _expand_words(base, ABBREVIATIONS)
    if len(words) >= 3 and words[-1] in US_STATE_CODES:
        words = words[:-1]
    return " ".join(_strip_trailing_suffixes(words))


def classify_match(roster_name: str, raw_length: int, filer_name: str) -> str | None:
    """Match ``roster_name`` against ``filer_name`` and report which rule tier matched.

    Returns:
        ``MATCH_RULE1`` if the names match under rule 1 alone (earlier study's v1.1 exact
        equality without rules 2 to 4), ``MATCH_RULES_2_TO_4`` if the match requires the
        16-character cut-off prefix (rule 2), an added short form (rule 3), or the
        roster-side state-code drop (rule 4), or ``None`` if they do not match.
    """
    if not isinstance(roster_name, str) or not isinstance(filer_name, str):
        return None
    r_base = _base_words(roster_name)
    if not r_base:
        return None
    f_base = _base_words(filer_name)
    if not f_base:
        return None

    r_rule1 = " ".join(_strip_trailing_suffixes(_expand_words(r_base, V11_ABBREVIATIONS)))
    f_rule1 = " ".join(_strip_trailing_suffixes(_expand_words(f_base, V11_ABBREVIATIONS)))
    if f_rule1 == r_rule1:
        return MATCH_RULE1

    r_expanded = _expand_words(r_base, ABBREVIATIONS)
    f_expanded = _expand_words(f_base, ABBREVIATIONS)
    f_clean = " ".join(_strip_trailing_suffixes(f_expanded))
    r_clean = " ".join(_strip_trailing_suffixes(r_expanded))
    if f_clean == r_clean:
        return MATCH_RULES_2_TO_4

    if len(r_expanded) >= 3 and r_expanded[-1] in US_STATE_CODES:
        r_state_dropped = " ".join(_strip_trailing_suffixes(r_expanded[:-1]))
        if f_clean == r_state_dropped:
            return MATCH_RULES_2_TO_4

    if raw_length == TRUNCATED_NAME_LENGTH:
        r_untrimmed = " ".join(r_expanded)
        f_untrimmed = " ".join(f_expanded)
        if f_untrimmed.startswith(r_untrimmed):
            return MATCH_RULES_2_TO_4

    return None


def name_matches(roster_name: str, raw_length: int, filer_name: str) -> bool:
    """Return True if ``roster_name`` (raw length ``raw_length``) matches ``filer_name``."""
    return classify_match(roster_name, raw_length, filer_name) is not None


class RosterNameIndex:
    """Indexed roster-name matcher for streaming quarterly SEC master indexes.

    Combines:
    - O(1) dict ``_rule1_exact`` keyed by ``normalize_name_rule1(roster_name)`` for rule 1.
    - O(1) dict ``_expanded_exact`` keyed by ``normalize_name(roster_name)`` and
      ``roster_name_key(roster_name)`` for rules 3 and 4.
    - Character trie ``_prefix_trie`` over ``normalize_name_untrimmed(roster_name)`` for
      16-character roster names (rule 2), guaranteeing exact prefix enumeration even when
      prefixes are nested and short-circuiting non-matching filers in a few character steps.
    """

    def __init__(self, roster_names: Iterable[str] = ()) -> None:
        self._rule1_exact: dict[str, list[str]] = defaultdict(list)
        self._expanded_exact: dict[str, list[str]] = defaultdict(list)
        self._prefix_trie: dict[str, Any] = {}
        self._seen_keys: set[tuple[str, str, int]] = set()
        for name in roster_names:
            if isinstance(name, str):
                self.add(name, name, len(name))

    @classmethod
    def from_items(cls, items: Iterable[tuple[str, str, int]]) -> RosterNameIndex:
        """Build an index from explicit ``(key, roster_name, raw_length)`` triples."""
        index = cls()
        for key, roster_name, raw_length in items:
            index.add(key, roster_name, raw_length)
        return index

    def add(self, key: str, roster_name: str, raw_length: int) -> None:
        """Register one roster entry under ``key``."""
        if not isinstance(roster_name, str):
            return
        entry_id = (key, roster_name, raw_length)
        if entry_id in self._seen_keys:
            return
        base = _base_words(roster_name)
        if not base:
            return
        self._seen_keys.add(entry_id)

        r1_key = " ".join(_strip_trailing_suffixes(_expand_words(base, V11_ABBREVIATIONS)))
        self._rule1_exact[r1_key].append(key)

        expanded_words = _expand_words(base, ABBREVIATIONS)
        expanded_clean = " ".join(_strip_trailing_suffixes(expanded_words))
        self._expanded_exact[expanded_clean].append(key)

        if len(expanded_words) >= 3 and expanded_words[-1] in US_STATE_CODES:
            state_dropped = " ".join(_strip_trailing_suffixes(expanded_words[:-1]))
            if state_dropped != expanded_clean:
                self._expanded_exact[state_dropped].append(key)

        if raw_length == TRUNCATED_NAME_LENGTH:
            untrimmed = " ".join(expanded_words)
            node = self._prefix_trie
            for ch in untrimmed:
                child = node.get(ch)
                if child is None:
                    child = {}
                    node[ch] = child
                node = child
            hits = node.get("")
            if hits is None:
                node[""] = [key]
            else:
                hits.append(key)

    def match_rules(self, filer_name: str) -> dict[str, str]:
        """Return ``{key: MATCH_RULE1 | MATCH_RULES_2_TO_4}`` for every matching roster entry."""
        base = _base_words(filer_name)
        if not base:
            return {}

        r1_words = _expand_words(base, V11_ABBREVIATIONS)
        r1_key = " ".join(_strip_trailing_suffixes(r1_words))
        results: dict[str, str] = {}
        r1_hits = self._rule1_exact.get(r1_key)
        if r1_hits:
            for key in r1_hits:
                results[key] = MATCH_RULE1

        if any(word in ADDED_ABBREVIATIONS for word in base):
            expanded_words = _expand_words(base, ABBREVIATIONS)
        else:
            expanded_words = r1_words

        expanded_key = " ".join(_strip_trailing_suffixes(expanded_words))
        exp_hits = self._expanded_exact.get(expanded_key)
        if exp_hits:
            for key in exp_hits:
                results.setdefault(key, MATCH_RULES_2_TO_4)

        if self._prefix_trie:
            untrimmed = " ".join(expanded_words)
            node = self._prefix_trie
            for ch in untrimmed:
                child = node.get(ch)
                if child is None:
                    break
                node = child
                prefix_hits = node.get("")
                if prefix_hits:
                    for key in prefix_hits:
                        results.setdefault(key, MATCH_RULES_2_TO_4)

        return results

    def match(self, filer_name: str) -> set[str]:
        """Return the set of roster keys matching ``filer_name``."""
        return set(self.match_rules(filer_name))


__all__ = [
    "ABBREVIATIONS",
    "ADDED_ABBREVIATIONS",
    "EDGAR_SLASH_TAG_PATTERN",
    "MATCH_RULE1",
    "MATCH_RULES_2_TO_4",
    "PLURAL_TO_SINGULAR",
    "SHARE_CLASS_PATTERN",
    "SUFFIX_WORDS",
    "TRUNCATED_NAME_LENGTH",
    "US_STATE_CODES",
    "V11_ABBREVIATIONS",
    "RosterNameIndex",
    "classify_match",
    "name_matches",
    "normalize_name",
    "normalize_name_rule1",
    "normalize_name_untrimmed",
    "roster_name_key",
]
