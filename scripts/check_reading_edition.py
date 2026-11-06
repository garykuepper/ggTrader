#!/usr/bin/env python3
"""Lint a plain-English reading edition for banned punctuation, labels, terms and numbers."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Tests import this module as `scripts.check_reading_edition`; the CLI runs it from scripts/.
try:
    from scripts.check_lens_report import _check_paths
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI
    from check_lens_report import _check_paths

EM_DASH = "—"
# Format and unit names that read as plain nouns; general English words are never added here.
DEFAULT_IGNORE: frozenset[str] = frozenset(
    {"MD", "JSON", "CSV", "PDF", "HTML", "URL", "US", "USD", "UTC"}
)

CODE_SPAN_RE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
ROW_SPLIT_RE = re.compile(r"(?<!\\)\|")
HTML_COMMENT_START = "<!--"
HTML_COMMENT_END = "-->"
SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+")
FILE_NAME_RE = re.compile(
    r"[\w./-]+\.(?:md|json|csv|py|txt|gz|tgz|yml|yaml|toml)\b(?::\d+(?:-\d+)?)?"
)
REPO_PATH_BARE_RE = re.compile(r"\b(?:docs|src|scripts|tests|results|research|data)/[\w./-]+")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# An abbreviation is a run of capitals, digits and ampersands (hyphen-joined parts stay one
# token) with an optional plural s that is not part of the stem. A leading dot excludes
# domain-style names such as Binance.US.
ABBREV_RE = re.compile(r"(?<![A-Za-z0-9_.])([A-Z][A-Z0-9&]*(?:-[A-Z0-9&]+)*)s?(?![A-Za-z0-9_])")
MIN_ABBREV_LETTERS = 2
LABEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?i:lens|phase|part|gate)(?:e?s)?[\s-]*(?:\d+|[A-D](?![A-Za-z]))"),
    re.compile(r"\bB0\b"),
    re.compile(r"\bB1\b"),
    re.compile(r"\bP2-\d+"),
)
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?%)?")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
MIN_PLAIN_DIGITS = 4
WORD_RE = re.compile(r"[A-Za-z0-9][\w'’%.&-]*")
FILE_COLUMN_HEADERS = frozenset({"file", "files", "file name"})
LABEL_ADVICE = "use the reference-sheet name, or a code span when citing an id"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _Line:
    """One checkable Markdown line with its code spans masked."""

    number: int
    raw: str
    text: str
    label_text: str
    kind: str  # "prose", "table" or "front"


def _mask_code_spans(line: str) -> str:
    """Replace inline code spans with spaces, keeping column positions."""
    return CODE_SPAN_RE.sub(lambda match: " " * len(match.group(0)), line)


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into its cells."""
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return ROW_SPLIT_RE.split(body)


def _parse_lines(text: str) -> list[_Line]:
    """Classify lines, dropping fences, fence bodies and table separator rows."""
    raw_lines = text.splitlines()
    front_end = 0
    if raw_lines and raw_lines[0].strip() == "---":
        for index in range(1, len(raw_lines)):
            if raw_lines[index].strip() == "---":
                front_end = index
                break
    parsed: list[_Line] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    in_table = False
    file_columns: set[int] = set()
    for index, raw in enumerate(raw_lines):
        number = index + 1
        if index <= front_end and front_end:
            if 0 < index < front_end:
                parsed.append(
                    _Line(number, raw, _mask_code_spans(raw), _mask_code_spans(raw), "front")
                )
            continue
        opener = FENCE_RE.match(raw)
        if fence is not None:
            marker = raw.strip()
            if marker and set(marker) == {fence[0]} and len(marker) >= fence[1]:
                fence = None
            continue
        if opener:
            fence = (opener.group(1)[0], len(opener.group(1)))
            continue
        masked = _mask_code_spans(raw)
        if HTML_COMMENT_START in raw:
            in_comment = True
        commented = in_comment
        if HTML_COMMENT_END in raw:
            in_comment = False
        if not masked.strip().startswith("|"):
            in_table = False
            kind = "comment" if commented else "prose"
            parsed.append(_Line(number, raw, masked, masked, kind))
            continue
        if SEPARATOR_RE.match(masked):
            continue
        if not in_table:
            in_table = True
            names = [cell.strip().strip("`").lower() for cell in _split_row(raw)]
            file_columns = {i for i, name in enumerate(names) if name in FILE_COLUMN_HEADERS}
            parsed.append(_Line(number, raw, masked, masked, "table"))
            continue
        cells = _split_row(masked)
        kept = [" " * len(c) if i in file_columns else c for i, c in enumerate(cells)]
        parsed.append(_Line(number, raw, masked, "|".join(kept), "table"))
    return parsed


def _strip_refs(text: str) -> str:
    """Blank links, URLs, file names and repo paths so their characters are not read as prose."""
    for pattern in (LINK_TARGET_RE, URL_RE, FILE_NAME_RE, REPO_PATH_BARE_RE):
        text = pattern.sub(lambda match: " " * len(match.group(0)), text)
    return text


def _letter_count(token: str) -> int:
    """Count the letters in a token."""
    return sum(char.isalpha() for char in token)


def load_sheet_terms(sheet: Path) -> set[str]:
    """Collect the defined terms from the first column of every table row in the sheet.

    Args:
        sheet: Reference sheet Markdown file.

    Returns:
        Abbreviation tokens found in first-column cells plus each cell split on `/`, `,`
        and ` or `, with backticks stripped.
    """
    terms: set[str] = set()
    for line in Path(sheet).read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|") or SEPARATOR_RE.match(line):
            continue
        cell = _split_row(line)[0].replace("`", "").strip()
        for token in ABBREV_RE.findall(cell):
            if _letter_count(token) >= MIN_ABBREV_LETTERS:
                terms.add(token)
        for entry in re.split(r",|\bor\b", cell):
            terms.add(entry.strip())
            terms.update(part.strip() for part in entry.split("/"))
        terms.discard("")
    return terms


def _number_variants(whole: str, decimals: str | None) -> list[str]:
    """Return a number's normalized string plus a trailing-zero-trimmed form when useful."""
    base = whole.replace(",", "") + (decimals or "")
    variants = [base]
    if decimals:
        trimmed = decimals.rstrip("0")
        if trimmed != ".":
            variants.append(whole.replace(",", "") + trimmed)
    return variants


@lru_cache(maxsize=8)
def _source_numbers(texts: tuple[str, ...]) -> tuple[frozenset[str], frozenset[str]]:
    """Index every number in the sources as (all numbers, numbers written with a percent sign)."""
    plain: set[str] = set()
    percent: set[str] = set()
    for text in texts:
        for match in NUMBER_RE.finditer(text):
            variants = _number_variants(match.group(1), match.group(2))
            plain.update(variants)
            if match.group(3):
                percent.update(variants)
    return frozenset(plain), frozenset(percent)


def _is_tracked(whole: str, decimals: str | None, percent: str | None, after_dollar: bool) -> bool:
    """Say whether a number needs a source.

    Decimals, percentages and numbers of four or more digits are tracked. A bare year is
    exempt, but a year-looking number with a thousands comma or a leading `$` is not.
    """
    if decimals or percent:
        return True
    digits = whole.replace(",", "")
    if len(digits) < MIN_PLAIN_DIGITS:
        return False
    return after_dollar or not YEAR_RE.match(whole)


def _check_punctuation(line: _Line) -> list[str]:
    """Report em dashes and semicolons outside code."""
    found: list[str] = []
    text = LINK_TARGET_RE.sub(lambda match: " " * len(match.group(0)), line.text)
    if EM_DASH in text:
        found.append("em dash (U+2014) in prose")
    if ";" in text:
        found.append("semicolon in prose")
    return found


def _check_labels(line: _Line) -> list[str]:
    """Report lens, phase, part and gate numbers outside code and the File column."""
    text = _strip_refs(line.label_text)
    return [
        f"numbered label `{match.group(0)}`: {LABEL_ADVICE}"
        for pattern in LABEL_PATTERNS
        for match in pattern.finditer(text)
    ]


def _check_terms(line: _Line, known: set[str]) -> list[str]:
    """Report all-caps tokens that are neither sheet terms nor ignored."""
    found: list[str] = []
    for token in ABBREV_RE.findall(_strip_refs(line.text)):
        if _letter_count(token) < MIN_ABBREV_LETTERS or token in known:
            continue
        # A hyphenated token passes when every part is known; otherwise name the unknown parts.
        parts = token.split("-") if "-" in token else [token]
        for part in parts:
            if _letter_count(part) >= MIN_ABBREV_LETTERS and part not in known:
                where = f" (in `{token}`)" if part != token else ""
                found.append(f"abbreviation `{part}`{where} is not defined in the reference sheet")
    return found


def _check_numbers(line: _Line, plain: frozenset[str], percent: frozenset[str]) -> list[str]:
    """Report tracked numbers that no source text contains."""
    text = ISO_DATE_RE.sub(lambda match: " " * len(match.group(0)), _strip_refs(line.text))
    found: list[str] = []
    for match in NUMBER_RE.finditer(text):
        whole, decimals, pct = match.groups()
        after_dollar = match.start() > 0 and text[match.start() - 1] == "$"
        if not _is_tracked(whole, decimals, pct, after_dollar):
            continue
        variants = _number_variants(whole, decimals)
        if pct and not decimals:
            traced = any(v in percent for v in variants)
        else:
            traced = any(v in plain for v in variants)
        if not traced:
            found.append(f"number {variants[0]}{'%' if pct else ''} not found in any source text")
    return found


def check_file(
    path: Path,
    sheet_terms: set[str],
    source_texts: dict[str, str],
    max_words: int | None = None,
    ignore_terms: frozenset[str] = DEFAULT_IGNORE,
    root: Path | None = None,
) -> list[str]:
    """Lint one reading-edition file and return its problems.

    Args:
        path: Markdown file to check.
        sheet_terms: Terms defined in the reference sheet.
        source_texts: Source documents (path to text) that numbers must trace to.
        max_words: Optional cap on prose words outside tables, code and front matter.
        ignore_terms: Capitalised tokens that need no definition.
        root: Repository root for resolving cited paths (default: the current directory).

    Returns:
        Problem strings of the form `<path>:<line>: <rule>: <detail>`.
    """
    repo_root = Path.cwd() if root is None else Path(root)
    known = set(sheet_terms) | set(ignore_terms)
    own = Path(path).resolve()
    others = {k: v for k, v in source_texts.items() if Path(k).resolve() != own}
    plain, percent = _source_numbers(tuple(others.values()))
    lines = _parse_lines(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    missing_seen: set[str] = set()
    words = 0
    for line in lines:
        prefix = f"{path}:{line.number}"
        problems += [f"{prefix}: punctuation: {d}" for d in _check_punctuation(line)]
        problems += [f"{prefix}: numbered-label: {d}" for d in _check_labels(line)]
        problems += [f"{prefix}: undefined-term: {d}" for d in _check_terms(line, known)]
        problems += [
            f"{prefix}: untraceable-number: {d}" for d in _check_numbers(line, plain, percent)
        ]
        for missing in _check_paths(line.raw, repo_root):
            if "<" not in missing and missing not in missing_seen:
                missing_seen.add(missing)
                problems.append(f"{prefix}: missing-path: {missing}")
        if line.kind == "prose":
            words += len(WORD_RE.findall(LINK_TARGET_RE.sub(" ", line.text)))
    if max_words is not None and words > max_words:
        problems.append(f"{path}:1: word-cap: {words} prose words exceeds the cap of {max_words}")
    return problems


def _read_sources(paths: list[Path]) -> dict[str, str]:
    """Read each source file into a path-to-text mapping."""
    return {str(p): Path(p).read_text(encoding="utf-8") for p in paths}


def main(argv: list[str] | None = None) -> int:
    """Run the lint from the command line; exit 0 when clean, 1 on any problem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE", help="reading edition files")
    parser.add_argument("--sheet", type=Path, required=True, help="reference sheet Markdown file")
    parser.add_argument(
        "--sources",
        nargs="*",
        type=Path,
        default=[],
        metavar="FILE",
        help="source documents that every tracked number must appear in",
    )
    parser.add_argument("--max-words", type=int, default=None, help="cap on prose words per file")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root for cited paths (default: this repo)",
    )
    args = parser.parse_args(argv)
    problems: list[str] = []
    try:
        terms = load_sheet_terms(args.sheet)
        sources = _read_sources(args.sources)
        for file in args.files:
            problems += check_file(file, terms, sources, args.max_words, root=args.root)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
