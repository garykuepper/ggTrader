"""Load frozen, checksummed monthly returns from the Ken French Data Library."""

import io
import json
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from ggTrader.utils.paths import find_project_root

PROJECT_ROOT = find_project_root()
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SIZE_MOM_FILE = "6_Portfolios_ME_Prior_12_2_CSV.zip"
FACTORS_FILE = "F-F_Research_Data_Factors_CSV.zip"
VALUE_FILE = "6_Portfolios_2x3_CSV.zip"
PROFITABILITY_FILE = "6_Portfolios_ME_OP_2x3_CSV.zip"
INVESTMENT_FILE = "6_Portfolios_ME_INV_2x3_CSV.zip"
FACTOR_SCHEMAS = {
    VALUE_FILE: ("BM", "HiBM"),
    PROFITABILITY_FILE: ("OP", "HiOP"),
    INVESTMENT_FILE: ("INV", "LoINV"),
}
FRENCH_DIR = PROJECT_ROOT / "data" / "external" / "french"
VW_SECTION = "Average Value Weighted Returns -- Monthly"
SIZE_MOM_COLUMNS = {
    "SMALL LoPRIOR": "small_lo",
    "ME1 PRIOR2": "small_mid",
    "SMALL HiPRIOR": "small_hi",
    "BIG LoPRIOR": "big_lo",
    "ME2 PRIOR2": "big_mid",
    "BIG HiPRIOR": "big_hi",
}


def _http_get(url: str) -> bytes:
    """Download one archive with a bounded network timeout."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download_and_freeze(
    files: list[str],
    dest: Path = FRENCH_DIR,
    http_get: Callable[[str], bytes] | None = None,
    force: bool = False,
) -> dict:
    """Save archives and their provenance, refusing overwrites unless forced.

    Args:
        files: Archive filenames relative to the French FTP directory.
        dest: Directory for the archives and MANIFEST.json.
        http_get: Optional offline substitute returning response bytes.
        force: Explicit permission to replace existing archives.

    Returns:
        The manifest, including any previously frozen entries.

    Raises:
        FileExistsError: An archive exists and force is false.
        ValueError: An archive name is not a plain filename.
    """
    for filename in files:
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Expected a plain French archive filename: {filename!r}")
        if (dest / filename).exists() and not force:
            raise FileExistsError(f"Frozen French archive already exists: {dest / filename}")

    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    fetch = http_get or _http_get
    for filename in files:
        url = FRENCH_BASE_URL + filename
        raw = fetch(url)
        with (dest / filename).open("wb" if force else "xb") as archive:
            archive.write(raw)
        manifest[filename] = {
            "url": url,
            "sha256": sha256(raw).hexdigest(),
            "bytes": len(raw),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(manifest_path)
    return manifest


def verify_manifest(dest: Path = FRENCH_DIR) -> dict:
    """Return the manifest or raise ValueError for missing or changed files."""
    manifest_path = dest / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing French manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for filename, metadata in manifest.items():
        path = dest / filename
        if not path.is_file():
            raise ValueError(f"Missing frozen French archive: {path}")
        if sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Checksum mismatch for frozen French archive: {path}")
    return manifest


def parse_french_monthly(text: str, section_title: str | None) -> pd.DataFrame:
    """Parse one monthly table into decimal returns indexed by month end.

    Args:
        text: Decoded French library CSV content, including its preamble.
        section_title: Exact section title, or None for the first table.

    Returns:
        A float frame with stripped source headers and NaN missing codes.

    Raises:
        ValueError: The section or monthly table is missing or malformed.
    """
    lines = text.splitlines()
    if section_title is not None:
        start = next((i for i, line in enumerate(lines) if line.strip() == section_title), None)
        if start is None:
            raise ValueError(f"French section not found: {section_title!r}")
        lines = lines[start + 1 :]
        # The header must directly follow the title; searching further would
        # silently read the next section (e.g. equal- for value-weighted).
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None or not lines[first].lstrip().startswith(","):
            raise ValueError(f"French table header missing directly after {section_title!r}")
    header = next((i for i, line in enumerate(lines) if line.lstrip().startswith(",")), None)
    if header is None:
        raise ValueError(f"French monthly table not found after {section_title!r}")
    table = [lines[header]]
    for line in lines[header + 1 :]:
        if not line.strip() or "annual" in line.casefold():
            break
        table.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(table)), index_col=0)
    months = frame.index.astype(str).str.strip()
    if frame.empty or not months.str.fullmatch(r"\d{6}").all():
        raise ValueError(f"Missing or invalid monthly dates after {section_title!r}")
    frame.index = pd.to_datetime(months, format="%Y%m") + pd.offsets.MonthEnd(0)
    frame.columns = frame.columns.str.strip()
    return frame.astype(float).replace([-99.99, -999], float("nan")) / 100


def _read_frozen_csv(filename: str, dest: Path) -> str:
    """Verify frozen archives before reading the sole CSV member."""
    if filename not in verify_manifest(dest):
        raise ValueError(f"French archive is not recorded in MANIFEST.json: {filename}")
    with zipfile.ZipFile(dest / filename) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member in French archive: {filename}")
        return archive.read(members[0]).decode("utf-8-sig")


def load_size_momentum_vw(dest: Path = FRENCH_DIR) -> pd.DataFrame:
    """Load six VW size/momentum portfolios, raising KeyError for missing headers."""
    frame = parse_french_monthly(_read_frozen_csv(SIZE_MOM_FILE, dest), VW_SECTION)
    source_columns = {"".join(column.casefold().split()): column for column in frame.columns}
    rename = {}
    for source, target in SIZE_MOM_COLUMNS.items():
        normalized = "".join(source.casefold().split())
        if normalized not in source_columns:
            raise KeyError(f"Missing French size/momentum header: {source}")
        rename[source_columns[normalized]] = target
    return frame.rename(columns=rename)[list(SIZE_MOM_COLUMNS.values())]


def load_market_total_return(dest: Path = FRENCH_DIR) -> pd.DataFrame:
    """Load market total return (Mkt-RF plus RF) and the risk-free return."""
    frame = parse_french_monthly(_read_frozen_csv(FACTORS_FILE, dest), None)
    return pd.DataFrame({"mkt": frame["Mkt-RF"] + frame["RF"], "rf": frame["RF"]})


def load_size_factor_vw(filename: str, dest: Path = FRENCH_DIR) -> pd.DataFrame:
    """Load the registered small/big factor legs after an exact six-column schema check."""
    characteristic, leg = FACTOR_SCHEMAS[filename]
    frame = parse_french_monthly(_read_frozen_csv(filename, dest), VW_SECTION)
    expected = [
        f"SMALL Lo{characteristic}",
        f"ME1 {characteristic}2",
        f"SMALL Hi{characteristic}",
        f"BIG Lo{characteristic}",
        f"ME2 {characteristic}2",
        f"BIG Hi{characteristic}",
    ]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected French factor headers in {filename}: {list(frame.columns)}")
    return frame[[f"SMALL {leg}", f"BIG {leg}"]].set_axis(["small", "big"], axis=1)
