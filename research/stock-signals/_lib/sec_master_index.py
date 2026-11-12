"""Extract candidate listing filings from a frozen SEC master-index archive."""

import csv
import io
import re
import zipfile
from datetime import date

_FORMS = frozenset({"8-A12B", "8-A12B/A", "25", "25/A", "25-NSE", "25-NSE/A"})
_HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"
_MAX_ARCHIVE = 30_000_000
_MAX_INDEX = 64_000_000


def parse_listing_archive(content: bytes) -> list[dict]:
    """Collect candidate filings and indexed parties, without inferring event timing.

    The SEC index can list one accession under several CIKs; retain each party
    instead of assuming that an index CIK or accession prefix identifies the
    security issuer. The caller must authenticate and pin the archive, establish
    its coverage and publication history, and inspect filings for security class,
    effective date and public availability. Only a UTF-8 SEC master-index ZIP is
    supported. This is not a listing universe.
    """
    if not isinstance(content, bytes) or len(content) > _MAX_ARCHIVE:
        raise ValueError("Invalid master-index archive")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.infolist()
            if len(info) != 1 or info[0].filename != "master.idx" or info[0].file_size > _MAX_INDEX:
                raise ValueError("Invalid master-index archive contents")
            with archive.open(info[0]) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                for _ in range(50):
                    if stream.readline().rstrip("\r\n") == _HEADER:
                        break
                else:
                    raise ValueError("Missing master-index header")
                if not re.fullmatch(r"-+", stream.readline().strip()):
                    raise ValueError("Invalid master-index separator")
                filings: dict[str, dict] = {}
                for row in csv.reader(stream, delimiter="|", strict=True):
                    if len(row) != 5:
                        raise ValueError("Invalid master-index row")
                    cik, name, form, filed, filename = row
                    if form not in _FORMS:
                        continue
                    match = re.fullmatch(
                        r"edgar/data/([0-9]{1,10})/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt",
                        filename,
                    )
                    try:
                        if (
                            not re.fullmatch(r"[0-9]{1,10}", cik)
                            or int(cik) == 0
                            or not name
                            or name != name.strip()
                            or date.fromisoformat(filed).isoformat() != filed
                            or match is None
                            or int(match.group(1)) != int(cik)
                        ):
                            raise ValueError("Invalid listing index row")
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Invalid listing index row") from exc
                    accession = match.group(2)
                    record = filings.setdefault(
                        accession,
                        {"accession": accession, "form": form, "filing_date": filed, "parties": []},
                    )
                    if record["form"] != form or record["filing_date"] != filed:
                        raise ValueError("Conflicting accession metadata")
                    party = {"cik": f"{int(cik):010d}", "name": name, "filename": filename}
                    if any(existing["cik"] == party["cik"] for existing in record["parties"]):
                        raise ValueError("Duplicate indexed party")
                    record["parties"].append(party)
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid master-index archive") from exc
    return [filings[key] for key in sorted(filings)]
