"""Read structured fields from an exchange-filed Form 25 notice."""

import hashlib
import re
from datetime import date
from html.parser import HTMLParser
from xml.etree import ElementTree

_MAX_XML_BYTES = 1024 * 1024


class _IndexLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.paths.extend(
                value for name, value in attrs if name == "href" and value is not None
            )


def parse_bound_exchange_notice(
    index_content: bytes,
    document_content: bytes,
    *,
    expected_accession: str,
    expected_cik: int,
    expected_index_sha256: str,
    expected_document_sha256: str,
) -> dict:
    """Check frozen hashes and the index's raw-XML path before parsing the notice.

    The caller must independently establish that both byte strings came from the
    SEC URLs and that the expected hashes belong to the requested filing. The SEC
    index does not itself hash the XML, so this check cannot prove that association,
    public availability, class identity or effective trading date. Only observed
    SEC root-relative primary_doc.xml paths are supported; ambiguity fails closed.
    """
    if not isinstance(expected_accession, str) or not re.fullmatch(
        r"[0-9]{10}-[0-9]{2}-[0-9]{6}", expected_accession
    ):
        raise ValueError("Invalid accession")
    for content, expected in (
        (index_content, expected_index_sha256),
        (document_content, expected_document_sha256),
    ):
        if (
            not isinstance(content, bytes)
            or len(content) > _MAX_XML_BYTES
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or hashlib.sha256(content).hexdigest() != expected
        ):
            raise ValueError("Invalid content or hash")
    try:
        text = index_content.decode("utf-8-sig")
        parser = _IndexLinks()
        parser.feed(text)
        parser.close()
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid index") from exc
    paths = [
        path
        for path in parser.paths
        if re.fullmatch(r"/Archives/edgar/data/[0-9]{1,10}/[0-9]{18}/primary_doc\.xml", path)
    ]
    compact = expected_accession.replace("-", "")
    if len(paths) != 1 or paths[0].split("/")[-2] != compact or expected_accession not in text:
        raise ValueError("Invalid or ambiguous accession in filing index")
    return {
        "accession": expected_accession,
        "document_path": paths[0],
        "index_sha256": expected_index_sha256,
        "document_sha256": expected_document_sha256,
        "notice": parse_exchange_notice(document_content, expected_cik=expected_cik),
    }


def parse_exchange_notice(content: bytes, *, expected_cik: int) -> dict:
    """Extract reported fields, without classifying securities or inferring event timing.

    Caller must verify source, accession, public availability and document meaning.
    The signature date is not a verified publication or effective date. This accepts
    only observed X0202 and X0203 exchange-filed XML, not issuer-filed HTML Form 25s.
    """
    if type(expected_cik) is not int or not 0 < expected_cik < 10**10:
        raise ValueError("Invalid expected issuer CIK")
    if not isinstance(content, bytes) or len(content) > _MAX_XML_BYTES:
        raise ValueError("Invalid XML content")
    try:
        text = content.decode("utf-8-sig")
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError("Invalid XML content")
        root = ElementTree.fromstring(text)
    except (ElementTree.ParseError, UnicodeError) as exc:
        raise ValueError("Invalid XML content") from exc
    if root.tag != "notificationOfRemoval":
        raise ValueError("Invalid notice root")

    def field(path: str) -> str:
        nodes = root.findall(path)
        if len(nodes) != 1 or len(nodes[0]) or not nodes[0].text or not nodes[0].text.strip():
            raise ValueError(f"Invalid {path}")
        return nodes[0].text.strip()

    if field("schemaVersion") not in {"X0202", "X0203"}:
        raise ValueError("Unsupported notice schema")
    issuer = field("issuer/cik")
    exchange = field("exchange/cik")
    if not re.fullmatch(r"[0-9]{1,10}", issuer) or int(issuer) != expected_cik:
        raise ValueError("Invalid or mismatched issuer CIK")
    if not re.fullmatch(r"[0-9]{1,10}", exchange) or int(exchange) == 0:
        raise ValueError("Invalid exchange CIK")
    signature = field("signatureData/signatureDate")
    try:
        if date.fromisoformat(signature).isoformat() != signature:
            raise ValueError("Invalid signature date")
    except ValueError as exc:
        raise ValueError("Invalid signature date") from exc
    return {
        "issuer_cik": f"{int(issuer):010d}",
        "issuer_name": field("issuer/entityName"),
        "exchange_cik": f"{int(exchange):010d}",
        "exchange_name": field("exchange/entityName"),
        "security_description": field("descriptionClassSecurity"),
        "rule_provision": field("ruleProvision"),
        "signature_date": signature,
    }
