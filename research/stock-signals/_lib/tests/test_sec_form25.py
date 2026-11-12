"""Test conservative extraction from exchange-filed Form 25 notice structures."""

import hashlib
from pathlib import Path

import pytest
import sec_form25

pytestmark = pytest.mark.study

XML = b"""<notificationOfRemoval>
<schemaVersion>X0203</schemaVersion>
<exchange><cik>0000876661</cik><entityName>NEW YORK STOCK EXCHANGE LLC</entityName></exchange>
<issuer><cik>0000051143</cik><entityName>EXAMPLE CORP</entityName></issuer>
<descriptionClassSecurity>2.875% Notes Due 2025</descriptionClassSecurity>
<ruleProvision>17 CFR 240.12d2-2(a)(2)</ruleProvision>
<signatureData><signatureDate>2025-11-07</signatureDate></signatureData>
</notificationOfRemoval>"""


@pytest.mark.parametrize(
    ("accession", "sha256", "cik", "description"),
    [
        (
            "000087666125000849",
            "97560a300cd5a0757d3af50848d45758016c3159f68c98904826aed8532640f4",
            51143,
            "2.875% Notes Due November 7, 2025",
        ),
        (
            "000087666117000796",
            "3de92fd39cc666ad67f3b9149e0f1c6c6d97777c6e80493ebd1fac3a6d17d4af",
            1222840,
            "7.00% Series A Cumulative Redeemable Preferred Stock",
        ),
        (
            "000087666121001526",
            "f16f2bf5d310757c1d2447fbee40d66f68049123c0d8c647a1f08dd96c1f0719",
            1222840,
            "Class A Common Stock",
        ),
        (
            "000135445721000003",
            "81e2b6a2010928679aa5044c3f4d993a40ff07bca98399cca4a585512bc54c04",
            1574815,
            "Common Stock",
        ),
        (
            "000135445721000004",
            "5eec4c6edc1a1c4eae267e255f257eda213c21b47670d6f7b8c0f2ebdfb32929",
            1578735,
            "Common Stock",
        ),
        (
            "000135445721000016",
            "b7b3807d2485777e7514a0e22f37d4378f2f88367009fc0f1309f96baca5c904",
            1143513,
            "6.125% Notes due 2023",
        ),
        (
            "000135445721000027",
            "d740c63f85582cf877539a539612fc530928df4b3994dee5f1f37d2857edc20d",
            1721386,
            "LF Capital Acquistion Corp. Unit",
        ),
        (
            "000135445721000045",
            "9c6df75c9baf0fce90d5cd25853996d7b96b3844c2bac59bf46116e29c1cb86a",
            1774155,
            "South Mountain Merger Corp. Unit",
        ),
    ],
)
def test_original_sec_notice_bytes_preserve_security_class(accession, sha256, cik, description):
    content = (Path(__file__).parent / "fixtures" / "sec_form25" / f"{accession}.xml").read_bytes()
    assert hashlib.sha256(content).hexdigest() == sha256
    assert (
        sec_form25.parse_exchange_notice(content, expected_cik=cik)["security_description"]
        == description
    )


def test_index_and_frozen_document_are_bound_to_the_requested_accession():
    fixtures = Path(__file__).parent / "fixtures" / "sec_form25"
    index = (fixtures / "000087666121001526_index.html").read_bytes()
    common = (fixtures / "000087666121001526.xml").read_bytes()
    preferred = (fixtures / "000087666117000796.xml").read_bytes()
    options = {
        "expected_accession": "0000876661-21-001526",
        "expected_cik": 1222840,
        "expected_index_sha256": "fca3c4b8c9183534ab9372bea4d6f49a73312eeadc45f0e455cd42a6a513ed51",
        "expected_document_sha256": (
            "f16f2bf5d310757c1d2447fbee40d66f68049123c0d8c647a1f08dd96c1f0719"
        ),
    }
    result = sec_form25.parse_bound_exchange_notice(index, common, **options)
    assert (
        result["document_path"] == "/Archives/edgar/data/876661/000087666121001526/primary_doc.xml"
    )
    assert result["notice"]["security_description"] == "Class A Common Stock"
    with pytest.raises(ValueError, match="hash"):
        sec_form25.parse_bound_exchange_notice(index, preferred, **options)
    with pytest.raises(ValueError, match="hash"):
        sec_form25.parse_bound_exchange_notice(index + b" ", common, **options)
    with pytest.raises(ValueError, match="accession"):
        sec_form25.parse_bound_exchange_notice(
            index, common, **{**options, "expected_accession": "0000876661-17-000796"}
        )
    with pytest.raises(ValueError, match="issuer"):
        sec_form25.parse_bound_exchange_notice(index, common, **{**options, "expected_cik": 51143})

    path = b"/Archives/edgar/data/876661/000087666121001526/primary_doc.xml"
    for changed_index in (
        index.replace(path, path + b"?unsafe=1"),
        index + b'<a href="' + path + b'">duplicate</a>',
    ):
        with pytest.raises(ValueError, match="accession"):
            sec_form25.parse_bound_exchange_notice(
                changed_index,
                common,
                **{**options, "expected_index_sha256": hashlib.sha256(changed_index).hexdigest()},
            )

    preferred_index = (fixtures / "000087666117000796_index.html").read_bytes()
    preferred_options = {
        **options,
        "expected_accession": "0000876661-17-000796",
        "expected_index_sha256": "967fb472b8e9eef54765a081eeac322af053b48695f8d131696e21a3af92ffb6",
        "expected_document_sha256": (
            "3de92fd39cc666ad67f3b9149e0f1c6c6d97777c6e80493ebd1fac3a6d17d4af"
        ),
    }
    result = sec_form25.parse_bound_exchange_notice(preferred_index, preferred, **preferred_options)
    assert "Preferred Stock" in result["notice"]["security_description"]


def test_index_path_can_use_issuer_cik_not_accession_prefix():
    fixtures = Path(__file__).parent / "fixtures" / "sec_form25"
    result = sec_form25.parse_bound_exchange_notice(
        (fixtures / "000135445721000016_index.html").read_bytes(),
        (fixtures / "000135445721000016.xml").read_bytes(),
        expected_accession="0001354457-21-000016",
        expected_cik=1143513,
        expected_index_sha256="3ab87cd807b77a6821a8f083bf00dd53204110345e0e75f36eef1572a03442ba",
        expected_document_sha256=(
            "b7b3807d2485777e7514a0e22f37d4378f2f88367009fc0f1309f96baca5c904"
        ),
    )
    assert (
        result["document_path"] == "/Archives/edgar/data/1143513/000135445721000016/primary_doc.xml"
    )
    assert result["notice"]["security_description"] == "6.125% Notes due 2023"


def test_extracts_class_without_classifying_or_inferring_event_date():
    result = sec_form25.parse_exchange_notice(XML, expected_cik=51143)
    assert result == {
        "issuer_cik": "0000051143",
        "issuer_name": "EXAMPLE CORP",
        "exchange_cik": "0000876661",
        "exchange_name": "NEW YORK STOCK EXCHANGE LLC",
        "security_description": "2.875% Notes Due 2025",
        "rule_provision": "17 CFR 240.12d2-2(a)(2)",
        "signature_date": "2025-11-07",
    }


def test_accepts_older_observed_schema_and_unpadded_exchange_cik():
    older = XML.replace(b"X0203", b"X0202").replace(b"0000876661", b"876661")
    result = sec_form25.parse_exchange_notice(older, expected_cik=51143)
    assert result["exchange_cik"] == "0000876661"
    assert result["security_description"] == "2.875% Notes Due 2025"


def test_extracts_fields_from_transcribed_sec_indexed_structure():
    # Transcribed structure, not original XML bytes: SEC accession 0000876661-06-000234.
    content = b"""<notificationOfRemoval>
    <schemaVersion>X0202</schemaVersion>
    <exchange><cik>876661</cik><entityName>NEW YORK STOCK EXCHANGE INC</entityName></exchange>
    <issuer><cik>0001030561</cik><entityName>FAIRMONT HOTELS &amp; RESORTS INC</entityName>
    <fileNumber>001-14574</fileNumber><address><city>TORONTO ONTARIO</city></address></issuer>
    <descriptionClassSecurity>Common Shares, Common Share Purchase Rights</descriptionClassSecurity>
    <ruleProvision>17 CFR 240.12d2-2(a)(3)</ruleProvision>
    <signatureData><signatureName>Paras Madho</signatureName>
    <signatureTitle>Director</signatureTitle>
    <signatureDate>2006-05-19</signatureDate></signatureData></notificationOfRemoval>"""
    result = sec_form25.parse_exchange_notice(content, expected_cik=1030561)
    assert result["issuer_name"] == "FAIRMONT HOTELS & RESORTS INC"
    assert result["exchange_cik"] == "0000876661"
    assert result["security_description"] == "Common Shares, Common Share Purchase Rights"
    assert result["signature_date"] == "2006-05-19"


def test_extracts_transcribed_nasdaq_notice():
    # Transcribed structure, not original XML bytes: SEC accession 0001354457-26-000831.
    content = b"""<notificationOfRemoval>
    <schemaVersion>X0203</schemaVersion>
    <exchange><cik>0001354457</cik><entityName>Nasdaq Stock Market LLC</entityName></exchange>
    <issuer><cik>0000008146</cik><entityName>AstroNova, Inc.</entityName>
    <fileNumber>000-13200</fileNumber><address><city>West Warwick</city></address></issuer>
    <descriptionClassSecurity>Common Stock</descriptionClassSecurity>
    <ruleProvision>17 CFR 240.12d2-2(a)(3)</ruleProvision>
    <signatureData><signatureName>Katelin Rowe</signatureName>
    <signatureTitle>CDO Analyst</signatureTitle>
    <signatureDate>2026-08-27</signatureDate></signatureData></notificationOfRemoval>"""
    result = sec_form25.parse_exchange_notice(content, expected_cik=8146)
    assert result["issuer_name"] == "AstroNova, Inc."
    assert result["exchange_cik"] == "0001354457"
    assert result["security_description"] == "Common Stock"
    assert result["signature_date"] == "2026-08-27"


def test_refuses_unknown_schema():
    with pytest.raises(ValueError, match="schema"):
        sec_form25.parse_exchange_notice(XML.replace(b"X0203", b"X9999"), expected_cik=51143)


def test_wrong_issuer_and_duplicate_class_are_rejected():
    with pytest.raises(ValueError, match="issuer"):
        sec_form25.parse_exchange_notice(XML, expected_cik=999)
    extra = XML.replace(
        b"</notificationOfRemoval>",
        b"<descriptionClassSecurity>Common</descriptionClassSecurity></notificationOfRemoval>",
    )
    with pytest.raises(ValueError, match="descriptionClassSecurity"):
        sec_form25.parse_exchange_notice(extra, expected_cik=51143)


@pytest.mark.parametrize(
    "content", [b"<bad>", b"<!DOCTYPE x [<!ENTITY y 'z'>]><x/>", b"x" * (1024 * 1024 + 1)]
)
def test_refuses_malformed_or_unsafe_content(content):
    with pytest.raises(ValueError):
        sec_form25.parse_exchange_notice(content, expected_cik=51143)


def test_rejects_non_utf8_xml_with_hidden_dtd():
    content = (
        "<?xml version='1.0' encoding='UTF-16'?><!DOCTYPE notificationOfRemoval [<!ENTITY y 'z'>]>"
        + XML.decode()
    ).encode("utf-16")
    with pytest.raises(ValueError):
        sec_form25.parse_exchange_notice(content, expected_cik=51143)
