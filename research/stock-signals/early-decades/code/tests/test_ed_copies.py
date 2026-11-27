"""Check the frozen engine copies and their documented import substitutions."""

import hashlib
import json

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study
STUDY = study_dir(__file__)
CODE = STUDY / "code"
SOURCE = STUDY.parent / "dead-stock-momentum-deaths" / "code"


def test_ed_copies_match_sources_by_hash() -> None:
    """Copies differ from their source only by recorded substitutions."""
    record = json.loads((CODE / "frozen_copies.json").read_text())
    assert len(record) == 5
    for name, entry in record.items():
        original = (SOURCE / name).read_bytes()
        assert hashlib.sha256(original).hexdigest() == entry["source_sha256"]
        text = original.decode()
        for old, new in entry["substitutions"]:
            assert text.count(old) == 1
            text = text.replace(old, new)
        copied = (CODE / entry["copy"]).read_bytes()
        assert copied == text.encode()
        assert hashlib.sha256(copied).hexdigest() == entry["copy_sha256"]
