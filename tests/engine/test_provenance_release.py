"""Release identity is part of the data behind a result."""

from ggTrader.engine.provenance import compare, provenance

RELEASE = {"release_id": 1, "created_at": "2026-01-01", "yahoo_fingerprint": {"sha256": "a"}}


def test_different_release_is_tape_diff_not_comparable():
    a = provenance(tape_sha256="same", data_release=RELEASE)
    for b in (
        {**a, "data_release": {**RELEASE, "release_id": 2}},
        {k: v for k, v in a.items() if k != "data_release"},
    ):
        result = compare(a, b)
        assert result.flag == "TAPE_DIFF" and not result.comparable


def test_same_release_same_hash_is_comparable():
    a = provenance(tape_sha256="same", data_release=RELEASE)
    assert compare(a, dict(a)).comparable


def test_no_release_block_is_unchanged():
    block = provenance()
    assert set(block) == {
        "schema",
        "python",
        "pandas",
        "numpy",
        "numba",
        "llvmlite",
        "scipy",
        "vectorbt",
        "vectorbt_rust",
        "engine_requested",
        "engine_effective",
        "rust_calls",
        "lock_sha256",
        "git_sha",
        "git_dirty",
        "contract_version",
        "baseline_epoch",
        "definitions",
        "tape_sha256",
        "created_utc",
    }


def test_different_dead_stock_settings_are_not_comparable():
    a = provenance(tape_sha256="same", data_release=RELEASE, extra={"dead_stock_loss": 0.0})
    b = {**a, "dead_stock_loss": 1.0}
    for other in (
        b,
        {k: v for k, v in a.items() if k != "dead_stock_loss"},
        {**a, "dead_stock_rule": {}},
    ):
        comparison = compare(a, other)
        assert comparison.flag == "TAPE_DIFF" and not comparison.comparable
        assert "dead-stock setting differs" in comparison.note
    assert compare(a, dict(a)).comparable
