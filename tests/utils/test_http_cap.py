"""``read_capped``: a hard byte limit applied while the body streams, not after it is held."""

from __future__ import annotations

import pytest

from ggTrader.utils.http_cap import CHUNK_BYTES, DownloadTooLarge, read_capped
from tests.data.test_membership_upstream import FakeResponse


def test_a_body_within_the_limit_comes_back_whole():
    body = b"abc" * 100_000
    assert read_capped(FakeResponse(200, body, {}), len(body)) == body


def test_an_endless_body_stops_at_the_first_chunk_past_the_limit():
    response = FakeResponse(200, b"x" * (10 * CHUNK_BYTES), {})
    with pytest.raises(DownloadTooLarge, match="passed the"):
        read_capped(response, CHUNK_BYTES + 5)
    assert response.served == 2 * CHUNK_BYTES  # read two chunks, not ten
    assert response.closed


def test_a_stated_length_over_the_limit_is_refused_before_any_byte_is_read():
    response = FakeResponse(200, b"x" * (10 * CHUNK_BYTES), {"content-length": "99999999"})
    with pytest.raises(DownloadTooLarge, match="states 99999999"):
        read_capped(response, 1000)
    assert response.served == 0 and response.closed
