"""Read an HTTP response body with a hard byte limit, stopping early instead of buffering it."""

from __future__ import annotations

from typing import Any

#: Bytes requested from the socket at a time.
CHUNK_BYTES = 64 * 1024


class DownloadTooLarge(ValueError):
    """The body is over the byte limit (stated by the host, or reached while reading)."""


def read_capped(response: Any, limit: int) -> bytes:
    """The body of ``response`` (a ``requests`` reply fetched with ``stream=True``).

    Raises :class:`DownloadTooLarge` as soon as a stated ``Content-Length`` or the bytes read
    exceed ``limit``, and closes the connection, so an oversized or endless body is never held
    in memory. A response without ``iter_content`` (a test double) falls back to ``.content``.
    """
    declared = str(response.headers.get("content-length") or "")
    if declared.isdigit() and int(declared) > limit:
        _close(response)
        raise DownloadTooLarge(f"the host states {declared} bytes, over the {limit} byte cap")
    iterate = getattr(response, "iter_content", None)
    if iterate is None:
        body = bytes(response.content)
        if len(body) > limit:
            raise DownloadTooLarge(f"the body is {len(body)} bytes, over the {limit} byte cap")
        return body
    parts: list[bytes] = []
    total = 0
    for chunk in iterate(chunk_size=CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            _close(response)
            raise DownloadTooLarge(f"the body passed the {limit} byte cap while downloading")
        parts.append(chunk)
    return b"".join(parts)


def _close(response: Any) -> None:
    close = getattr(response, "close", None)
    if close is not None:
        close()
