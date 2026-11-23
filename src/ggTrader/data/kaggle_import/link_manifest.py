"""Verify the link artifact before it is used to publish a release."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ggTrader.data.kaggle_import.links import APPROVED_LINK_SETTINGS, LinkSettings


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def settings_dict(settings: LinkSettings) -> dict:
    """Represent the link settings consistently in JSON."""
    result = asdict(settings)
    for name in ("window_start", "window_end"):
        result[name] = result[name].isoformat()
    return result


def verify_link_manifest(csv_path: Path, raw_sha256: str) -> dict:
    """Require approved settings and matching links and source bytes."""
    manifest_path = csv_path.with_name(csv_path.name + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Link manifest missing; create it with scripts/kaggle_find_links.py")
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings_dict(APPROVED_LINK_SETTINGS):
            raise ValueError("Link manifest settings are not approved")
        if manifest.get("sample_permnos") is not None:
            raise ValueError("Sampled link tables cannot be published")
        if manifest["csv_sha256"] != file_sha256(csv_path):
            raise ValueError("Link CSV does not match its manifest")
        if manifest["crsp_file_sha256"] != raw_sha256:
            raise ValueError("Link manifest CRSP source does not match the derived load")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid link manifest") from exc
