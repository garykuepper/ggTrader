"""Shared forward-listing candidate rules."""

import re
from typing import Any


def common_candidate(row: dict[str, Any]) -> bool:
    """Select explicit common or ordinary share names, excluding other securities."""
    name = row["security_name"].lower()
    return (
        not row["is_etf"]
        and not row["is_test"]
        and not re.search(r"\b(warrants?|units?|rights?|preferred)\b", name)
        and bool(re.search(r"\b(common|ordinary) (stock|shares?)\b", name))
    )
