def _periods_per_year_from_interval(interval: str) -> int:
    # Handles "4h", "1h", "1d" and similar
    if interval.endswith("h"):
        hours = int(interval[:-1])
        per_day = 24 // max(1, hours)
        return per_day * 365
    if interval.endswith("d"):
        days = int(interval[:-1])
        per_day = 1 // max(1, days) if days > 0 else 1
        return per_day * 365
    # Fallbacks for your common choices
    mapping = {"4h": 6 * 365, "1h": 24 * 365, "1d": 365}
    return mapping.get(interval, 6 * 365)
