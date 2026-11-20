"""Count missing SPY regular-session 1-minute SIP bars on Alpaca, 2016 to a pinned end date.

Records timestamps and counts only. No price, volume-weighted price or return is kept.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DATA = "https://data.alpaca.markets/v2/stocks/SPY/bars"
CAL = "https://paper-api.alpaca.markets/v2/calendar"
OUT = Path(__file__).with_name("spy_minute_gap_count.json")
FIRST_HALF = [f"09:{m:02d}" for m in range(30, 60)]


def read_env(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines from a dotenv file."""
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get(url: str, params: dict, headers: dict) -> dict:
    """GET JSON with bounded retries on throttling and server errors."""
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            req = urllib.request.Request(full, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    """Fetch the calendar and bar timestamps, then write per-day gap counts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--feed", default="sip")
    args = ap.parse_args()

    env = read_env(Path(__file__).resolve().parents[3] / ".env")
    headers = {
        "APCA-API-KEY-ID": env["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": env["APCA_API_SECRET_KEY"],
    }
    cal = get(CAL, {"start": args.start, "end": args.end}, headers)
    sessions = {}
    for c in cal:
        o = datetime.strptime(c["date"] + c["open"], "%Y-%m-%d%H:%M")
        cl = datetime.strptime(c["date"] + c["close"], "%Y-%m-%d%H:%M")
        grid = []
        t = o
        while t < cl:
            grid.append(t.strftime("%H:%M"))
            t += timedelta(minutes=1)
        sessions[c["date"]] = grid

    seen: dict[str, set[str]] = {d: set() for d in sessions}
    out_of_grid = 0
    pages = 0
    months = sorted({d[:7] for d in sessions})
    for mo in months:
        y, m = map(int, mo.split("-"))
        nxt = f"{y + (m == 12)}-{(m % 12) + 1:02d}-01"
        # The free plan rejects SIP requests that reach the latest 15 minutes, so never
        # ask past the pinned end date.
        stop = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        nxt = min(nxt, stop)
        params = {
            "timeframe": "1Min", "start": f"{mo}-01T00:00:00Z", "end": f"{nxt}T00:00:00Z",
            "limit": 10000, "adjustment": "raw", "feed": args.feed, "sort": "asc",
        }
        token = None
        while True:
            if token:
                params["page_token"] = token
            else:
                params.pop("page_token", None)
            d = get(DATA, params, headers)
            pages += 1
            for b in d.get("bars") or []:
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
                day, hm = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")
                if day in seen and hm in sessions[day]:
                    seen[day].add(hm)
                else:
                    out_of_grid += 1
            token = d.get("next_page_token")
            if not token:
                break
        print(mo, "pages so far", pages, flush=True)

    days = []
    for day, grid in sessions.items():
        missing = [hm for hm in grid if hm not in seen[day]]
        last_half = grid[-30:]
        days.append({
            "date": day,
            "expected": len(grid),
            "missing": len(missing),
            "missing_first_half": [hm for hm in FIRST_HALF if hm not in seen[day]],
            "missing_last_half": [hm for hm in last_half if hm not in seen[day]],
            "early_close": len(grid) != 390,
        })
    rule_days = [x for x in days if x["missing_first_half"] or x["missing_last_half"]]
    summary = {
        "symbol": "SPY", "feed": args.feed, "timeframe": "1Min", "adjustment": "raw",
        "start": args.start, "end": args.end,
        "run_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sessions": len(days),
        "early_close_sessions": sum(x["early_close"] for x in days),
        "expected_bars": sum(x["expected"] for x in days),
        "missing_bars": sum(x["missing"] for x in days),
        "sessions_with_any_missing": sum(x["missing"] > 0 for x in days),
        "sessions_missing_a_rule_bar": len(rule_days),
        "sessions_with_no_bars": sum(x["missing"] == x["expected"] for x in days),
        "out_of_grid_bars": out_of_grid,
        "pages": pages,
        "note": "Timestamps and counts only; no prices, volumes or returns recorded.",
    }
    OUT.write_text(json.dumps({"summary": summary, "days": days}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
