"""Compare listing coverage and daily closing prices."""

import pandas as pd


def return_agreement(
    old: pd.DataFrame, new: pd.DataFrame, ref_old: pd.DataFrame, ref_new: pd.DataFrame, tol: float
) -> tuple[int, float | None]:
    """Compare same-source daily returns on the intersection of four price sets."""
    frames = [f.set_index("symbol")["close"] for f in (old, new, ref_old, ref_new)]
    values = pd.concat(frames, axis=1, join="inner").dropna()
    values = values[(values.iloc[:, 0] > 0) & (values.iloc[:, 2] > 0)]
    if values.empty:
        return 0, None
    delta = (values.iloc[:, 1] / values.iloc[:, 0] - values.iloc[:, 3] / values.iloc[:, 2]).abs()
    return len(values), float((delta <= tol).mean())


def classify_security(symbol: str, name: str) -> str:
    """Classify explicit security descriptions; leave ambiguous cases unclassified."""
    import re

    text = str(name).lower()
    for pattern, label in [
        (r"\bwarrants?\b", "warrants"),
        (r"\brights?\b", "rights"),
        (r"\bunits?\b", "units"),
        (r"\bpreferred\b", "preferred"),
        (r"\bcommon (stock|shares?)\b", "common"),
    ]:
        if re.search(pattern, text):
            return label
    return "other"


def filter_listing(listed: pd.DataFrame) -> pd.DataFrame:
    """Keep ordinary, non-test listings."""
    return listed.loc[~listed["is_etf"] & ~listed["is_test"]].copy()


def coverage(listed: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Count listed symbols with at least one returned bar."""
    symbols = set(listed["symbol"])
    returned = len(symbols & set(bars["symbol"]))
    return {
        "listed": len(symbols),
        "returned": returned,
        "missing": len(symbols) - returned,
        "share": returned / len(symbols) if symbols else 0.0,
    }


def price_agreement(bars: pd.DataFrame, reference: pd.DataFrame, tol: float) -> float:
    """Return the fraction of matched closes within relative tolerance."""
    matched = bars.merge(reference, on="symbol", suffixes=("_bar", "_reference"))
    if matched.empty:
        return 0.0
    return float(
        (
            (matched["close_bar"] - matched["close_reference"]).abs()
            <= tol * matched["close_reference"].abs()
        ).mean()
    )


def read_directory(url: str) -> object:
    """Download one public symbol directory."""
    import io

    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return io.StringIO(response.text)


def fetch_listing() -> pd.DataFrame:
    """Fetch and filter the two Nasdaq Trader listing files."""
    root = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    frames = []
    for filename, symbol in [("nasdaqlisted.txt", "Symbol"), ("otherlisted.txt", "ACT Symbol")]:
        raw = pd.read_csv(read_directory(root + filename), sep="|", dtype=str)
        raw = raw.loc[~raw[symbol].fillna("").str.startswith("File Creation Time")].copy()
        frame = pd.DataFrame(
            {
                "symbol": raw[symbol],
                "name": raw["Security Name"],
                "exchange": raw.get("Exchange", "Q"),
                "is_etf": raw["ETF"].eq("Y"),
                "is_test": raw["Test Issue"].eq("Y"),
            }
        )
        frames.append(frame)
    return (
        filter_listing(pd.concat(frames, ignore_index=True))
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol")
    )


def fetch_bars(
    symbols: list[str], day: object, source: str = "yahoo", feed: str = "iex"
) -> pd.DataFrame:
    """Fetch daily prices from Yahoo or the free Alpaca IEX feed."""
    import datetime
    import os
    import time

    import requests
    import yfinance as yf

    columns = ["symbol", "open", "high", "low", "close", "volume"]
    rows = []
    if source == "yahoo":
        for offset in range(0, len(symbols), 100):
            batch = symbols[offset : offset + 100]
            data = yf.download(
                batch,
                start=str(day),
                end=str(day + datetime.timedelta(days=1)),
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=15,
            )
            if data.empty:
                continue
            for symbol in batch:
                try:
                    item = data.xs(symbol, axis=1, level=1).iloc[0]
                    if pd.notna(item["Close"]):
                        rows.append(
                            [
                                symbol,
                                *[
                                    item[name]
                                    for name in ["Open", "High", "Low", "Close", "Volume"]
                                ],
                            ]
                        )
                except (KeyError, IndexError):
                    continue
    elif source == "alpaca":
        headers = {
            "APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID", ""),
            "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY", ""),
        }
        batches = [symbols[offset : offset + 200] for offset in range(0, len(symbols), 200)]
        while batches:
            batch = batches.pop(0)
            params = {
                "symbols": ",".join(batch),
                "timeframe": "1Day",
                "start": str(day),
                "end": str(day + datetime.timedelta(days=1)),
                "feed": feed,
                "limit": 10000,
                "adjustment": "raw",
            }
            while True:
                for attempt in range(5):
                    time.sleep(0.5)
                    response = requests.get(
                        "https://data.alpaca.markets/v2/stocks/bars",
                        params=params,
                        headers=headers,
                        timeout=60,
                    )
                    if response.status_code != 429 or attempt == 4:
                        break
                    retry = response.headers.get("Retry-After", "")
                    time.sleep(float(retry) if retry.isdecimal() else 10 * 2**attempt)
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 400:
                        if len(batch) > 1:
                            middle = len(batch) // 2
                            batches[0:0] = [batch[:middle], batch[middle:]]
                        break
                    raise
                payload = response.json()
                for symbol, bars in (payload.get("bars") or {}).items():
                    for bar in bars:
                        if bar["t"][:10] == str(day):
                            rows.append([symbol, *[bar[key] for key in ["o", "h", "l", "c", "v"]]])
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
    else:
        raise ValueError("Unknown price source")
    return pd.DataFrame(rows, columns=columns).drop_duplicates("symbol")


def main() -> None:
    """Run the read-only source pilot and save its evidence."""
    import argparse
    import datetime
    import json
    import logging

    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    from ggTrader.utils.config import get_db_connection_string
    from ggTrader.utils.study import bootstrap, study_dir

    root = bootstrap(__file__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day", type=datetime.date.fromisoformat, default=datetime.date(2026, 11, 25)
    )
    parser.add_argument(
        "--comparison-day", type=datetime.date.fromisoformat, default=datetime.date(2026, 11, 4)
    )
    parser.add_argument("--alpaca-only", action="store_true", help="Retry Alpaca only")
    parser.add_argument("--resume", action="store_true", help="Preserve the prior live Yahoo count")
    args = parser.parse_args()
    load_dotenv(root / ".env")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    output = study_dir(__file__) / "_source_pilot.json"
    report = {
        "provenance": {
            "live_day": str(args.day),
            "comparison_day": str(args.comparison_day),
            "listing": "Nasdaq Trader, fetched at run time",
            "alpaca_feed": "IEX free",
            "yahoo_adjustment": "raw",
            "reference": "market.yahoo_daily_frozen_20261126",
            "tolerance_fraction": 0.01,
            "delisted_sample": ["SIVB", "FRC", "BBBY", "TWTR", "ATVI"],
            "delisted_day": "2022-11-04",
            "alpaca_400_handling": (
                "Split rejected batches; rejected individual symbols count as missing"
            ),
            "reference_prices": "Frozen Yahoo adjusted closes; source requests raw closes",
            "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "sources": {},
        "errors": {},
    }

    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        report["previous_yahoo_live"] = previous.get("sources", {}).get("yahoo", {}).get("live")
        if args.alpaca_only:
            report["sources"]["yahoo"] = previous.get("sources", {}).get("yahoo", {})
            report["shares_outstanding"] = previous.get("shares_outstanding")

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    listing = fetch_listing()
    symbols = listing["symbol"].tolist()
    report["listing_count"] = len(symbols)
    save()
    reference = pd.DataFrame(columns=["symbol", "close"])
    try:
        engine = create_engine(get_db_connection_string())
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            cols = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='market' AND "
                    "table_name='yahoo_daily_frozen_20261126'"
                )
            )
            names = [row[0] for row in cols]
            report["provenance"]["reference_columns"] = names
            reference = pd.read_sql(
                text(
                    "SELECT symbol, close FROM market.yahoo_daily_frozen_20261126 "
                    "WHERE venue = 'yfinance' AND interval = '1d' "
                    "AND timestamp >= :day AND timestamp < :next_day"
                ),
                conn,
                params={
                    "day": args.comparison_day,
                    "next_day": args.comparison_day + datetime.timedelta(days=1),
                },
            )
    except Exception as exc:
        report["errors"]["reference"] = type(exc).__name__
    save()
    for source in ["alpaca"] if args.alpaca_only else ["alpaca", "yahoo"]:
        report["sources"][source] = {}
        for label, day, requested in [
            ("live", args.day, symbols),
            ("comparison", args.comparison_day, reference["symbol"].tolist()),
            ("delisted", datetime.date(2022, 11, 4), report["provenance"]["delisted_sample"]),
        ]:
            if args.resume and source == "yahoo" and label == "live":
                report["sources"][source][label] = report.get("previous_yahoo_live")
                continue
            if not requested:
                report["sources"][source][label] = {"error": "No symbols available"}
                continue
            try:
                bars = fetch_bars(requested, day, source)
                entry = coverage(pd.DataFrame({"symbol": requested}), bars)
                if label == "comparison":
                    entry["matched"] = len(set(bars["symbol"]) & set(reference["symbol"]))
                    entry["within_1_percent"] = (
                        price_agreement(bars, reference, 0.01) if entry["matched"] else None
                    )
                if label == "delisted":
                    entry["symbols_returned"] = bars["symbol"].tolist()
                report["sources"][source][label] = entry
            except Exception as exc:
                report["sources"][source][label] = {
                    "error": type(exc).__name__,
                    "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                }
            save()
    if args.alpaca_only:
        return
    import yfinance as yf

    yf.config.debug.hide_exceptions = False
    try:
        shares = yf.Ticker("AAPL").get_shares_full(start="2026-01-01")
        report["shares_outstanding"] = {
            "source": "Yahoo via yfinance",
            "sample": "AAPL",
            "returned": shares is not None and not shares.empty,
        }
    except Exception as exc:
        report["shares_outstanding"] = {"error": type(exc).__name__}
    save()


if __name__ == "__main__":
    main()
