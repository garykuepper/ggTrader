import pandas as pd
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.config import load_symbols_from_json


def load_data_and_setup(config: dict) -> pd.DataFrame:
    """
    Loads symbols and fetches 4h OHLCV data from KrakenHistoricalData.

    Args:
        config (dict): Configuration dictionary containing:
            - 'SYMBOLS': List of symbol strings (optional).
            - 'SYMBOLS_FILE': Path to JSON file with symbols (optional).
            - 'INTERVAL': Time interval (e.g., '4h').
            - 'START_DATE': Start date string.
            - 'END_DATE': End date string.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data for the requested symbols.
    """
    # Priority: Direct SYMBOLS list -> SYMBOLS_FILE JSON
    if "SYMBOLS" in config and config["SYMBOLS"]:
        symbols = config["SYMBOLS"]
    elif "SYMBOLS_FILE" in config and config["SYMBOLS_FILE"]:
        symbols = load_symbols_from_json(config["SYMBOLS_FILE"])
        if symbols is None:
            raise ValueError(
                f"Symbols file '{config['SYMBOLS_FILE']}' not found or invalid."
            )
    else:
        raise ValueError(
            "Config must contain 'SYMBOLS' (list) or 'SYMBOLS_FILE' (path)."
        )

    if not symbols:
        raise ValueError("No symbols provided or symbols list is empty.")

    k_h = KrakenHistoricalData()

    ohlcv_df = k_h.get_ohlcv_df(
        symbols,
        interval=config["INTERVAL"],
        start=pd.to_datetime(config["START_DATE"]).tz_localize("UTC"),
        end=pd.to_datetime(config["END_DATE"]).tz_localize("UTC"),
    )

    if ohlcv_df.empty:
        raise ValueError(
            f"No OHLCV data returned from database.\n"
            f"  Symbols requested: {symbols[:5]}{'...' if len(symbols) > 5 else ''}\n"
            f"  Date range: {config['START_DATE']} to {config['END_DATE']}\n"
            f"  Interval: {config['INTERVAL']}\n"
            f"  Check that the database is running and contains data for these symbols."
        )

    return ohlcv_df


def build_mover_mask(
    ohlcv_df: pd.DataFrame,
    config: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a boolean mask of daily top-N movers aligned to the OHLCV index.

    The DB is queried once for daily rankings, then forward-filled to match
    the intraday frequency of the OHLCV data (e.g. 4h bars).
    """
    start = pd.to_datetime(config["START_DATE"]).tz_localize("UTC")
    end = pd.to_datetime(config["END_DATE"]).tz_localize("UTC")

    k_h = KrakenHistoricalData()
    daily_mask = k_h.get_daily_mover_mask(start=start, end=end, top_n=top_n)

    if daily_mask.empty:
        raise ValueError("No mover data returned from the database.")

    # Align symbols: only keep columns present in the OHLCV data
    ohlcv_symbols = ohlcv_df.columns.get_level_values(0).unique()
    common = daily_mask.columns.intersection(ohlcv_symbols)
    daily_mask = daily_mask[common]

    # Reindex to the intraday OHLCV index and forward-fill within each day
    mask = daily_mask.reindex(ohlcv_df.index, method="ffill").fillna(False)
    return mask.astype(bool)
