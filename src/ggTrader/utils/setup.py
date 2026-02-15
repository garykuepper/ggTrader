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
    else:
        raise ValueError(
            "Config must contain 'SYMBOLS' (list) or 'SYMBOLS_FILE' (path)."
        )

    k_h = KrakenHistoricalData()

    ohlcv_df = k_h.get_ohlcv_df(
        symbols,
        interval=config["INTERVAL"],
        start=pd.to_datetime(config["START_DATE"]).tz_localize("UTC"),
        end=pd.to_datetime(config["END_DATE"]).tz_localize("UTC"),
    )

    return ohlcv_df
