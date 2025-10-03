# ggTrader

A lightweight Python toolset for generating and analyzing synthetic OHLCV data and computing technical signals.

## Purpose

- Provide a quick way to generate fake OHLCV data for testing and development.
- Compute a set of technical indicators and signals on top of OHLCV data.
- Serve as a foundation for experimentation with trading signals and backtesting.

## Key Components

- Signals.py
  - Generates fake data (OHLCV) for testing.
  - Builds a set of technical indicators (EMA, MACD, ATR, PSAR, ADX) and derived signals.
  - Exposes a simple workflow to produce signals for a given dataset.

> Note: The project uses Pandas, NumPy, and the `ta` library for technical indicators.

## Setup

1. Create and activate a virtual environment
   - python -m venv .venv
   - On Windows: .\.venv\Scripts\activate
   - On macOS/Linux: source .venv/bin/activate

2. Install dependencies
   - pip install -r requirements.txt

3. Verify installation
   - python -c "import pandas as pd; import numpy as np; from ta.trend import EMAIndicator"

## How to Use

- Generate a fake OHLCV DataFrame
  - Use Signals.generate_fake_data(rows=200, seed=42, start=100.0, drift=0.5, vol=2.0)
  - This returns a DataFrame with columns: close, high, low and a date index (daily by default)

- Compute signals on a DataFrame
  - Create an instance of Signals and call compute(df)
  - Result is a DataFrame with computed signal columns such as ema_fast, ema_slow, macd, atr, psar, adx, etc.

- Quick run (example)
  - from ggTrader.Signals import Signals
  - signals = Signals()
  - df = signals.generate_fake_data(rows=200)
  - out = signals.compute(df)
  - print(out.tail())

## Project Structure (high level)

- ggTrader/
  - Signals.py        # Main signal generation and fake data utilities
  - Portfolio.py      # (if present) portfolio-related logic
  - Position.py         # (if present) position tracking logic
  - Utils.py            # (if present) utility helpers
  - __init__.py
- data/                # sample data files (e.g., Kraken data)
- notes/
  - mongodb_notes
  - opt params
  - top_crypto
- tests/               # test suite
- readme.md            # this file
- requirements.txt
- scratch.py
- other utilities and scripts (as needed)

## Dependencies

- Python 3.13+ (project uses Python 3.13.5 in the environment)
- pandas
- numpy
- ta (technical analysis library)
- tabulate (for pretty printing in console)

## Configuration Tips

- Timezone handling:
  - If you generate dates, prefer a standard timezone like UTC to avoid cross-system issues.
  - Ensure any tz-aware timestamps are consistently handled across your data processing steps.

- Reproducibility:
  - Use the seed parameter in fake data generation to obtain deterministic results for testing.

## Testing and Validation

- Basic checks:
  - Generated DataFrame has the expected columns: close, high, low (and index as dates).
  - Computed signals DataFrame aligns with input dates (same index, with extra signal columns).
- Quick sanity tests:
  - df = Signals.generate_fake_data(rows=50, seed=1)
  - result = Signals().compute(df)
  - result.head()
  - result.tail()

## Extending the Project

- Add more indicators or signals by extending the _build_signals method.
- Make the data generator support multiple frequencies (e.g., hourly) and handle weekends/holidays as needed.
- Add unit tests around data generation, indicator outputs, and edge cases (empty DataFrames, NaNs, etc.).

## Contributing

- Fork the repository, create a feature or bugfix branch, and submit a pull request.
- Prefer small, well-documented changes with corresponding tests.

## License

- This project is provided as-is for development and experimentation purposes. (Add your preferred license here.)

## Contact

- If you have questions or feature requests, open an issue in the repository and provide a minimal reproducible example.