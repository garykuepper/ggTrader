"""Tests for TiingoDataLoader."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture()
def loader():
    with patch.dict("os.environ", {"TIINGO_API_KEY": "test-key-123"}), \
         patch("ggTrader.data.live.tiingo_loader.MIN_REQUEST_INTERVAL", 0.0):
        from ggTrader.data.live.tiingo_loader import TiingoDataLoader

        yield TiingoDataLoader()


def _mock_tiingo_response(symbol: str = "AAPL", days: int = 5):
    """Build a realistic Tiingo JSON response."""
    dates = pd.bdate_range("2024-01-02", periods=days, tz="UTC")
    return [
        {
            "date": d.isoformat(),
            "close": 150.0 + i,
            "high": 152.0 + i,
            "low": 148.0 + i,
            "open": 149.0 + i,
            "volume": 1_000_000 + i * 100,
            "adjClose": 150.0 + i,
            "adjHigh": 152.0 + i,
            "adjLow": 148.0 + i,
            "adjOpen": 149.0 + i,
            "adjVolume": 1_000_000 + i * 100,
        }
        for i, d in enumerate(dates)
    ]


class TestTiingoDataLoader:
    def test_unsupported_interval_raises(self, loader):
        with pytest.raises(ValueError, match="not supported by Tiingo"):
            loader.fetch_ohlcv(["AAPL"], "1h")

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from ggTrader.data.live.tiingo_loader import _get_api_key

            with pytest.raises(ValueError, match="TIINGO_API_KEY not set"):
                _get_api_key()

    def test_fetch_single_symbol(self, loader):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_tiingo_response("AAPL", 5)
        mock_resp.raise_for_status = MagicMock()

        with patch.object(loader._session, "get", return_value=mock_resp):
            df = loader.fetch_ohlcv(
                ["AAPL"],
                "1d",
                start_date=pd.Timestamp("2024-01-01", tz="UTC"),
                end_date=pd.Timestamp("2024-01-10", tz="UTC"),
            )

        assert not df.empty
        assert "AAPL" in df.columns.get_level_values(0)
        fields = set(df["AAPL"].columns)
        assert fields == {"open", "high", "low", "close", "volume"}
        assert df.index.tz is not None

    def test_fetch_multiple_symbols(self, loader):
        responses = {
            "AAPL": _mock_tiingo_response("AAPL", 3),
            "MSFT": _mock_tiingo_response("MSFT", 3),
        }

        def mock_get(url, **kwargs):
            sym = url.split("/")[-2]
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = responses.get(sym, [])
            resp.raise_for_status = MagicMock()
            return resp

        with patch.object(loader._session, "get", side_effect=mock_get):
            df = loader.fetch_ohlcv(["AAPL", "MSFT"], "1d")

        syms = set(df.columns.get_level_values(0).unique())
        assert syms == {"AAPL", "MSFT"}

    def test_404_symbol_skipped(self, loader):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(loader._session, "get", return_value=mock_resp):
            df = loader.fetch_ohlcv(["DELISTED"], "1d")

        assert df.empty

    def test_uses_adjusted_prices(self, loader):
        data = _mock_tiingo_response("AAPL", 1)
        data[0]["adjClose"] = 999.0
        data[0]["close"] = 100.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()

        with patch.object(loader._session, "get", return_value=mock_resp):
            df = loader.fetch_ohlcv(["AAPL"], "1d")

        assert df["AAPL"]["close"].iloc[0] == 999.0

    def test_limit_truncates_result(self, loader):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_tiingo_response("AAPL", 10)
        mock_resp.raise_for_status = MagicMock()

        with patch.object(loader._session, "get", return_value=mock_resp):
            df = loader.fetch_ohlcv(["AAPL"], "1d", limit=3)

        assert len(df) == 3

    def test_list_symbols_returns_sp500(self, loader):
        syms = loader.list_symbols()
        assert len(syms) >= 45
        assert "AAPL" in syms

    def test_interval_map_coverage(self):
        from ggTrader.data.live.tiingo_loader import TIINGO_INTERVAL_MAP

        assert TIINGO_INTERVAL_MAP["1d"] == "daily"
        assert TIINGO_INTERVAL_MAP["1wk"] == "weekly"
        assert TIINGO_INTERVAL_MAP["1mo"] == "monthly"

    def test_auth_header_set(self, loader):
        assert "Token test-key-123" in loader._session.headers["Authorization"]
