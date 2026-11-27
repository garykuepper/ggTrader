"""Check comparison source boundaries and accounting without database reads."""

import pandas as pd
import pytest
from ed_comparison import after_tax_growth, comparison_label, comparison_returns

pytestmark = pytest.mark.study


def test_labels_and_missing_january() -> None:
    assert comparison_label("1969-02-03", "1992-12-31") == "CRSP market return (stand-in)"
    assert comparison_label("1993-01-29", "1998-07-31") == "SPY"
    with pytest.raises(ValueError, match="1993-01-01"):
        comparison_returns("1969-02-03", "1998-07-31")


def test_tax_on_final_gain() -> None:
    assert after_tax_growth(pd.Series([0.1, 0.1]), 0.15) == pytest.approx(1.1785)
    assert after_tax_growth(pd.Series([-0.1]), 0.15) == pytest.approx(0.9)


def test_spy_reader_uses_adjusted_close_and_closes_connection(monkeypatch) -> None:
    import ed_comparison

    class Connection:
        closed = False

        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql, args):
            assert "spy_daily_yahoo_1993_1999" in sql
            assert args == ("1993-01-29", "1993-02-01")

        def fetchall(self):
            return [("1993-01-29", 10.0), ("1993-02-01", 11.0)]

        def close(self):
            self.closed = True

    conn = Connection()
    monkeypatch.setattr(ed_comparison, "connect_readonly", lambda: conn)
    result = comparison_returns("1993-01-29", "1993-02-01")
    assert result.tolist() == pytest.approx([0.0, 0.1])
    assert conn.closed
