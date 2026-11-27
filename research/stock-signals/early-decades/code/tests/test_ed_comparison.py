"""Check comparison source boundaries and accounting without database reads."""

import ed_comparison
import pandas as pd
import pytest
from ed_comparison import after_tax_growth, comparison_label, comparison_returns

pytestmark = pytest.mark.study


def test_labels() -> None:
    assert comparison_label("1969-02-03", "1993-01-29") == "CRSP market return (stand-in)"
    assert comparison_label("1993-02-01", "1998-07-31") == "SPY"
    assert comparison_label("1993-01-29", "1993-02-01") == "CRSP market return (stand-in) + SPY"


def test_tax_on_final_gain() -> None:
    assert after_tax_growth(pd.Series([0.1, 0.1]), 0.15) == pytest.approx(1.1785)
    assert after_tax_growth(pd.Series([-0.1]), 0.15) == pytest.approx(0.9)


def test_join_has_no_duplicate_and_spy_first_return(monkeypatch) -> None:
    class Connection:
        closed = False
        source = ""

        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql, args):
            self.source = "spy" if "spy_daily" in sql else "crsp"

        def fetchall(self):
            if self.source == "crsp":
                return [("1993-01-28", 0.01, 0.01), ("1993-01-29", 0.02, 0.02)]
            return [("1993-01-29", 10.0), ("1993-02-01", 11.0)]

        def close(self):
            self.closed = True

    conn = Connection()
    monkeypatch.setattr(ed_comparison, "connect_readonly", lambda: conn)
    result = comparison_returns("1993-01-28", "1993-02-01")
    assert result.index.tolist() == list(pd.to_datetime(["1993-01-28", "1993-01-29", "1993-02-01"]))
    assert result.tolist() == pytest.approx([0.01, 0.02, 0.1])
    assert conn.closed
