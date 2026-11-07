"""Probe pandas 2.3.3 vs 3.1.1 behaviour for idioms used in a MultiIndex-OHLCV / groupby / SQL / yfinance codebase. Synthetic data only."""
import warnings, sys, numpy as np, pandas as pd, sqlalchemy as sa
warnings.simplefilter("default")
print("pandas", pd.__version__, "numpy", np.__version__)
def t(name, fn):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); r = fn()
        ws = sorted({f"{x.category.__name__}: {str(x.message)[:70]}" for x in w})
        print(f"- {name}: {r}" + (f"   [warnings: {ws[:2]}]" if ws else ""))
    except Exception as e:
        print(f"- {name}: RAISES {type(e).__name__}: {str(e)[:90]}")
idx = pd.bdate_range("2024-01-02", periods=6)
ohlcv = pd.concat({s: pd.DataFrame({"open": 1.0, "high": 2.0, "low": .5, "close": np.arange(6) + 1.0, "volume": 100}, index=idx) for s in ["AAA", "BBB"]}, axis=1)
t("MultiIndex ohlcv['AAA']['close'] works", lambda: ohlcv["AAA"]["close"].iloc[-1])
t("dtype of datetime index (bdate_range)", lambda: str(idx.dtype))
t("to_datetime(['2024-01-02']) dtype", lambda: str(pd.to_datetime(["2024-01-02"]).dtype))
t("pd.Timestamp('2024-01-02').unit", lambda: pd.Timestamp("2024-01-02").unit)
t("DatetimeIndex.astype('int64')[0] (epoch units)", lambda: int(idx.astype("int64")[0]))
t("DatetimeIndex.view('i8')[0]", lambda: int(idx.values.view("i8")[0]))
t("dates .values dtype", lambda: str(idx.values.dtype))
t("str column dtype", lambda: str(pd.Series(["a", "b"]).dtype))
t("str columns Index dtype (symbols)", lambda: str(pd.Index(["AAA", "BBB"]).dtype))
t("Series(['a',None]).isna() / dtype==object?", lambda: (pd.Series(["a", None]).dtype == object, list(pd.Series(["a", None]).isna())))
t(".values is read-only?", lambda: not ohlcv.values.flags.writeable)
t("close.to_numpy() writeable?", lambda: ohlcv["AAA"]["close"].to_numpy().flags.writeable)
def chained():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}); df["a"][df["b"] > 1] = 0; return df["a"].tolist()
t("chained assignment df['a'][mask]=0 -> a", chained)
def view_mut():
    df = pd.DataFrame({"a": [1., 2., 3.]}); s = df["a"]; s.iloc[0] = 99; return df["a"].iloc[0]
t("subset mutation leaks to parent? df['a'] after s=df['a']; s.iloc[0]=99", view_mut)
t("Series[int] on DatetimeIndex-ed series s[0]", lambda: pd.Series([1., 2.], index=pd.bdate_range("2024-01-02", periods=2))[0])
t("Series[int] on int index [10,20]: s[0]", lambda: pd.Series([1., 2.], index=[10, 20])[0])
t("pct_change(fill_method='pad') (default in 2.x)", lambda: pd.Series([1., np.nan, 2.]).pct_change().tolist())
t("pct_change(fill_method=None)", lambda: pd.Series([1., np.nan, 2.]).pct_change(fill_method=None).tolist())
t("df.stack() on 2-level columns (future_stack)", lambda: ohlcv.stack(future_stack=True).shape if pd.__version__ >= "3" else ohlcv.stack().shape)
t("df.stack() default dropna behaviour w/ NaN", lambda: pd.DataFrame({"a": [1., np.nan]}).stack().shape)
gdf = pd.DataFrame({"k": ["x", "x", "y", "y"], "v": [1., 2., 3., 4.]})
t("groupby.agg(np.std) value (numpy ddof=0 vs pandas ddof=1)", lambda: gdf.groupby("k")["v"].agg(np.std).round(4).tolist())
t("groupby.apply include_groups default", lambda: gdf.groupby("k").apply(lambda g: g["v"].sum()).tolist())
t("groupby(categorical key, observed default)", lambda: gdf.assign(k=pd.Categorical(gdf["k"], categories=["x", "y", "z"])).groupby("k")["v"].sum().to_dict())
t("groupby(axis=1) removed", lambda: gdf[["v"]].groupby(level=0, axis=1).sum().shape)
t("resample('M')", lambda: len(pd.Series(1., index=idx).resample("M").sum()))
t("resample('ME')", lambda: len(pd.Series(1., index=idx).resample("ME").sum()))
t("date_range freq='H'", lambda: len(pd.date_range("2024-01-01", periods=3, freq="H")))
t("date_range freq='h'", lambda: len(pd.date_range("2024-01-01", periods=3, freq="h")))
t("fillna(method='ffill')", lambda: pd.Series([1., np.nan]).fillna(method="ffill").tolist())
t("DataFrame.applymap", lambda: pd.DataFrame({"a": [1]}).applymap(lambda x: x).shape)
t("tz_localize('UTC') index -> tz object type", lambda: type(idx.tz_localize("UTC").tz).__name__ + "/" + type(idx.tz_localize("US/Pacific").tz).__module__)
t("tz-aware .values (drops tz?)", lambda: str(pd.Series(idx.tz_localize("US/Pacific")).values.dtype))
# SQL round trip
eng = sa.create_engine("sqlite://")
df = pd.DataFrame({"ts": idx, "symbol": ["AAA"] * 6, "close": np.arange(6) + 1.0})
def sql_rt():
    df.to_sql("ohlcv", eng, index=False); r = pd.read_sql("select * from ohlcv", eng)
    return {c: str(t_) for c, t_ in r.dtypes.items()}
t("to_sql/read_sql round trip dtypes (sqlite)", sql_rt)
def sql_ts():
    r = pd.read_sql("select * from ohlcv", eng, parse_dates=["ts"]); return str(r["ts"].dtype)
t("read_sql parse_dates ts dtype", sql_ts)
def sql_naive_utc():
    d = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02 14:30:00"]), "v": [1.]}); d.to_sql("n", eng, index=False)
    return pd.read_sql("select ts from n", eng)["ts"].iloc[0]
t("naive UTC timestamp survives to_sql(sqlite) as", sql_naive_utc)
t("read_sql_query with chunksize works", lambda: sum(len(c) for c in pd.read_sql("select * from ohlcv", eng, chunksize=4)))
def pg_like_insert():
    from sqlalchemy import text
    with eng.begin() as c: c.execute(text("create table if not exists z(symbol text, ts text)"))
    d = pd.DataFrame({"symbol": ["A", "B"], "ts": ["x", None]})
    d.to_sql("z", eng, if_exists="append", index=False, method="multi"); return pd.read_sql("select * from z", eng)["ts"].tolist()
t("to_sql method='multi' with None/NaN in str col", pg_like_insert)
# yfinance parse (no network): check that MultiIndex columns layout is as ggTrader expects
t("yfinance imports", lambda: __import__("yfinance").__version__)
if "--net" in sys.argv:
    import yfinance as yf
    def net():
        d = yf.download(["SPY", "AAPL"], start="2024-01-02", end="2024-01-12", auto_adjust=True, progress=False)
        return f"cols={list(d.columns.names)} idxdtype={d.index.dtype} lvl0={sorted(set(d.columns.get_level_values(0)))} shape={d.shape}"
    t("yf.download 2 tickers (network)", net)
