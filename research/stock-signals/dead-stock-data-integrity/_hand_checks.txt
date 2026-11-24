# Hand checks behind the findings note

Read-only queries run on 2026-11-24 against the `ggtrader` database (read-only session), rerun once to capture these outputs. Nothing here is produced by `run.py`; the findings note cites this file for figures that are not in `_integrity.json`.

## 1. Permno 92449: rows by exchange code (no row filter)

```sql
SELECT exchcd, shrcd, count(*) AS n_rows, min(date) AS first_day, max(date) AS last_day
FROM kaggle_crsp.daily
WHERE load_id = 1 AND permno = 92449 AND date >= '2004-01-01'
GROUP BY exchcd, shrcd ORDER BY first_day;
```

Output (0.0 s):

```
exchcd | shrcd | n_rows | first_day | last_day
3 | 11 | 2230 | 2007-11-28 | 2023-02-15
2 | 11 | 1600 | 2012-12-31 | 2019-05-09
```

## 2. Permno 77263: rows per year (no row filter)

```sql
SELECT extract(year FROM date)::int AS year, shrcd, exchcd, count(*) AS n_rows
FROM kaggle_crsp.daily
WHERE load_id = 1 AND permno = 77263
GROUP BY 1, 2, 3 ORDER BY 1;
```

Output (0.0 s):

```
year | shrcd | exchcd | n_rows
1992 | 11 | 3 | 243
1993 | 11 | 3 | 253
1994 | 11 | 3 | 252
1995 | 11 | 3 | 252
1996 | 11 | 3 | 254
1997 | 11 | 3 | 253
1998 | 11 | 3 | 252
1999 | 11 | 3 | 252
2000 | 11 | 3 | 252
2001 | 11 | 3 | 248
2002 | 11 | 3 | 252
2003 | 11 | 3 | 252
2004 | 11 | 3 | 252
2005 | 11 | 3 | 248
2022 | 11 | 3 | 30
2023 | 11 | 3 | 250
2024 | 11 | 3 | 252
```

## 3. The slow `daily_of()` probe (the "over 12 minutes" figure)

**3a. Query as first run (with the stock-spans filter), rerun with a 60-second limit**

```sql
SELECT DISTINCT d.permno FROM kaggle_crsp.daily_of(1::bigint) d
WHERE d.shrcd = ANY(ARRAY[10,11]) AND d.exchcd = ANY(ARRAY[1,3])
AND d.date >= '2004-01-01' AND d.date <= '2024-12-31'
AND d.permno IN (SELECT permno FROM market.stock_spans WHERE release_id = 1 AND permno IS NOT NULL);
```

Result: stopped by the 60 s limit after 60.1 s: QueryCanceled.

**3b. Same query without the stock-spans filter**

```sql
SELECT DISTINCT d.permno FROM kaggle_crsp.daily_of(1::bigint) d
WHERE d.shrcd = ANY(ARRAY[10,11]) AND d.exchcd = ANY(ARRAY[1,3])
AND d.date >= '2004-01-01' AND d.date <= '2024-12-31';
```

Result: finished, 9422 rows, 25.4 s.

**3c. Same as 3a, reading the table by load id (what `run.py` does)**

```sql
SELECT DISTINCT d.permno FROM kaggle_crsp.daily d
WHERE d.load_id = 1 AND d.shrcd = ANY(ARRAY[10,11]) AND d.exchcd = ANY(ARRAY[1,3])
AND d.date >= '2004-01-01' AND d.date <= '2024-12-31'
AND d.permno IN (SELECT permno FROM market.stock_spans WHERE release_id = 1 AND permno IS NOT NULL);
```

Result: finished, 9422 rows, 0.7 s.

Original observation (2026-11-24, `pg_stat_activity`, a first probe that I then stopped): a query like 3a had been active for 12 minutes 15 seconds, and 12 minutes 19 seconds a little later, still not finished. The 12 minutes is a lower bound from that one observation, not a measured total. The reason it was slow was not determined; do not read it as a measured property of `daily_of()`.

## 4. Figures from the earlier, wrongly filtered scan (commit `ba4b7f83`)

Command: `git show ba4b7f83:research/stock-signals/dead-stock-data-integrity/_integrity.json` filtered as shown.

```
211-        "permno": 81761,
212-        "date": "2005-10-05",
213:        "detail": "3865 days missing",
--
545-            "permno": 92449,
546-            "date": "2012-12-31",
547:            "detail": "1600 days missing",
```

```
{'spikes': (342, 251), 'gaps': (220, 195), 'stale': (239, 152)}
[{'permno': 92449, 'date': '2012-12-31', 'detail': '1600 days missing', 'magnitude': 1600.0}]
```
