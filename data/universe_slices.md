# Universe slice JSON files

Curated subsets of the same schema as `top_25_USD_2023-01-01_2025-12-31.json` (list of objects with `symbol`, `rank`, etc.). Use with `--symbols-file` and `--max-symbols` on `scripts/run_full_pipeline.py`.

| File | Purpose |
|------|---------|
| `majors_top8_usd_2023-01-01_2025-12-31.json` | Top eight by notional volume from that ranking (BTC … LTC). |
| `alt_ranks_9_16_usd_2023-01-01_2025-12-31.json` | Next eight ranks (DOT … INJ) for alt-heavy experiments. |

Regenerate from a new master list by copying the corresponding slice; keep date suffix aligned with the source ranking file when possible.
