Status: DONE_WITH_CONCERNS
Base: d03bcc8f5f222c819fce44a96f7bb82032594411.
Implemented linked reader, missing-input mapping, and integration checks for join, rescaling, drift, and backfill. Scratch integration run: 29 passed. Ruff check and format passed.
Coverage missing: dedicated SMCI first-day split, two-linked-symbols, missing CRSP join row, other-symbol drift isolation, and chunk-exclusion EXPLAIN tests. Chunk exclusion has not been verified; performance and those cases need verification before relying on this change.
