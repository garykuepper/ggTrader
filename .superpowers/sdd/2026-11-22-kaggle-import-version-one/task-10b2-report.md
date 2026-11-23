Round 2: correlated scalar probes replace the single-row joins; the linked and placeholder CRSP branches use parameterized LATERAL probes with OFFSET 0 to retain the boundary. Release-load lookup is outside the per-row probes.

Plan evidence: the orchestrator measured the old helper reading 68,879,302 real-data daily-bar rows. In the synthetic EXPLAIN test, the copied helper lookup executes on 1 of 12 chunks and returns one matching row; the linked CRSP branch executes on 2 of 12 chunks. The synthetic and real-data row counts are not directly comparable; the new real-data rows read and timing have not been measured. The test checks the copied helper lookup and branch, rather than introspecting the opaque PL/pgSQL function plan.

Verification: 36 scratch integration tests passed across reader, release and DDL; Ruff check and format passed.
