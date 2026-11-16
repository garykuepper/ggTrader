# Event-timing audit cache location

The raw SEC responses, manifests, checkpoints and registrations of the event-timing audit are one
immutable, hash-verified bundle in the gg-library (shelf Trading, `bundles/event_timing_audit/`).
File names, bytes and relative layout are unchanged. `data/external/event_timing_audit/<run>` are
links into that bundle; code and the dated audit documents keep using those repository paths.

| Former location | Current location |
| :--- | :--- |
| `/media/thesix/ggTrader-data/event_timing_audit/<run>/` | `<gg-library>/Trading/bundles/event_timing_audit/<run>/` |

Dated audit documents are historical records and are not rewritten. Verify with
`research library bundle verify event_timing_audit`.
