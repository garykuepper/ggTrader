# Intraday r9 SPY proxy screen: automatic score

**Verdict: NO-GO** — Post-original-sample and single-shot proxy feasibility screen (not an untouched holdout).

- Scored sessions: 2677. Exclusion (R1, closed list): 2019-08-12, 2019-08-13, removed from the scored series (not counted as flat days).
- R1 (r9 protocol section 3): 2019-08-12 and 2019-08-13 are removed from the scored series, not counted as flat days, because their required price bars do not exist in the data (2019-08-12 lacks the 15:30 entry bar and the 15:59 exit bar; 2019-08-13 consequently lacks its prior-close anchor). The list was chosen after the missing timestamps were known but before any price, signal or return was seen. Losing a day for a reason linked to how the market moved that day could bias the result; that effect cannot be measured without prices, so it is a named limitation.
- A1 NW-5 t (unrounded, threshold 3.0): -3.479259029171949 -> FAIL
- A2 after-tax-policy CAGR (threshold 0.03): -0.036758185289904044 -> FAIL
- Stress (2 bp/side, cannot change verdict): t -6.067339476333766, CAGR -0.06240581456888006
- SPY distribution-reinvested CAGR (no tax): 0.15068957827324403; sleeve minus SPY: -0.18744776356314807
- A standalone A1/A2 pass says nothing about beating SPY going forward; SPY displacement is a mandatory non-gating diagnostic.
- Diagnostic errors: none

Identities: {"code_sha256": "ebef81e13777e295474247c57e50fb094d374ab059d4f8cc9af0db4245f70612", "data_manifest_sha256": "fec4cce762c65795c27d271974a552f690ff028da713cee764ff8bfd43d1359a", "issuer_workbook_sha256": "6b65cbf956b5d25705d08470006a0e97e6a7d2a0d5cfd08b63f6f79783b06320", "manifest_sha256": "cbdd63c82eb25d43b2c113c2309fcfbb9dc228ae5ffa77758626af570c08422a", "protocol_sha256": "222521cebe23aed66f9a44672dd1645a210f0c545a7cfbd30482496922593ed3", "runtime": {"numpy": "2.5.4", "pandas": "3.1.1", "pandas_market_calendars": "5.6.0", "python": "3.12.13", "requests": "2.34.2", "statsmodels": "0.15.0"}, "vendor_review_sha256": "30924032071c6a51cfd87a3e119671b1993bfb54c70bc966cef5d098083fcfbb"}
Validation SHA256: 22b696ad22a3aa365800993e8a0e93e311136420cc89ab0c4152d293056cb212

All diagnostics are in the raw JSON next to this file.
