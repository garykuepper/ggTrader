#!/bin/bash
# Resolver-level compat: can <pkg> be installed together with pandas>=3.0.3 and numpy>=2.4.6 and vectorbt 1.2.0?
V=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt120/vectorbt-1.2.0-py3-none-any.whl
cd $(dirname $0)
for p in "$@"; do
  out=$(timeout 180 uv pip install --dry-run --python venv/bin/python -c constraints.txt "$V" "$p" 2>&1)
  rc=$?
  name=$(echo "$p" | tr 'A-Z_' 'a-z-'); pd=$(echo "$out" | grep -iE "^ \+ ($name|pandas)==" | tr '\n' ' ')
  if [ $rc -eq 0 ]; then echo "RESOLVES  $p | $pd"; else echo "FAILS     $p | $(echo "$out" | grep -iE "because|conflict|no solution|not found" | head -2 | tr '\n' ' ' | cut -c1-300)"; fi
done
