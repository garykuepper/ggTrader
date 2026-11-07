#!/usr/bin/env bash
# usage: runsuite.sh <venvname> <engine|default> 
n=$1; eng=$2; cd /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/p8/copy
E=(); [[ $eng != default ]] && E=(GGT_VBT_ENGINE=$eng)
systemd-run --user --scope -q --collect -p MemoryMax=6G -p MemorySwapMax=0 \
  env -i PATH=/usr/bin:/bin HOME=/tmp/p8home PYTHONPATH=src "${E[@]}" ../venv_$n/bin/python -m pytest tests/engine -q -p no:cacheprovider -rfE --tb=short -x --maxfail=1000 > ../out/${n}_${eng}.log 2>&1
echo "exit=$?" >> ../out/${n}_${eng}.log
