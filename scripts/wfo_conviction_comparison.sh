#!/bin/bash
# WFO Comparison: ensemble (flat 2%) vs ensemble_conviction (1%-4%)
# Uses reduced grid (16 combos) to fit in memory with 562 symbols
set -e
cd "$(dirname "$0")/.."

LOG="results/wfo_conviction_comparison_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "===========================================" | tee "$LOG"
echo "WFO Comparison: ensemble vs ensemble_conviction" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- RUN 1: ensemble (flat 2%) ---" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy ensemble --wfo \
  --sweep-param min_agree=2 \
  --sweep-param bb_period=20 \
  --sweep-param bb_std=2.0 \
  --sweep-param rsi_period=14 \
  --sweep-param rsi_oversold=30 \
  --sweep-param ema_fast=10,20 \
  --sweep-param ema_slow=50,100 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- RUN 2: ensemble_conviction (1%-4%) ---" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy ensemble_conviction --wfo \
  --sweep-param min_agree=2 \
  --sweep-param bb_period=20 \
  --sweep-param bb_std=2.0 \
  --sweep-param rsi_period=14 \
  --sweep-param rsi_oversold=30 \
  --sweep-param ema_fast=10,20 \
  --sweep-param ema_slow=50,100 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "Both runs complete: $(date)" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
