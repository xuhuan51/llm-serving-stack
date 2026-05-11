#!/usr/bin/env bash
# V5: 70B AWQ TP=8 concurrency sweep on Q4
# 用法: bash run_v5_sweep.sh <ENGINE> <PORT> <GPU_LIST>
#   bash run_v5_sweep.sh VLLM 8001 0,1,2,3,4,5,6,7
#   bash run_v5_sweep.sh SGLANG 8002 0,1,2,3,4,5,6,7
set -euo pipefail
ENGINE="${1:?usage: run_v5_sweep.sh <ENGINE> <PORT> <GPU_LIST>}"
PORT="${2:?port}"
GPU_LIST="${3:?gpu list}"
NREQ="${NREQ:-500}"
WARMUP="${WARMUP:-10}"
RES=/home/liuguangli/learn-ai-infra/serving-benchmark/results/p1_5_chunked_prefill
SB=/home/liuguangli/learn-ai-infra/serving-benchmark
mkdir -p "$RES"

for C in ${CS:-32 64 128 256}; do
  TAG="${ENGINE}_TP8_C${C}"
  PREFIX="${RES}/RV5${TAG}"
  echo "============================================="
  echo ">>> ${TAG}  $(date +'%H:%M:%S')"
  echo "============================================="

  nvidia-smi dmon -s pucm -i "$GPU_LIST" -d 1 -o T > "${PREFIX}_dmon.log" 2>&1 &
  DMON_PID=$!

  python3 "$SB/v4_quadrant_runner.py" \
    --base-url "http://localhost:${PORT}/v1" \
    --model qwen \
    --quadrant Q4 \
    --concurrency "$C" \
    --num-requests "$NREQ" \
    --warmup "$WARMUP" \
    --ignore-eos \
    --output-dir "$RES" \
    --tag "V5${TAG}" \
    2>&1 | tee "${PREFIX}_raw.txt"

  kill $DMON_PID 2>/dev/null || true
  wait $DMON_PID 2>/dev/null || true
  echo "  done ${TAG}, sleeping 15s for engine to settle"
  sleep 15
done

echo "============================================="
echo "${ENGINE} sweep complete"
ls -la "$RES" | grep "V5${ENGINE}_TP8" || true
