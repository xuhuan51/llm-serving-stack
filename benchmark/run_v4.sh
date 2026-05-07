#!/usr/bin/env bash
# V4 — 跑完一个 TP 配置下的全部 4 象限。
#
# 前置条件：vLLM 已经在 BASE_URL 启动并就绪。
# 用法：
#   bash run_v4.sh T1 8000 7         # T1 (TP=1), 端口 8000, 监控 GPU 7
#   bash run_v4.sh T2 8001 4,5       # T2 (TP=2 PIX), 端口 8001, 监控 GPU 4,5
#
# 每个象限产出 4 个文件：
#   R<TAG>_<Q>_result.json    benchmark 主输出（含分位数 + 原始数据）
#   R<TAG>_<Q>_raw.txt        终端日志
#   R<TAG>_<Q>_metrics.jsonl  Prometheus 时间序列（1Hz 采样）
#   R<TAG>_<Q>_dmon.log       nvidia-smi dmon GPU 利用率
set -euo pipefail

TAG="${1:?usage: run_v4.sh <TAG> <PORT> <GPU_LIST>}"
PORT="${2:?missing port}"
GPU_LIST="${3:?missing gpu list, e.g. 7 or 4,5}"

CONCURRENCY="${CONCURRENCY:-16}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
WARMUP="${WARMUP:-5}"
INTERVAL="${INTERVAL:-1}"
RESULTS="${RESULTS:-/home/liuguangli/learn-ai-infra/serving-benchmark/results/p1_4_workload_jitter}"

cd "$(dirname "$0")"
mkdir -p "$RESULTS"

BASE_URL="http://localhost:${PORT}/v1"
METRICS_URL="http://localhost:${PORT}/metrics"

echo "==========================================="
echo "V4 run: TAG=$TAG PORT=$PORT GPU=$GPU_LIST"
echo "  concurrency=$CONCURRENCY num_requests=$NUM_REQUESTS warmup=$WARMUP"
echo "  results -> $RESULTS"
echo "==========================================="

# 验证 vLLM 就绪
echo "checking vLLM at $BASE_URL ..."
for i in {1..30}; do
  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "  vLLM ready"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: vLLM not ready after 30s, abort"
    exit 1
  fi
  sleep 1
done

run_quadrant() {
  local Q=$1
  local PREFIX="${RESULTS}/R${TAG}_${Q}"
  echo ""
  echo ">>> Quadrant ${Q}"

  # 起 dmon (GPU 利用率时间序列)
  nvidia-smi dmon -s pucm -i "$GPU_LIST" -d 1 -o T > "${PREFIX}_dmon.log" &
  local DMON_PID=$!

  # 起 metrics poller (Prometheus 时间序列)
  python v4_metrics_poller.py \
    --url "$METRICS_URL" \
    --interval "$INTERVAL" \
    --output "${PREFIX}_metrics.jsonl" \
    > "${PREFIX}_poller.stderr" 2>&1 &
  local POLLER_PID=$!

  # 跑 benchmark
  # Q2 / Q4 使用 ignore_eos 强制满输出（控制 output token 数）
  local IGNORE_EOS_FLAG=""
  case "$Q" in
    Q2|Q4) IGNORE_EOS_FLAG="--ignore-eos" ;;
  esac

  python v4_quadrant_runner.py \
    --base-url "$BASE_URL" \
    --model qwen \
    --quadrant "$Q" \
    --concurrency "$CONCURRENCY" \
    --num-requests "$NUM_REQUESTS" \
    --warmup "$WARMUP" \
    $IGNORE_EOS_FLAG \
    --output-dir "$RESULTS" \
    --tag "$TAG" \
    2>&1 | tee "${PREFIX}_raw.txt"

  # 收尾
  kill "$DMON_PID" 2>/dev/null || true
  kill "$POLLER_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true
  wait "$POLLER_PID" 2>/dev/null || true

  echo "  done ${Q}, files in $RESULTS"
  # 让 GPU 缓一下，避免象限之间 KV cache / batch 状态串味
  sleep 5
}

run_quadrant Q1
run_quadrant Q2
run_quadrant Q3
run_quadrant Q4

echo ""
echo "==========================================="
echo "V4 ${TAG} complete. Files:"
ls -la "$RESULTS" | grep "R${TAG}_"
echo "==========================================="
