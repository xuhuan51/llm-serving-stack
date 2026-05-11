#!/usr/bin/env bash
# 起 SGLang 72B AWQ TP=8 (Marlin)，准备 sweep
set -euo pipefail
PORT="${PORT:-8002}"
MEM="${MEM:-0.78}"
CTX="${CTX:-4096}"
NAME=sglang-v5-72b-tp8

docker rm -f $NAME 2>/dev/null || true

docker run -d --gpus all --name $NAME \
  --network host --ipc host --shm-size 32g \
  -v /home/liuguangli/models:/models \
  --entrypoint python3 \
  lmsysorg/sglang:latest-runtime \
  -m sglang.launch_server \
    --model-path /models/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name qwen \
    --tensor-parallel-size 8 \
    --quantization awq_marlin \
    --mem-fraction-static $MEM \
    --context-length $CTX \
    --host 0.0.0.0 --port $PORT \
    --enable-metrics

echo "container started: $NAME on :$PORT"
echo "wait for ready (model load takes ~1-2 min)..."
