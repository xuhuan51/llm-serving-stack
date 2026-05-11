#!/usr/bin/env bash
set -uo pipefail
SB=/home/liuguangli/learn-ai-infra/serving-benchmark
RES=$SB/results/p1_5_chunked_prefill
LOG=$RES/V5_extreme.log
log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a $LOG; }

log "=== V5 极限加压：c=384 + c=512 双引擎 ==="
log "GPU 0 同事占用："
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -3 >> $LOG

# vLLM
log "Step 1: 起 vLLM"
docker rm -f vllm-v5-72b-tp8 2>/dev/null
docker run -d --gpus all --name vllm-v5-72b-tp8 \
  --network host --ipc host --shm-size 32g \
  -v /home/liuguangli/models:/models \
  vllm/vllm-openai:latest \
  /models/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name qwen \
  --tensor-parallel-size 8 --quantization awq_marlin \
  --port 8001 --dtype float16 \
  --max-model-len 4096 --gpu-memory-utilization 0.78 2>&1 | tail -1 >> $LOG

log "Step 2: 等 vLLM ready"
WS=$(date +%s)
until curl -sf http://localhost:8001/metrics 2>/dev/null | grep -q '^vllm:'; do
  if [ $(( $(date +%s) - WS )) -gt 600 ]; then log "❌ vLLM 10min 未起"; exit 1; fi
  sleep 5
done
log "  ✅ vLLM ready"

log "Step 3: vLLM sweep CS='384 512'"
CS="384 512" bash $SB/run_v5_sweep.sh VLLM 8001 0,1,2,3,4,5,6,7 2>&1 | tail -50 >> $LOG

log "Step 4: stop vLLM"
docker stop vllm-v5-72b-tp8 2>&1 | tail -1 >> $LOG
docker rm vllm-v5-72b-tp8 2>&1 | tail -1 >> $LOG
sleep 15

# SGLang
log "Step 5: 起 SGLang"
docker rm -f sglang-v5-72b-tp8 2>/dev/null
docker run -d --gpus all --name sglang-v5-72b-tp8 \
  --network host --ipc host --shm-size 32g \
  -v /home/liuguangli/models:/models \
  --entrypoint python3 lmsysorg/sglang:latest-runtime \
  -m sglang.launch_server \
    --model-path /models/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name qwen \
    --tensor-parallel-size 8 --quantization awq_marlin \
    --mem-fraction-static 0.78 --context-length 4096 \
    --host 0.0.0.0 --port 8002 --enable-metrics 2>&1 | tail -1 >> $LOG

log "Step 6: 等 SGLang ready (CUDA graph compile ~6min)"
WS=$(date +%s)
until curl -sf http://localhost:8002/v1/models > /dev/null 2>&1; do
  if [ $(( $(date +%s) - WS )) -gt 900 ]; then log "❌ SGLang 15min 未起"; exit 1; fi
  sleep 10
done
log "  ✅ SGLang ready"

log "Step 7: SGLang sweep CS='384 512'"
CS="384 512" bash $SB/run_v5_sweep.sh SGLANG 8002 0,1,2,3,4,5,6,7 2>&1 | tail -50 >> $LOG

log "Step 8: stop SGLang (no rm yet)"
docker stop sglang-v5-72b-tp8 2>&1 | tail -1 >> $LOG

log "Step 9: PromQL pull (新增的 4 个点)"
python3 $SB/v5_promql_pull.py --result-dir $RES --engine vllm --pattern 'RV5VLLM_TP8_C{384,512}_Q4_result.json' 2>&1 | tail -10 >> $LOG
python3 $SB/v5_promql_pull.py --result-dir $RES --engine vllm --pattern 'RV5VLLM_TP8_C384_Q4_result.json' 2>&1 | tail -5 >> $LOG
python3 $SB/v5_promql_pull.py --result-dir $RES --engine vllm --pattern 'RV5VLLM_TP8_C512_Q4_result.json' 2>&1 | tail -5 >> $LOG
python3 $SB/v5_promql_pull.py --result-dir $RES --engine sglang --container sglang-v5-72b-tp8 --pattern 'RV5SGLANG_TP8_C384_Q4_result.json' 2>&1 | tail -5 >> $LOG
python3 $SB/v5_promql_pull.py --result-dir $RES --engine sglang --container sglang-v5-72b-tp8 --pattern 'RV5SGLANG_TP8_C512_Q4_result.json' 2>&1 | tail -5 >> $LOG

log "Step 10: 重出图（6 c 点全集）"
# 改 plot 的 xticks
python3 $SB/v5_plot.py --dir $RES 2>&1 | tail -10 >> $LOG

log "Step 11: cleanup"
docker rm sglang-v5-72b-tp8 2>&1 | tail -1 >> $LOG

log "=== V5 极限实验 DONE ==="
touch $RES/.EXTREME_DONE
