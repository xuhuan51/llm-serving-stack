# V4 — Workload Jitter Attribution 执行计划

> 8×A30 PCIe-only。固定模型 Qwen2.5-7B BF16。先 TP=1 再 TP=2 PIX。
> 详细 workload 规格见 `results/p1_4_workload_jitter/workload_spec.md`。

---

## 准备工作（一次性）

### 1. 环境前置确认

```bash
# 确认 8 张卡空闲（每张 ≤ 50MiB）
nvidia-smi --query-gpu=index,memory.used --format=csv

# 复核拓扑
nvidia-smi topo -m

# 确认无残留 vLLM 容器
docker ps | grep -i vllm
# 如果有，逐个 stop:
# docker stop vllm-tp1 vllm-tp2-pix vllm-tp4-pix 2>/dev/null

# 模型路径确认
ls /home/liuguangli/models/Qwen2.5-7B-Instruct
```

### 2. 镜像 / 路径

- 镜像：`vllm/vllm-openai:latest`
- 模型：`/home/liuguangli/models/Qwen2.5-7B-Instruct`
- 结果：`/home/liuguangli/learn-ai-infra/serving-benchmark/results/p1_4_workload_jitter/`

---

## Run T1：TP=1 (GPU 7, port 8000)

### 启动 vLLM

```bash
docker run -d --gpus '"device=7"' \
  --name vllm-tp1 --network host --ipc host --shm-size 16g \
  -v /home/liuguangli/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --tensor-parallel-size 1 \
  --port 8000 --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.85
```

### 等就绪 + warmup CUDA graph

```bash
# 等 ready
for i in {1..60}; do
  if curl -sf http://localhost:8000/v1/models > /dev/null; then echo "ready"; break; fi
  sleep 2
done

# 多等 15s 让 CUDA graph 编译稳定（V3 经验：第 1 个请求 TTFT 8 秒 outlier 就是这个）
sleep 15
```

### 跑 4 象限

```bash
cd /home/liuguangli/learn-ai-infra/serving-benchmark
bash run_v4.sh T1 8000 7
```

预计耗时：每象限 5-15 分钟，总 30-60 分钟。

### 跑完手动做的事

1. 打开 Grafana，用 `result.json` 里的 `wall_start_ts` / `wall_end_ts` 截图同时段的 panel：
   - TPOT P95 + num_requests_running 叠图（最关键）
   - GPU cache usage
   - TTFT P95
2. 截图存为 `RT1_Q1_grafana.png` ... `RT1_Q4_grafana.png`

### 停 vLLM

```bash
docker stop vllm-tp1 && docker rm vllm-tp1
nvidia-smi  # 确认 GPU 7 释放
```

---

## Run T2：TP=2 PIX (GPU 4,5, port 8001)

### 启动

```bash
docker run -d --gpus '"device=4,5"' \
  --name vllm-tp2-pix --network host --ipc host --shm-size 16g \
  -v /home/liuguangli/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --tensor-parallel-size 2 \
  --port 8001 --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.85
```

### 等就绪 + 跑 + 截图 + 停

```bash
for i in {1..60}; do
  if curl -sf http://localhost:8001/v1/models > /dev/null; then echo "ready"; break; fi
  sleep 2
done
sleep 15

cd /home/liuguangli/learn-ai-infra/serving-benchmark
bash run_v4.sh T2 8001 4,5

# Grafana 截图为 RT2_Q1..Q4_grafana.png

docker stop vllm-tp2-pix && docker rm vllm-tp2-pix
nvidia-smi
```

---

## 数据落地确认

```bash
ls /home/liuguangli/learn-ai-infra/serving-benchmark/results/p1_4_workload_jitter/ \
  | grep -E "R(T1|T2)_Q[1-4]"
```

应该看到 8 × 4 = 32 个文件（不算手动截图）：
```
RT1_Q1_result.json   RT1_Q1_raw.txt   RT1_Q1_metrics.jsonl   RT1_Q1_dmon.log
RT1_Q2_*  RT1_Q3_*  RT1_Q4_*
RT2_Q1_*  RT2_Q2_*  RT2_Q3_*  RT2_Q4_*
```

---

## 写 summary（V4 收尾）

跑完后下次开课写 `results/p1_4_workload_jitter/summary.md`，参考 V3 的 summary 格式：

1. **8 runs 分位数对照表**（TTFT/TPOT P50/P95/P99 × 4 象限 × 2 TP）
2. **Workload jitter 对照**：Q4/Q1 P95 比值，证明长 workload 抖动严重
3. **Communication amplification 对照**：T2/T1 同象限 P95 比值，证明 TP 通信放大效应
4. **时间序列归因**：从 `metrics.jsonl` 抽几个 P95 spike 时刻，对照 `num_requests_running` 证明是 batch 涌入引起的
5. **resume bullet**

---

## 失败处理

- vLLM 起不来：`docker logs vllm-tp1 --tail 200`，常见 OOM → 调 `--gpu-memory-utilization 0.80`
- benchmark hang：检查端口、检查 max-model-len 是否够（Q3/Q4 input 2000 + output 1000 < 4096 OK）
- ignore_eos 不识别：vLLM 0.5+ 才完整支持，旧版本通过 `extra_body` 透传可能被忽略——若发现 Q2/Q4 实际输出不到 max_tokens，先确认 vLLM 版本

---

## 不做的事

- 不调 batch / scheduling 参数：留给 V5 对照
- 不开 chunked prefill / prefix cache：留给 V5
- 不跑 TP=4：V3 已经证明在短 workload 下 net loss，V4 不重复
- 不主推 req/s：V3 已说过 wall-time 数据被 outlier 污染，V4 也一样，**主推 TPOT/TTFT 分位数**
