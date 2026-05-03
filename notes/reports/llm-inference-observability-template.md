# LLM Inference Observability Experiment Template

## Summary

- Date:
- Goal:
- Model:
- Serving engine:
- GPU:
- Main conclusion:

## Service Configuration

```text
base_url:
metrics_url:
model:
gpu:
max_model_len:
max_num_seqs:
max_num_batched_tokens:
gpu_memory_utilization:
```

## Workload

```text
prompt_mode:
stream:
concurrency:
num_requests:
max_tokens:
warmup_requests:
```

## Commands

Benchmark:

```bash
python3 week1-vllm-basics/benchmark_v1.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --prompt-mode long-output \
  --stream \
  --concurrency 8 \
  --num-requests 16 \
  --max-tokens 300 \
  --warmup-requests 1
```

Monitor:

```bash
python3 week1-vllm-basics/monitor_vllm.py \
  --metrics-url http://localhost:8000/metrics \
  --interval 2 \
  --max-num-seqs 32
```

## Benchmark Result

| Metric | Value |
|---|---:|
| Total requests |  |
| Output tokens/chunks |  |
| Throughput |  |
| Avg latency |  |
| P95 latency |  |
| P95 TTFT |  |
| P95 TPOT |  |

## Monitor Evidence

| Signal | Observation |
|---|---|
| running |  |
| waiting |  |
| swapped |  |
| queue_ratio |  |
| running_saturation |  |
| GPU util |  |
| GPU memory |  |
| KV cache usage |  |
| prompt tok/s |  |
| output tok/s |  |

## Diagnosis

```text
diagnosis:
reason:
action:
```

## Interpretation

- What changed:
- Why it changed:
- Bottleneck type:
- Recommended tuning:
- Follow-up experiment:

## Resume Talking Point

```text
Designed and ran a controlled LLM inference experiment that connected benchmark latency/throughput with vLLM scheduler, GPU, and KV cache metrics to identify the serving bottleneck and propose tuning actions.
```
