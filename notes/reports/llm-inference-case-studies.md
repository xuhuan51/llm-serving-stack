# LLM Inference Case Studies

These case studies summarize completed vLLM experiments from the learning server. They are intended as portfolio material and interview talking points.

## Case 1: Long-Output Concurrency Scaling

**Goal:** Understand whether higher client concurrency improves throughput or hurts latency for long generation workloads.

**Workload:**

```text
model: qwen
prompt_mode: long-output
stream: true
num_requests: 64
max_tokens: 300
concurrency: 8,16,24,32
```

**Benchmark result:**

| Concurrency | Throughput | Avg latency | P95 latency | Avg TTFT | P95 TTFT | Avg TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 390.8 tok/s | 6.09s | 6.11s | 0.07s | 0.08s | 0.020s/token |
| 16 | 741.4 tok/s | 6.42s | 6.50s | 0.10s | 0.19s | 0.021s/token |
| 24 | 968.7 tok/s | 6.58s | 6.97s | 0.13s | 0.48s | 0.022s/token |
| 32 | 1304.3 tok/s | 7.27s | 7.64s | 0.50s | 0.86s | 0.023s/token |

**Interpretation:** Throughput kept improving through concurrency 32, but first-token latency and tail latency began to rise. TPOT only increased mildly, so the main degradation was not per-token decode slowdown; it was queueing or scheduling pressure before the first token.

**Tuning conclusion:** For chat workloads that care about responsiveness, concurrency 16-24 is a better operating range than 32. If pure throughput matters more than first-token latency, concurrency 32 may still be acceptable.

## Case 2: Forced Queue With `max_num_seqs=8`

**Goal:** Prove that server-side scheduler capacity can dominate TTFT even when per-token generation remains healthy.

**Workload:**

```text
model: qwen
prompt_mode: long-output
stream: true
num_requests: 32
max_tokens: 300
client concurrency: 32
server max_num_seqs: 8
```

**Benchmark result:**

| Metric | Value |
|---|---:|
| Throughput | 391.9 tok/s |
| Avg latency | 15.21s |
| P50 latency | 18.22s |
| P95 latency | 24.29s |
| Avg TTFT | 9.20s |
| P50 TTFT | 12.20s |
| P95 TTFT | 18.27s |
| Avg TPOT | 0.020s/token |
| P95 TPOT | 0.021s/token |

**Monitor evidence:**

```text
running ~= 8
waiting stayed high, then drained
GPU util ~= 100%
KV cache usage low
output rate ~= 400 tok/s
```

**Diagnosis:**

```text
TTFT high + TPOT normal + waiting high + running at max_num_seqs
=> scheduler_queueing
```

**Tuning conclusion:** This is not a decode bottleneck or KV cache bottleneck. The active sequence limit constrained admission. Increase `max_num_seqs` only if KV cache and GPU memory have headroom, or reduce client concurrency / add replicas.

## Case 3: Long Context vs Long Output

**Goal:** Separate prefill pressure from decode pressure.

**Workload:**

```text
model: qwen
stream: true
concurrency: 8
num_requests: 16
```

**Benchmark result:**

| Prompt mode | Max tokens | Throughput | Avg latency | P95 latency | Avg TTFT | P95 TTFT | Avg TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-output | 300 | 391.8 tok/s | 6.08s | 6.10s | 0.07s | 0.09s | 0.020s/token |
| long-context | 50 | 304.7 tok/s | 1.14s | 1.25s | 0.15s | 0.18s | 0.022s/token |
| long-context-output | 300 | 358.7 tok/s | 6.64s | 6.66s | 0.14s | 0.17s | 0.022s/token |

**Monitor evidence:** In long-context-output mode, `running=8`, `waiting=0`, GPU 1 was near full utilization, prompt token rate spiked, output rate stayed around 368 tok/s, and KV cache usage stayed around 4-6%.

**Interpretation:** Long context raised TTFT because prefill had to process more prompt tokens and build KV cache before the first output token. Long output raised total latency because decode generated many more tokens. The combined workload showed both effects.

**Tuning conclusion:** Use TTFT/TPOT separation to decide whether to optimize prompt/prefill behavior or output/decode behavior. In this case, queueing was not the main cause because waiting stayed at zero.

## Case 4: BF16 vs AWQ Quantized Serving

**Goal:** Measure how AWQ quantization changes model memory, KV cache capacity, latency, and throughput under the same vLLM workload.

**Service setup:**

```text
serving engine: vLLM 0.19.1
gpu: NVIDIA A30 24GB
max_model_len: 8192
BF16 model: /models/Qwen2.5-7B-Instruct
AWQ model: /models/Qwen2.5-7B-Instruct-AWQ
AWQ kernel: awq_marlin / MarlinLinearKernel
```

**Model memory and KV cache planning:**

| Model | Model loading memory | Available KV cache memory | GPU KV cache tokens | Max full-length concurrency |
|---|---:|---:|---:|---:|
| BF16 | 14.25 GiB | 6.24 GiB | 116,768 | 14.25x at 8,192 tokens/request |
| AWQ | 5.2 GiB | 14.8 GiB | 277,184 | 33.84x at 8,192 tokens/request |

**Warm benchmark result:**

| Model | Concurrency | Throughput | Avg latency | P95 latency | P95 TTFT | P95 TPOT |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 8 | 392.1 tok/s | 6.08s | 6.08s | 0.08s | 0.021s/token |
| AWQ | 8 | 950.0 tok/s | 2.51s | 2.53s | 0.06s | 0.008s/token |
| BF16 | 32 | 1372.6 tok/s | 6.89s | 6.92s | 0.19s | 0.023s/token |
| AWQ | 32 | 2557.8 tok/s | 3.72s | 3.73s | 0.31s | 0.012s/token |

**Monitor evidence:**

```text
BF16 concurrency=32:
running=32, waiting=0, GPU util ~= 100%, KV cache usage peaked around 7.5%

AWQ concurrency=32:
running=32, waiting=0, GPU util ~= 100%, KV cache usage stayed low
```

**Interpretation:** AWQ reduced model loading memory from 14.25 GiB to 5.2 GiB. vLLM used the freed memory to allocate more KV cache, increasing GPU KV cache capacity from 116,768 to 277,184 tokens. On this long-output workload, AWQ also improved decode speed: P95 TPOT dropped from about 0.021-0.023s/token to 0.008-0.012s/token.

**Important caveat:** The first AWQ run included runtime compilation/warmup effects and showed abnormally high TTFT. Warm runs are the fair comparison for steady-state serving.

**Tuning conclusion:** Quantization should be evaluated with both memory metrics and latency metrics. In this experiment, AWQ improved both memory headroom and decode throughput, making it a strong candidate for higher-concurrency or longer-context serving. Quality impact still needs a separate evaluation if the workload is user-facing.
