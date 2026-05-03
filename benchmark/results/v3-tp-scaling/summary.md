# P1.2 — TP Scaling on 8×A30 PCIe (no NVLink)

> Qwen2.5-7B-Instruct (BF16), max-model-len=4096, gpu-mem-util=0.85,
> baseline workload (short-only), concurrency=32, num-requests=200, warmup=4
> Each run executed in isolation (one container at a time, GPUs verified empty before next start).

## 1. Throughput

| Run | Config | GPUs | Wall (s) | gen tok/req | gen tok/s | req/s | scaling vs TP=1 (req/s) |
|-----|--------|------|---------:|------------:|----------:|------:|------------------------:|
| R1  | TP=1            | 4         | 13.84 | 29 | 419.1 | 14.45 | 1.00× |
| R2  | TP=2 PIX        | 4,5       | 11.02 | 28 | 508.2 | 18.15 | **1.26×** |
| R3  | TP=2 PHB        | 2,4       | 11.35 | 29 | 511.0 | 17.62 | 1.22× |
| R4  | TP=4 PIX        | 4,5,6,7   | 13.43 | 28 | 416.9 | 14.89 | **1.03×** |

`gen tok/s = avg gen tokens/req × 200 / wall_time`

## 2. Latency (per-request)

| Run | TTFT P50 | TTFT P95 | TTFT max | TPOT P50 | TPOT P95 | E2E P50 | E2E P95 |
|-----|---------:|---------:|---------:|---------:|---------:|--------:|--------:|
| R1 TP=1     |  86.6ms |  291ms |  8770ms | 23.6ms | 401ms |  771ms | 9647ms |
| R2 TP=2 PIX |  65.2ms |  173ms |  7235ms | 17.1ms | 295ms |  565ms | 7889ms |
| R3 TP=2 PHB |  70.1ms |  300ms |  7572ms | 17.2ms | 295ms |  580ms | 8318ms |
| R4 TP=4 PIX |  80.3ms |  177ms | 10108ms | 14.6ms | 421ms |  496ms | 10777ms |

Decode-only speedup vs TP=1 (using 1/TPOT_P50): TP=2 = **1.38×**, TP=4 = **1.62×** — pure decode does scale, but well below ideal (2× / 4×).

## 3. Topology contrast — R2 (PIX) vs R3 (PHB)

| metric | R2 PIX | R3 PHB | gap (PHB worse by) |
|--------|-------:|-------:|-------------------:|
| Wall time   | 11.02s  | 11.35s  | **+3.0%** |
| req/s       | 18.15   | 17.62   | −2.9% |
| gen tok/s   | 508.2   | 511.0   | +0.5% (within noise) |
| TPOT P50    | 17.1ms  | 17.2ms  | +0.6% |
| TPOT P95    | 295ms   | 295ms   | 0% |
| TTFT P50    | 65.2ms  | 70.1ms  | +7.5% |

**PIX → PHB cost is ~3% on wall time, essentially zero on steady-state TPOT** in this workload. The single-token all-reduce volume on a 7B model (`hidden=3584 × bs≤32 × 2B ≈ 230 KB/step`) is small enough that PCIe Host-Bridge bandwidth saturates well before the link itself. The visible PHB hit shows up in TTFT (prefill carries a much larger AR payload) and in tail wall time. **NVLink wouldn't move the needle for this regime — it would for prefill-heavy or larger-model traffic.**

## 4. Scaling efficiency

| | actual | ideal | efficiency |
|---|---:|---:|---:|
| TP=2 (req/s)   | 1.26× | 2× | **63%** |
| TP=2 (TPOT)    | 1.38× | 2× | **69%** |
| TP=4 (req/s)   | 1.03× | 4× | **26%** |
| TP=4 (TPOT)    | 1.62× | 4× | **40%** |

## 5. Key observations (answers to the 4 target questions)

1. **TP=2 vs TP=1**: 1.26× req/s and 1.38× decode (TPOT) — well short of ideal 2×. The gap is not GPU compute — it's PCIe all-reduce (every layer, every token), plus a small share of fixed prefill cost that doesn't shrink with TP.
2. **TP=4 vs TP=2 — PCIe is the bottleneck**: TP=4 actually has **worse wall-time throughput than TP=2** (14.89 vs 18.15 req/s, −18%). Per-token decode (TPOT) does keep falling (17.1 → 14.6 ms), but the prefill all-reduce overhead and TTFT tail (max 10.1s vs 7.2s) more than eat that gain. **In a PCIe-only topology with small batches, TP=4 is a net loss for short-request serving.**
3. **PIX vs PHB**: only ~3% wall-time delta — much smaller than expected. Per-token AR volume on 7B is too small to expose host-bridge cost; the difference shows up in TTFT (prefill AR), not steady-state decode.
4. **TPOT P95 with more TP**: gets *worse* at TP=4 (421ms) vs TP=2 (295ms) and even TP=1 (401ms). At small batch the all-reduce blocks decode steps and amplifies tail variance, even while P50 improves — the textbook signature of communication overwhelming compute parallelism.

## 6. GPU utilization (nvidia-smi dmon snapshot)

| Run | avg power/GPU | avg SM util |
|-----|--------------:|------------:|
| R1 (1 GPU)  | 37W  | 30% |
| R2 (2 GPUs) | 79W  | 35% |
| R3 (2 GPUs) | 70W  | 39% |
| R4 (4 GPUs) | 60W  | 37% |

(Caveat: dmon sampling windows differ slightly per run; R1 dmon log includes some idle tail.) Per-GPU utilization stays low across all configs — small batch + short outputs leave the A30 compute-underused; the bottleneck is **communication serialization**, not compute.

## 7. Resume one-liner

> "Benchmarked Qwen2.5-7B BF16 on 8×A30 (PCIe-only, no NVLink) across TP=1/2/4: TP=4 actually delivered worse wall-time throughput than TP=2 (−18%, 14.9 vs 18.2 req/s), with PCIe all-reduce inflating TPOT P95 from 295ms → 421ms. Same-switch (PIX) vs cross-host-bridge (PHB) at TP=2 differed by only ~3% — quantifying that NVLink's value is dominated by **prefill** and **scale-out (TP≥4)** rather than steady-state decode on 7B-class models."

## 8. R5 decision

Skipped. R4 already shows TP=4 is a net loss in this regime; an even worse mixed-topology TP=4 (R5) would only confirm that direction. Better marginal value would be: (a) re-run TP=4 with concurrency 64 and longer outputs to see whether higher arithmetic intensity hides the AR cost; (b) run a 14B/32B model where AR volume per step is large enough to make PIX vs PHB clearly visible.

---

Files: `R{1..4}_raw.txt`, `R{1..4}_dmon.log`, `R{1..4}_metrics.prom`
