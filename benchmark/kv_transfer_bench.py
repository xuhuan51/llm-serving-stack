"""
KV transfer microbenchmark: GPU4 -> GPU5 cudaMemcpyPeer bandwidth test.

Simulates the cost of transferring KV cache tensors from a prefill worker
to a decode worker in a disaggregated serving setup.

Measures: latency (ms) and bandwidth (GB/s) for 128MB / 256MB / 512MB tensors,
across N repetitions, reporting min/p50/p95/max.
"""

import argparse
import statistics
import time

import torch


def percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def run_transfer(src_gpu: int, dst_gpu: int, size_mb: int, reps: int) -> None:
    size_bytes = size_mb * 1024 * 1024
    n_elements = size_bytes // 4  # float32

    src = torch.zeros(n_elements, dtype=torch.float32, device=f"cuda:{src_gpu}")
    dst = torch.zeros(n_elements, dtype=torch.float32, device=f"cuda:{dst_gpu}")

    # check peer access
    can_access = torch.cuda.can_device_access_peer(dst_gpu, src_gpu)
    if can_access:
        torch.cuda.set_device(dst_gpu)

    # warmup
    for _ in range(4):
        dst.copy_(src)
    torch.cuda.synchronize(src_gpu)
    torch.cuda.synchronize(dst_gpu)

    latencies = []
    for _ in range(reps):
        torch.cuda.synchronize(src_gpu)
        torch.cuda.synchronize(dst_gpu)
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize(dst_gpu)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    bw_list = [size_bytes / (lat / 1000) / 1e9 for lat in latencies]  # GB/s

    print(f"\n  GPU{src_gpu}->GPU{dst_gpu}  {size_mb:4d}MB  peer={can_access}  reps={reps}")
    print(f"    {'metric':<10} {'min':>8} {'p50':>8} {'p95':>8} {'max':>8} {'avg':>8}")
    print(f"    {'-'*54}")
    print(f"    {'latency':<10} "
          f"{min(latencies):>7.2f}ms "
          f"{percentile(latencies,50):>7.2f}ms "
          f"{percentile(latencies,95):>7.2f}ms "
          f"{max(latencies):>7.2f}ms "
          f"{statistics.mean(latencies):>7.2f}ms")
    print(f"    {'bandwidth':<10} "
          f"{min(bw_list):>7.2f}GB "
          f"{percentile(bw_list,50):>7.2f}GB "
          f"{percentile(bw_list,95):>7.2f}GB "
          f"{max(bw_list):>7.2f}GB "
          f"{statistics.mean(bw_list):>7.2f}GB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=int, default=4, help="source GPU index (prefill worker)")
    p.add_argument("--dst", type=int, default=5, help="dest GPU index (decode worker)")
    p.add_argument("--sizes", default="128,256,512", help="tensor sizes in MB")
    p.add_argument("--reps", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    print(f"KV transfer microbenchmark  src=GPU{args.src}  dst=GPU{args.dst}")
    print(f"sizes={sizes}MB  reps={args.reps}")

    for mb in sizes:
        run_transfer(args.src, args.dst, mb, args.reps)

    print("\n--- context ---")
    print("A typical Qwen2.5-7B KV cache per token per layer: ~0.5KB (fp16, 28 layers, 8 heads)")
    print("For 4096-token context: ~4096 * 28 * 2 * 128 * 2B = ~57MB")
    print("For 8192-token context: ~114MB")
    print("Bandwidth above tells you how long KV transfer takes in a PD-disaggregated system.")
