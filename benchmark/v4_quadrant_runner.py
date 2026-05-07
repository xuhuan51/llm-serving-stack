"""
V4 — 4-Quadrant workload jitter runner.

设计目标：在固定并发下用 4 种 (input-len, output-len) 组合压测同一个 vLLM 实例，
分位数指标按象限分别统计，定位 P95 抖动到具体 workload 类型。

4 象限：
  Q1 short-short   : 短 prompt + 短输出 (API 调用)
  Q2 short-long    : 短 prompt + 长输出 (生成式)
  Q3 long-short    : 长 prompt + 短输出 (RAG)
  Q4 long-long     : 长 prompt + 长输出 (深度对话)

控制变量：
  --concurrency / --num-requests / --warmup
  --ignore-eos 强制生成到 max_tokens（让输出长度可控可比）

用法：
  python v4_quadrant_runner.py \\
    --base-url http://localhost:8000/v1 --model qwen \\
    --quadrant Q1 --concurrency 16 --num-requests 100 \\
    --output-dir results/p1_4_workload_jitter --tag T1
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

from benchmark_v1 import PROMPT_MODES, percentile


# 象限 → (prompt_mode, max_tokens)
QUADRANTS = {
    "Q1": {"prompt_mode": "short", "max_tokens": 50, "label": "short-short (API call)"},
    "Q2": {"prompt_mode": "short", "max_tokens": 1000, "label": "short-long (generation)"},
    "Q3": {"prompt_mode": "long-context", "max_tokens": 50, "label": "long-short (RAG)"},
    "Q4": {"prompt_mode": "long-context", "max_tokens": 1000, "label": "long-long (deep dialog)"},
}


async def streaming_request_v4(client, model, prompt, max_tokens, request_id, ignore_eos):
    """流式请求，返回 ttft / tpot / elapsed / tokens / submit_ts。

    submit_ts 用于事后关联 Grafana 时间轴。
    ignore_eos 通过 extra_body 透传给 vLLM。
    """
    submit_ts = time.time()  # wall clock for Grafana correlation
    start = time.perf_counter()
    first_token_time = None
    chunks = 0

    extra_body = {"ignore_eos": True} if ignore_eos else {}
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        extra_body=extra_body,
    )

    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = delta.content or ""
        if not piece:
            continue
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        chunks += 1

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed
    tpot = (elapsed - ttft) / (chunks - 1) if chunks > 1 else 0.0

    return {
        "id": request_id,
        "submit_ts": submit_ts,
        "elapsed": elapsed,
        "ttft": ttft,
        "tpot": tpot,
        "tokens": chunks,
    }


async def run_one(client, model, prompt, max_tokens, rid, ignore_eos, semaphore):
    async with semaphore:
        return await streaming_request_v4(client, model, prompt, max_tokens, rid, ignore_eos)


def summarize(results, label):
    """打印分位数表，返回 dict 便于 JSON 落地。"""
    ttfts = [r["ttft"] for r in results]
    tpots = [r["tpot"] for r in results if r["tpot"] > 0]
    elapsed = [r["elapsed"] for r in results]
    tokens = [r["tokens"] for r in results]

    print(f"\n  [{label}]  n={len(results)}  avg_tokens/req={statistics.mean(tokens):.0f}")
    header = f"    {'metric':<6}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}  {'avg':>8}"
    print(header)
    print(f"    {'-' * (len(header) - 4)}")

    def row(name, vals):
        if not vals:
            print(f"    {name:<6}  (no data)")
            return None
        ms = [v * 1000 for v in vals]
        out = {
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "max": max(vals),
            "avg": statistics.mean(vals),
        }
        print(f"    {name:<6}  "
              f"{out['p50']*1000:>7.1f}ms  "
              f"{out['p95']*1000:>7.1f}ms  "
              f"{out['p99']*1000:>7.1f}ms  "
              f"{out['max']*1000:>7.1f}ms  "
              f"{out['avg']*1000:>7.1f}ms")
        return out

    return {
        "ttft": row("TTFT", ttfts),
        "tpot": row("TPOT", tpots),
        "e2e": row("E2E", elapsed),
        "n": len(results),
        "avg_tokens": statistics.mean(tokens) if tokens else 0,
    }


async def run(args):
    if args.quadrant not in QUADRANTS:
        raise SystemExit(f"unknown quadrant {args.quadrant}, expect one of {list(QUADRANTS)}")
    q = QUADRANTS[args.quadrant]
    prompts = PROMPT_MODES[q["prompt_mode"]]
    max_tokens = q["max_tokens"]

    print(f"\n=== V4 Quadrant {args.quadrant}: {q['label']} ===")
    print(f"  prompt_mode = {q['prompt_mode']}")
    print(f"  max_tokens = {max_tokens}, ignore_eos = {args.ignore_eos}")
    print(f"  concurrency = {args.concurrency}, num_requests = {args.num_requests}")
    print(f"  base_url = {args.base_url}")

    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")

    # warmup
    print(f"\n  warmup ({args.warmup} requests)...")
    for i in range(args.warmup):
        await streaming_request_v4(client, args.model, prompts[i % len(prompts)],
                                    max_tokens, i, args.ignore_eos)

    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"  benchmarking {args.num_requests} requests...\n")

    wall_start = time.perf_counter()
    wall_start_ts = time.time()
    tasks = [
        run_one(client, args.model, prompts[i % len(prompts)],
                max_tokens, i, args.ignore_eos, semaphore)
        for i in range(args.num_requests)
    ]
    results = await asyncio.gather(*tasks)
    wall_total = time.perf_counter() - wall_start
    wall_end_ts = time.time()

    print(f"  total wall time: {wall_total:.2f}s")
    print(f"  throughput: {args.num_requests / wall_total:.2f} req/s")
    total_tokens = sum(r["tokens"] for r in results)
    print(f"  total tokens generated: {total_tokens}")
    print(f"  token throughput: {total_tokens / wall_total:.1f} tok/s")

    summary = summarize(results, args.quadrant)
    summary.update({
        "wall_time": wall_total,
        "req_per_sec": args.num_requests / wall_total,
        "total_tokens": total_tokens,
        "tok_per_sec": total_tokens / wall_total,
        "wall_start_ts": wall_start_ts,  # for Grafana time-range query
        "wall_end_ts": wall_end_ts,
    })

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        prefix = f"R{args.tag}_{args.quadrant}"
        # JSON 结果（含原始数据 + 分位数）
        with open(outdir / f"{prefix}_result.json", "w") as f:
            json.dump({
                "config": {
                    "tag": args.tag,
                    "quadrant": args.quadrant,
                    "label": q["label"],
                    "prompt_mode": q["prompt_mode"],
                    "max_tokens": max_tokens,
                    "ignore_eos": args.ignore_eos,
                    "concurrency": args.concurrency,
                    "num_requests": args.num_requests,
                    "warmup": args.warmup,
                    "base_url": args.base_url,
                    "model": args.model,
                },
                "summary": summary,
                "raw": results,
            }, f, indent=2)
        print(f"\n  saved: {outdir / f'{prefix}_result.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen")
    p.add_argument("--quadrant", required=True, choices=list(QUADRANTS.keys()))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--ignore-eos", action="store_true",
                   help="force generation to max_tokens for length-controlled comparison")
    p.add_argument("--output-dir", default="")
    p.add_argument("--tag", default="T1",
                   help="run tag, e.g. T1 for TP=1, T2 for TP=2 PIX")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
