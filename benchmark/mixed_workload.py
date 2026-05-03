"""
Mixed workload baseline for PD separation experiments.

Sends short requests and long-context (prefill-heavy) requests through the same
semaphore so they compete for the same GPU. Reports TTFT and TPOT separately per
request type.

Key comparison:
  baseline mode  -- short requests only, establishes clean TPOT P95
  mixed mode     -- short + long-context, shows how long prefill inflates TPOT P95

Usage:
  # Step 1: baseline (short only)
  python mixed_workload.py --mode baseline --concurrency 16 --num-requests 64

  # Step 2: mixed (70% short, 30% long-context)
  python mixed_workload.py --mode mixed --concurrency 16 --num-requests 64
"""

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

from benchmark_v1 import PROMPT_MODES, percentile, streaming_request


SHORT_MAX_TOKENS = 200      # short output: isolate decode bottleneck from output length
LONG_MAX_TOKENS = 200       # also moderate output: want prefill cost visible but not trivial


def build_request_plan(num_requests: int, short_ratio: float, mode: str) -> list[str]:
    """Return a list of prompt-mode labels for each request slot."""
    if mode == "baseline":
        return ["short"] * num_requests
    n_long = max(1, round(num_requests * (1 - short_ratio)))
    n_short = num_requests - n_long
    # interleave so long requests don't all land at the end
    plan = []
    long_every = max(1, round(1 / (1 - short_ratio)))
    long_count = 0
    for i in range(num_requests):
        if long_count < n_long and (i % long_every == long_every - 1):
            plan.append("long-context")
            long_count += 1
        else:
            plan.append("short")
    # fill any remaining slots
    while len(plan) < num_requests:
        plan.append("short")
    return plan[:num_requests]


async def run_one(client, model: str, rtype: str, rid: int, semaphore: asyncio.Semaphore) -> dict:
    max_tokens = LONG_MAX_TOKENS if rtype == "long-context" else SHORT_MAX_TOKENS
    prompts = PROMPT_MODES[rtype]
    prompt = prompts[rid % len(prompts)]
    async with semaphore:
        result = await streaming_request(client, model, prompt, max_tokens, rid)
    result["rtype"] = rtype
    return result


def summarize_group(results: list[dict], label: str, raw: bool = False) -> None:
    ttfts = [r["ttft"] for r in results]
    tpots = [r["tpot"] for r in results if r["tpot"] > 0]
    elapsed = [r["elapsed"] for r in results]
    tokens = [r["tokens"] for r in results]

    print(f"\n  [{label}]  n={len(results)}  tokens/req avg={statistics.mean(tokens):.0f}")
    print(f"    {'metric':<8}  {'min':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}  {'avg':>8}")
    print(f"    {'-'*80}")

    def row(name, vals):
        ms = [v * 1000 for v in vals]
        print(f"    {name:<8}  "
              f"{min(ms):>7.1f}ms  "
              f"{percentile(vals, 25)*1000:>7.1f}ms  "
              f"{percentile(vals, 50)*1000:>7.1f}ms  "
              f"{percentile(vals, 75)*1000:>7.1f}ms  "
              f"{percentile(vals, 95)*1000:>7.1f}ms  "
              f"{percentile(vals, 99)*1000:>7.1f}ms  "
              f"{max(ms):>7.1f}ms  "
              f"{statistics.mean(vals)*1000:>7.1f}ms")

    row("TTFT", ttfts)
    if tpots:
        row("TPOT", tpots)
    row("E2E", elapsed)

    if raw:
        print(f"\n    {'id':>4}  {'rtype':<14}  {'TTFT(ms)':>9}  {'TPOT(ms)':>9}  {'tokens':>7}  {'E2E(ms)':>9}")
        print(f"    {'-'*58}")
        for r in sorted(results, key=lambda x: x["id"]):
            tpot_ms = r["tpot"] * 1000 if r["tpot"] > 0 else 0.0
            print(f"    {r['id']:>4}  {r['rtype']:<14}  "
                  f"{r['ttft']*1000:>9.1f}  {tpot_ms:>9.1f}  "
                  f"{r['tokens']:>7}  {r['elapsed']*1000:>9.1f}")


async def run(args) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")

    # warmup
    for i in range(args.warmup_requests):
        await streaming_request(client, args.model,
                                PROMPT_MODES["short"][i % len(PROMPT_MODES["short"])],
                                SHORT_MAX_TOKENS, i)

    plan = build_request_plan(args.num_requests, args.short_ratio, args.mode)
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"\nmode={args.mode}  concurrency={args.concurrency}  "
          f"n={args.num_requests}  "
          f"short={plan.count('short')}  long-context={plan.count('long-context')}")

    start = time.perf_counter()
    tasks = [run_one(client, args.model, rtype, rid, semaphore)
             for rid, rtype in enumerate(plan)]
    results = await asyncio.gather(*tasks)
    total = time.perf_counter() - start

    print(f"  total wall time: {total:.2f}s")

    short_results = [r for r in results if r["rtype"] == "short"]
    long_results = [r for r in results if r["rtype"] == "long-context"]

    if short_results:
        summarize_group(short_results, "short", raw=args.raw)
    if long_results:
        summarize_group(long_results, "long-context (prefill-heavy)", raw=args.raw)

    if short_results and long_results:
        short_p95_tpot = percentile([r["tpot"] for r in short_results if r["tpot"] > 0], 95)
        short_p95_ttft = percentile([r["ttft"] for r in short_results], 95)
        print(f"\n  >> short P95 TPOT={short_p95_tpot*1000:.1f}ms  P95 TTFT={short_p95_ttft*1000:.0f}ms "
              f"(compare with baseline to quantify prefill interference)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--short-ratio", type=float, default=0.7,
                   help="fraction of short requests in mixed mode (default 0.7)")
    p.add_argument("--warmup-requests", type=int, default=2)
    p.add_argument("--mode", choices=["baseline", "mixed"], default="baseline")
    p.add_argument("--raw", action="store_true", help="print per-request TTFT/TPOT/tokens")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
