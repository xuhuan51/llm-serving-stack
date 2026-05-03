import argparse
import asyncio
import statistics
import time
from openai import AsyncOpenAI


SHORT_PROMPTS = [
    "请用一句话解释什么是机器学习",
    "请用一句话解释什么是深度学习",
    "请用一句话解释什么是神经网络",
    "请用一句话解释什么是注意力机制",
    "请用一句话解释什么是向量数据库",
]


LONG_PROMPTS = [
    "请用不少于500字详细解释什么是机器学习，分成5个要点，每个要点都要包含一个例子。",
    "请用不少于500字详细解释什么是深度学习，分成5个要点，并说明它和传统机器学习的区别。",
    "请用不少于500字详细解释什么是神经网络，分成5个要点，并说明训练和推理分别在做什么。",
    "请用不少于500字详细解释什么是注意力机制，分成5个要点，并说明Q、K、V分别有什么作用。",
    "请用不少于500字详细解释什么是向量数据库，分成5个要点，并说明它在RAG系统里的作用。",
]


CONTEXT_PARAGRAPH = (
    "在大模型推理服务中，用户请求会先进入调度队列。服务端需要处理prompt tokens，"
    "为每一层Transformer建立KV cache，然后在decode阶段逐token生成输出。"
    "当并发增加时，continuous batching可以提高GPU利用率，但请求也可能因为排队而增加TTFT。"
    "当上下文长度增加时，prefill计算量和KV cache占用都会增加。"
    "当输出长度增加时，decode阶段会持续占用GPU算力和显存带宽。"
)


def build_long_context_prompt(topic: str, answer_style: str) -> str:
    context = "\n".join(f"{idx}. {CONTEXT_PARAGRAPH}" for idx in range(1, 41))
    return (
        f"下面是一段较长的技术背景材料：\n{context}\n\n"
        f"问题：请基于上面的材料解释{topic}。\n"
        f"要求：{answer_style}"
    )


LONG_CONTEXT_PROMPTS = [
    build_long_context_prompt("LLM推理服务里的prefill阶段", "用3句话回答，保持简洁。"),
    build_long_context_prompt("LLM推理服务里的decode阶段", "用3句话回答，保持简洁。"),
    build_long_context_prompt("KV cache对显存的影响", "用3句话回答，保持简洁。"),
    build_long_context_prompt("continuous batching如何影响吞吐和延迟", "用3句话回答，保持简洁。"),
    build_long_context_prompt("TTFT和TPOT分别代表什么", "用3句话回答，保持简洁。"),
]


LONG_CONTEXT_OUTPUT_PROMPTS = [
    build_long_context_prompt("LLM推理服务里的prefill和decode", "用不少于700字回答，分成6个要点，每个要点都要结合材料解释。"),
    build_long_context_prompt("KV cache和PagedAttention", "用不少于700字回答，分成6个要点，每个要点都要结合材料解释。"),
    build_long_context_prompt("continuous batching和GPU利用率", "用不少于700字回答，分成6个要点，每个要点都要结合材料解释。"),
    build_long_context_prompt("吞吐、延迟、TTFT和TPOT的关系", "用不少于700字回答，分成6个要点，每个要点都要结合材料解释。"),
    build_long_context_prompt("如何设计LLM serving benchmark", "用不少于700字回答，分成6个要点，每个要点都要结合材料解释。"),
]


PROMPT_MODES = {
    "short": SHORT_PROMPTS,
    "long": LONG_PROMPTS,
    "long-output": LONG_PROMPTS,
    "long-context": LONG_CONTEXT_PROMPTS,
    "long-context-output": LONG_CONTEXT_OUTPUT_PROMPTS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES.keys(), default="short")
    return parser.parse_args()


def pick_prompt(i: int, mode: str = "short") -> str:
    prompts = PROMPT_MODES[mode]
    return prompts[i % len(prompts)]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(len(sorted_values) * p / 100)

    if index >= len(sorted_values):
        index = len(sorted_values) - 1

    return sorted_values[index]


async def single_request(client, model: str, prompt: str, max_tokens: int, request_id: int) -> dict:
    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    elapsed = time.perf_counter() - start
    content = response.choices[0].message.content or ""
    tokens = response.usage.completion_tokens if response.usage else 0

    return {"id": request_id, "elapsed": elapsed, "tokens": tokens, "content": content}


async def streaming_request(client, model: str, prompt: str, max_tokens: int, request_id: int) -> dict:
    start = time.perf_counter()
    first_token_time = None
    chunks = 0
    content_parts = []

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
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
        content_parts.append(piece)

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed
    tpot = (elapsed - ttft) / (chunks - 1) if chunks > 1 else 0.0

    return {
        "id": request_id,
        "elapsed": elapsed,
        "tokens": chunks,
        "content": "".join(content_parts),
        "ttft": ttft,
        "tpot": tpot,
    }


async def run_one(client, args, semaphore, request_id: int) -> dict:
    async with semaphore:
        request_fn = streaming_request if getattr(args, "stream", False) else single_request
        return await request_fn(
            client=client,
            model=args.model,
            prompt=pick_prompt(request_id, getattr(args, "prompt_mode", "short")),
            max_tokens=args.max_tokens,
            request_id=request_id,
        )


def summarize_results(results: list[dict], total_elapsed: float) -> dict:
    latencies = [r["elapsed"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    ttfts = [r["ttft"] for r in results if "ttft" in r]
    tpots = [r["tpot"] for r in results if "tpot" in r]

    metrics = {
        "total_time": total_elapsed,
        "total_requests": len(results),
        "total_tokens": total_tokens,
        "throughput": total_tokens / total_elapsed if total_elapsed > 0 else 0.0,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency": percentile(latencies, 50),
        "p95_latency": percentile(latencies, 95),
        "p99_latency": percentile(latencies, 99),
    }

    if ttfts:
        metrics.update(
            {
                "avg_ttft": statistics.mean(ttfts),
                "p50_ttft": percentile(ttfts, 50),
                "p95_ttft": percentile(ttfts, 95),
                "avg_tpot": statistics.mean(tpots) if tpots else 0.0,
                "p50_tpot": percentile(tpots, 50),
                "p95_tpot": percentile(tpots, 95),
            }
        )

    return metrics


def print_summary(metrics: dict) -> None:
    print("-" * 50)
    print(f"总耗时: {metrics['total_time']:.2f}s")
    print(f"总请求数: {metrics['total_requests']}")
    print(f"总 tokens: {metrics['total_tokens']}")
    print(f"吞吐量: {metrics['throughput']:.1f} tokens/s")
    print(f"平均延迟: {metrics['avg_latency']:.2f}s")
    print(f"50th 百分位延迟: {metrics['p50_latency']:.2f}s")
    print(f"95th 百分位延迟: {metrics['p95_latency']:.2f}s")
    print(f"99th 百分位延迟: {metrics['p99_latency']:.2f}s")
    if "avg_ttft" in metrics:
        print(f"平均 TTFT: {metrics['avg_ttft']:.2f}s")
        print(f"50th 百分位 TTFT: {metrics['p50_ttft']:.2f}s")
        print(f"95th 百分位 TTFT: {metrics['p95_ttft']:.2f}s")
        print(f"平均 TPOT: {metrics['avg_tpot']:.3f}s/token")
        print(f"50th 百分位 TPOT: {metrics['p50_tpot']:.3f}s/token")
        print(f"95th 百分位 TPOT: {metrics['p95_tpot']:.3f}s/token")
        print("注意: streaming 模式下 tokens 由非空内容 chunk 近似统计")
    print("-" * 50)


async def benchmark(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")

    print(f"\n并发数: {args.concurrency} 个请求同时发送")
    print(f"请求模式: {'streaming' if getattr(args, 'stream', False) else 'non-streaming'}")
    print(f"Prompt 模式: {getattr(args, 'prompt_mode', 'short')}")
    for i in range(args.warmup_requests):
        warmup_fn = streaming_request if getattr(args, "stream", False) else single_request
        await warmup_fn(
            client=client,
            model=args.model,
            prompt=pick_prompt(i, getattr(args, "prompt_mode", "short")),
            max_tokens=args.max_tokens,
            request_id=i,
        )

    print(f"Benchmark:{args.num_requests} requests with concurrency {args.concurrency}")    

    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    tasks = [run_one(client, args, semaphore, i) for i in range(args.num_requests)]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - start

    metrics = summarize_results(results, total_elapsed)
    print_summary(metrics)
    return metrics
    
async def main():
    args = parse_args()
    await benchmark(args)


if __name__ == "__main__":
    asyncio.run(main())
