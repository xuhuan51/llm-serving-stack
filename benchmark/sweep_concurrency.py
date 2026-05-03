import argparse
import asyncio
from types import SimpleNamespace

from benchmark_v1 import PROMPT_MODES, benchmark


def parse_concurrency_list(value: str) -> list[int]:
    concurrencies = []
    for item in value.split(","):
        item = item.strip()
        if item:
            concurrency = int(item)
            if concurrency <= 0:
                raise argparse.ArgumentTypeError("concurrency must be positive")
            concurrencies.append(concurrency)

    if not concurrencies:
        raise argparse.ArgumentTypeError("concurrency list cannot be empty")

    return concurrencies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--concurrency-list", type=parse_concurrency_list, default=parse_concurrency_list("1,2,4,8"))
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES.keys(), default="short")
    return parser.parse_args()


def print_comparison(rows: list[dict]) -> None:
    has_stream_metrics = any("avg_ttft" in row for row in rows)

    print("\nConcurrency comparison")
    if has_stream_metrics:
        print("-" * 122)
        print(
            f"{'conc':>6} "
            f"{'reqs':>6} "
            f"{'tokens':>8} "
            f"{'time(s)':>9} "
            f"{'tok/s':>9} "
            f"{'avg(s)':>8} "
            f"{'p95(s)':>8} "
            f"{'avg_ttft':>9} "
            f"{'p95_ttft':>9} "
            f"{'avg_tpot':>10} "
            f"{'p95_tpot':>10}"
        )
        print("-" * 122)
    else:
        print("-" * 86)
        print(
            f"{'conc':>6} "
            f"{'reqs':>6} "
            f"{'tokens':>8} "
            f"{'time(s)':>9} "
            f"{'tok/s':>9} "
            f"{'avg(s)':>8} "
            f"{'p50(s)':>8} "
            f"{'p95(s)':>8}"
        )
        print("-" * 86)

    for row in rows:
        if has_stream_metrics:
            print(
                f"{row['concurrency']:>6} "
                f"{row['total_requests']:>6} "
                f"{row['total_tokens']:>8} "
                f"{row['total_time']:>9.2f} "
                f"{row['throughput']:>9.1f} "
                f"{row['avg_latency']:>8.2f} "
                f"{row['p95_latency']:>8.2f} "
                f"{row.get('avg_ttft', 0.0):>9.2f} "
                f"{row.get('p95_ttft', 0.0):>9.2f} "
                f"{row.get('avg_tpot', 0.0):>10.3f} "
                f"{row.get('p95_tpot', 0.0):>10.3f}"
            )
        else:
            print(
                f"{row['concurrency']:>6} "
                f"{row['total_requests']:>6} "
                f"{row['total_tokens']:>8} "
                f"{row['total_time']:>9.2f} "
                f"{row['throughput']:>9.1f} "
                f"{row['avg_latency']:>8.2f} "
                f"{row['p50_latency']:>8.2f} "
                f"{row['p95_latency']:>8.2f}"
            )

    print("-" * (122 if has_stream_metrics else 86))


async def main():
    args = parse_args()
    rows = []

    for concurrency in args.concurrency_list:
        run_args = SimpleNamespace(
            base_url=args.base_url,
            model=args.model,
            concurrency=concurrency,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            warmup_requests=args.warmup_requests,
            stream=args.stream,
            prompt_mode=args.prompt_mode,
        )
        metrics = await benchmark(run_args)
        metrics["concurrency"] = concurrency
        rows.append(metrics)

    print_comparison(rows)


if __name__ == "__main__":
    asyncio.run(main())
