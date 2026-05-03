import argparse
import asyncio
from types import SimpleNamespace

from benchmark_v1 import benchmark
from sweep_concurrency import parse_concurrency_list


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            parsed = int(item)
            if parsed <= 0:
                raise argparse.ArgumentTypeError("values must be positive")
            values.append(parsed)

    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")

    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--concurrency-list", type=parse_concurrency_list, default=parse_concurrency_list("1,2,4,8"))
    parser.add_argument("--max-tokens-list", type=parse_int_list, default=parse_int_list("50,100,200"))
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--warmup-requests", type=int, default=1)
    return parser.parse_args()


def print_matrix(rows: list[dict]) -> None:
    print("\nBenchmark matrix")
    print("-" * 105)
    print(
        f"{'tokens':>7} "
        f"{'conc':>6} "
        f"{'reqs':>6} "
        f"{'out_tok':>8} "
        f"{'time(s)':>9} "
        f"{'tok/s':>9} "
        f"{'avg(s)':>8} "
        f"{'p50(s)':>8} "
        f"{'p95(s)':>8} "
        f"{'p99(s)':>8}"
    )
    print("-" * 105)

    for row in rows:
        print(
            f"{row['max_tokens']:>7} "
            f"{row['concurrency']:>6} "
            f"{row['total_requests']:>6} "
            f"{row['total_tokens']:>8} "
            f"{row['total_time']:>9.2f} "
            f"{row['throughput']:>9.1f} "
            f"{row['avg_latency']:>8.2f} "
            f"{row['p50_latency']:>8.2f} "
            f"{row['p95_latency']:>8.2f} "
            f"{row['p99_latency']:>8.2f}"
        )

    print("-" * 105)


async def main():
    args = parse_args()
    rows = []

    for max_tokens in args.max_tokens_list:
        for concurrency in args.concurrency_list:
            run_args = SimpleNamespace(
                base_url=args.base_url,
                model=args.model,
                concurrency=concurrency,
                num_requests=args.num_requests,
                max_tokens=max_tokens,
                warmup_requests=args.warmup_requests,
            )
            metrics = await benchmark(run_args)
            metrics["max_tokens"] = max_tokens
            metrics["concurrency"] = concurrency
            rows.append(metrics)

    print_matrix(rows)


if __name__ == "__main__":
    asyncio.run(main())
