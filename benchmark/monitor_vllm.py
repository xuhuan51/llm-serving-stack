import argparse
import csv
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import StringIO


METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class Diagnosis:
    name: str
    severity: str
    reasons: list[str]
    actions: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny vLLM /metrics and GPU monitor")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi polling")
    parser.add_argument(
        "--gpu-index",
        type=parse_gpu_indexes,
        default=None,
        help="Comma-separated host GPU indexes to include, for example: 1 or 1,3",
    )
    parser.add_argument("--max-num-seqs", type=float, default=None, help="Configured vLLM max_num_seqs")
    parser.add_argument("--high-ttft", type=float, default=0.5, help="P95 TTFT threshold in seconds")
    parser.add_argument("--high-tpot", type=float, default=0.05, help="P95 TPOT threshold in seconds/token")
    parser.add_argument("--high-queue-ratio", type=float, default=0.2)
    parser.add_argument("--high-kv-cache", type=float, default=0.8)
    parser.add_argument("--high-gpu-util", type=float, default=90.0)
    return parser.parse_args()


def parse_gpu_indexes(value: str) -> set[str]:
    indexes = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise argparse.ArgumentTypeError("gpu indexes must be non-negative integers")
        indexes.add(item)

    if not indexes:
        raise argparse.ArgumentTypeError("gpu index list cannot be empty")

    return indexes


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}

    labels = {}
    reader = csv.reader(StringIO(raw))
    for row in reader:
        for item in row:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            labels[key.strip()] = value.strip().strip('"')
    return labels


def parse_value(raw: str) -> float:
    if raw == "+Inf":
        return math.inf
    if raw == "-Inf":
        return -math.inf
    if raw == "NaN":
        return math.nan
    return float(raw)


def parse_prometheus(text: str) -> list[Sample]:
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = METRIC_LINE_RE.match(line)
        if not match:
            continue

        samples.append(
            Sample(
                name=match.group("name"),
                labels=parse_labels(match.group("labels")),
                value=parse_value(match.group("value")),
            )
        )
    return samples


def scrape_metrics(url: str, timeout: float = 5.0) -> list[Sample]:
    request = urllib.request.Request(url, headers={"User-Agent": "learn-ai-infra-monitor/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return parse_prometheus(body)


def first_metric(samples: list[Sample], predicate) -> float | None:
    for sample in samples:
        if predicate(sample):
            return sample.value
    return None


def metric_name_has(sample: Sample, *parts: str) -> bool:
    name = sample.name.lower()
    return all(part.lower() in name for part in parts)


def collect_counters(samples: list[Sample]) -> dict[str, float]:
    counters = {}
    for sample in samples:
        if sample.name.endswith("_total") and math.isfinite(sample.value):
            counters[sample.name] = counters.get(sample.name, 0.0) + sample.value
    return counters


def counter_rate(
    name_fragment: str,
    previous: dict[str, float],
    current: dict[str, float],
    interval: float,
) -> float | None:
    matched = [
        name
        for name in set(previous) | set(current)
        if name_fragment.lower() in name.lower() and name.endswith("_total")
    ]
    if not matched:
        return None

    delta = sum(current.get(name, 0.0) - previous.get(name, 0.0) for name in matched)
    if delta < 0:
        return None
    return delta / interval if interval > 0 else None


def collect_buckets(samples: list[Sample]) -> dict[tuple[str, str, float], float]:
    buckets = {}
    for sample in samples:
        if not sample.name.endswith("_bucket"):
            continue
        le = sample.labels.get("le")
        if le is None:
            continue
        if le == "+Inf":
            upper = math.inf
        else:
            try:
                upper = float(le)
            except ValueError:
                continue
        label_key = repr(tuple(sorted((k, v) for k, v in sample.labels.items() if k != "le")))
        key = (sample.name, label_key, upper)
        buckets[key] = buckets.get(key, 0.0) + sample.value
    return buckets


def histogram_quantile_from_deltas(
    name_fragment: str,
    previous: dict[tuple[str, str, float], float],
    current: dict[tuple[str, str, float], float],
    quantile: float,
) -> float | None:
    per_upper: dict[float, float] = {}
    for key in set(previous) | set(current):
        metric_name, _label_key, upper = key
        if name_fragment.lower() not in metric_name.lower():
            continue
        delta = current.get(key, 0.0) - previous.get(key, 0.0)
        if delta < 0:
            return None
        per_upper[upper] = per_upper.get(upper, 0.0) + delta

    if not per_upper:
        return None

    ordered = sorted(per_upper.items(), key=lambda item: item[0])
    total = ordered[-1][1]
    if total <= 0:
        return None

    target = total * quantile
    previous_upper = 0.0
    previous_count = 0.0
    for upper, cumulative in ordered:
        if cumulative >= target:
            if math.isinf(upper):
                return previous_upper
            bucket_count = cumulative - previous_count
            if bucket_count <= 0:
                return upper
            position = (target - previous_count) / bucket_count
            return previous_upper + (upper - previous_upper) * position
        previous_upper = upper
        previous_count = cumulative

    return None


def read_gpu_stats() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5.0)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    rows = []
    reader = csv.reader(StringIO(completed.stdout))
    for row in reader:
        if len(row) < 8:
            continue
        rows.append(
            {
                "index": row[0].strip(),
                "name": row[1].strip(),
                "gpu_util": row[2].strip(),
                "mem_util": row[3].strip(),
                "mem_used": row[4].strip(),
                "mem_total": row[5].strip(),
                "power": row[6].strip(),
                "temp": row[7].strip(),
            }
        )
    return rows


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def fmt_rate(value: float | None, unit: str) -> str:
    if value is None:
        return f"n/a {unit}"
    return f"{value:.1f} {unit}"


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def parse_optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def max_gpu_util(gpu_rows: list[dict[str, str]]) -> float | None:
    values = [parse_optional_float(row["gpu_util"]) for row in gpu_rows]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def filter_gpu_rows(
    gpu_rows: list[dict[str, str]],
    gpu_indexes: set[str] | None,
) -> list[dict[str, str]]:
    if gpu_indexes is None:
        return gpu_rows
    return [row for row in gpu_rows if row["index"] in gpu_indexes]


def build_diagnosis(
    running: float | None,
    waiting: float | None,
    swapped: float | None,
    kv_usage: float | None,
    ttft_p95: float | None,
    tpot_p95: float | None,
    queue_ratio: float | None,
    running_saturation: float | None,
    gpu_util: float | None,
    args: argparse.Namespace,
) -> Diagnosis:
    reasons = []
    actions = []

    ttft_high = ttft_p95 is not None and ttft_p95 >= args.high_ttft
    tpot_high = tpot_p95 is not None and tpot_p95 >= args.high_tpot
    queue_high = (
        waiting is not None
        and waiting > 0
        and (queue_ratio is None or queue_ratio >= args.high_queue_ratio)
    )
    running_limited = running_saturation is not None and running_saturation >= 0.95
    kv_high = kv_usage is not None and kv_usage >= args.high_kv_cache
    swapped_high = swapped is not None and swapped > 0
    gpu_high = gpu_util is not None and gpu_util >= args.high_gpu_util

    if swapped_high or kv_high:
        if swapped_high:
            reasons.append(f"swapped={swapped:.0f} > 0")
        if kv_high:
            reasons.append(f"kv/cache usage={kv_usage:.2f} >= {args.high_kv_cache:.2f}")
        if ttft_high:
            reasons.append(f"p95 TTFT={ttft_p95:.3f}s is high")
        actions.extend(
            [
                "reduce max_num_seqs or client concurrency",
                "lower max_model_len or limit long-context requests",
                "use more GPU memory only if there is safe headroom",
            ]
        )
        return Diagnosis("kv_cache_or_memory_pressure", "critical", reasons, actions)

    if queue_high and ttft_high and not tpot_high:
        reasons.append(f"waiting={waiting:.0f}, queue_ratio={fmt_ratio(queue_ratio)}")
        reasons.append(f"p95 TTFT={ttft_p95:.3f}s is high while p95 TPOT is normal")
        if running_limited:
            reasons.append(f"running saturation={running_saturation:.2f}")
        actions.extend(
            [
                "increase max_num_seqs if KV cache and GPU memory allow",
                "reduce client concurrency or add another serving replica",
                "check whether long prefill requests are blocking short requests",
            ]
        )
        return Diagnosis("scheduler_queueing", "warning", reasons, actions)

    if ttft_high and tpot_high:
        reasons.append(f"p95 TTFT={ttft_p95:.3f}s is high")
        reasons.append(f"p95 TPOT={tpot_p95:.3f}s is high")
        if queue_high:
            reasons.append(f"waiting={waiting:.0f}, queue_ratio={fmt_ratio(queue_ratio)}")
        if gpu_high:
            reasons.append(f"max GPU util={gpu_util:.0f}% is high")
        actions.extend(
            [
                "treat this as overall saturation",
                "reduce traffic pressure or add replicas",
                "split diagnosis by testing lower concurrency and shorter outputs",
            ]
        )
        return Diagnosis("overall_saturation", "critical", reasons, actions)

    if tpot_high and gpu_high and not queue_high:
        reasons.append(f"p95 TPOT={tpot_p95:.3f}s is high")
        reasons.append(f"max GPU util={gpu_util:.0f}% is high")
        if ttft_p95 is not None:
            reasons.append(f"p95 TTFT={ttft_p95:.3f}s")
        actions.extend(
            [
                "reduce output length or client concurrency",
                "add serving capacity or use a faster model/quantization",
                "compare tokens/s before and after tuning",
            ]
        )
        return Diagnosis("decode_compute_bottleneck", "warning", reasons, actions)

    if ttft_high and not tpot_high:
        reasons.append(f"p95 TTFT={ttft_p95:.3f}s is high while p95 TPOT is normal")
        if waiting is not None:
            reasons.append(f"waiting={waiting:.0f}")
        actions.extend(
            [
                "check prompt length, prompt token rate, and chunked prefill settings",
                "check scheduler waiting if traffic is bursty",
            ]
        )
        return Diagnosis("prefill_or_queueing_pressure", "warning", reasons, actions)

    if queue_high:
        reasons.append(f"waiting={waiting:.0f}, queue_ratio={fmt_ratio(queue_ratio)}")
        if running_limited:
            reasons.append(f"running saturation={running_saturation:.2f}")
        actions.extend(
            [
                "watch TTFT and e2e latency for queueing impact",
                "compare client concurrency with max_num_seqs",
            ]
        )
        return Diagnosis("queue_building", "info", reasons, actions)

    reasons.append("no strong bottleneck pattern matched")
    actions.append("keep watching p95 TTFT, p95 TPOT, waiting, GPU util, and KV cache")
    return Diagnosis("healthy_or_insufficient_signal", "info", reasons, actions)


def print_diagnosis(diagnosis: Diagnosis) -> None:
    print(
        "  diagnosis "
        f"name={diagnosis.name} severity={diagnosis.severity}"
    )
    print(f"   reason: {'; '.join(diagnosis.reasons)}")
    print(f"   action: {'; '.join(diagnosis.actions)}")


def print_snapshot(
    samples: list[Sample],
    previous_counters: dict[str, float] | None,
    previous_buckets: dict[tuple[str, str, float], float] | None,
    interval: float,
    gpu_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[tuple[str, str, float], float]]:
    counters = collect_counters(samples)
    buckets = collect_buckets(samples)

    running = first_metric(samples, lambda s: metric_name_has(s, "num_requests_running"))
    waiting = first_metric(samples, lambda s: metric_name_has(s, "num_requests_waiting"))
    swapped = first_metric(samples, lambda s: metric_name_has(s, "num_requests_swapped"))
    kv_usage = first_metric(samples, lambda s: metric_name_has(s, "cache_usage"))

    prompt_rate = None
    generation_rate = None
    request_rate = None
    ttft_p95 = None
    tpot_p95 = None
    latency_p95 = None

    if previous_counters is not None:
        prompt_rate = counter_rate("prompt_tokens", previous_counters, counters, interval)
        generation_rate = counter_rate("generation_tokens", previous_counters, counters, interval)
        request_rate = counter_rate("request", previous_counters, counters, interval)

    if previous_buckets is not None:
        ttft_p95 = histogram_quantile_from_deltas("time_to_first_token", previous_buckets, buckets, 0.95)
        tpot_p95 = histogram_quantile_from_deltas("time_per_output_token", previous_buckets, buckets, 0.95)
        latency_p95 = histogram_quantile_from_deltas("e2e_request_latency", previous_buckets, buckets, 0.95)

    active_requests = None
    if running is not None or waiting is not None:
        active_requests = (running or 0.0) + (waiting or 0.0)
    queue_ratio = safe_ratio(waiting, active_requests)
    running_saturation = safe_ratio(running, args.max_num_seqs)
    gpu_util = max_gpu_util(gpu_rows)

    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] vLLM")
    print(
        "  requests "
        f"running={running if running is not None else 'n/a'} "
        f"waiting={waiting if waiting is not None else 'n/a'} "
        f"swapped={swapped if swapped is not None else 'n/a'}"
    )
    print(
        "  rates "
        f"qps={fmt_rate(request_rate, 'req/s')} "
        f"prompt={fmt_rate(prompt_rate, 'tok/s')} "
        f"output={fmt_rate(generation_rate, 'tok/s')}"
    )
    print(
        "  p95 "
        f"ttft={fmt_seconds(ttft_p95)} "
        f"tpot={fmt_seconds(tpot_p95)} "
        f"e2e={fmt_seconds(latency_p95)}"
    )
    print(f"  kv/cache usage={kv_usage if kv_usage is not None else 'n/a'}")
    print(
        "  derived "
        f"queue_ratio={fmt_ratio(queue_ratio)} "
        f"running_saturation={fmt_ratio(running_saturation)}"
    )

    if gpu_rows:
        print("  GPU")
        for row in gpu_rows:
            print(
                "   "
                f"gpu{row['index']} {row['name']} "
                f"util={row['gpu_util']}% mem_util={row['mem_util']}% "
                f"mem={row['mem_used']}/{row['mem_total']} MiB "
                f"power={row['power']}W temp={row['temp']}C"
            )

    diagnosis = build_diagnosis(
        running=running,
        waiting=waiting,
        swapped=swapped,
        kv_usage=kv_usage,
        ttft_p95=ttft_p95,
        tpot_p95=tpot_p95,
        queue_ratio=queue_ratio,
        running_saturation=running_saturation,
        gpu_util=gpu_util,
        args=args,
    )
    print_diagnosis(diagnosis)

    return counters, buckets


def main() -> None:
    args = parse_args()
    previous_counters = None
    previous_buckets = None

    print(f"Scraping {args.metrics_url}")
    print("Press Ctrl+C to stop.")

    iteration = 0
    while args.iterations <= 0 or iteration < args.iterations:
        start = time.perf_counter()
        try:
            samples = scrape_metrics(args.metrics_url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"\nmetrics scrape failed: {exc}")
            time.sleep(args.interval)
            iteration += 1
            continue

        gpu_rows = [] if args.no_gpu else filter_gpu_rows(read_gpu_stats(), args.gpu_index)
        previous_counters, previous_buckets = print_snapshot(
            samples=samples,
            previous_counters=previous_counters,
            previous_buckets=previous_buckets,
            interval=args.interval,
            gpu_rows=gpu_rows,
            args=args,
        )

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, args.interval - elapsed)
        time.sleep(sleep_time)
        iteration += 1


if __name__ == "__main__":
    main()
