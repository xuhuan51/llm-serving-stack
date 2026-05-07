"""
V4 — Prometheus metrics poller.

每 N 秒拉一次 vLLM /metrics，落地为 JSONL，便于事后画时间序列对照 P95 抖动。
关键 metric：
  - vllm:num_requests_running    当前 batch 大小（最关键）
  - vllm:num_requests_waiting    排队请求
  - vllm:gpu_cache_usage_perc    KV cache 利用率
  - vllm:time_to_first_token_seconds_bucket  TTFT 分位数（histogram）
  - vllm:time_per_output_token_seconds_bucket TPOT 分位数

用法：
  python v4_metrics_poller.py \\
    --url http://localhost:8000/metrics \\
    --interval 1 \\
    --output results/p1_4_workload_jitter/RT1_Q1_metrics.jsonl

Ctrl+C 优雅退出。
"""
import argparse
import json
import signal
import sys
import time
import urllib.request


# 关心的 metric 名前缀（histogram 也会带 _bucket / _sum / _count）
WANTED_PREFIXES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    # prefix cache（如果开启）
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hits_total",
)


def parse_metric_line(line):
    """简单解析一行 OpenMetrics 文本。返回 (name, labels, value) 或 None。

    例如：vllm:num_requests_running{model_name="qwen"} 5.0
    """
    if not line or line.startswith("#"):
        return None
    # split on first space (or last space if value has spaces)
    # OpenMetrics 格式：<metric_name>{labels} <value> [timestamp]
    if "{" in line:
        idx = line.index("{")
        name = line[:idx]
        rest = line[idx:]
        end_brace = rest.index("}")
        labels_str = rest[1:end_brace]
        value_str = rest[end_brace + 1:].strip().split()[0]
    else:
        parts = line.split()
        if len(parts) < 2:
            return None
        name = parts[0]
        labels_str = ""
        value_str = parts[1]

    try:
        value = float(value_str)
    except ValueError:
        return None

    return name, labels_str, value


def fetch_and_filter(url):
    """拉一次 metrics endpoint，过滤出关心的 metric，返回 dict。"""
    with urllib.request.urlopen(url, timeout=5) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    out = {}
    for line in text.splitlines():
        parsed = parse_metric_line(line)
        if parsed is None:
            continue
        name, labels, value = parsed
        if not any(name.startswith(prefix) for prefix in WANTED_PREFIXES):
            continue
        # 多 label 时聚合到 list
        key = f"{name}{{{labels}}}" if labels else name
        out[key] = value
    return out


_stop = False


def _handle_sigint(*_):
    global _stop
    _stop = True
    print("\n[poller] received SIGINT, stopping...", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/metrics")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seconds", type=int, default=0,
                   help="auto-stop after N seconds (0 = run until SIGINT)")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    start = time.time()
    n = 0
    with open(args.output, "w") as f:
        print(f"[poller] writing to {args.output}, interval={args.interval}s", file=sys.stderr)
        while not _stop:
            try:
                metrics = fetch_and_filter(args.url)
                rec = {"ts": time.time(), "metrics": metrics}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
            except Exception as e:
                f.write(json.dumps({"ts": time.time(), "error": str(e)}) + "\n")
                f.flush()
            if args.max_seconds and (time.time() - start) >= args.max_seconds:
                break
            time.sleep(args.interval)
    print(f"[poller] wrote {n} samples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
